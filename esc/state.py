"""
ESC state representation with causal graph components.

s_t = φ(H_t, G_t) = concat(H̄_t, C̄_t, e_t, p_t) ∈ ℝ^d

Changes from v1
---------------
- CausalGraph is now a first-class field; cause_embeddings are derived
  from it rather than stored as a raw tensor.
- phase_embedding is always a 3-element soft probability vector produced
  by softmax(next_phase_logits) from the TransitionModel.  This makes
  get_phase_index() in reward.py deterministic and meaningful.
- from_dialogue() accepts an optional encoder callable so real backbone
  embeddings can be plugged in without changing the ESCState interface.
- encoder hook signature: encoder(turns: list[str]) -> dict with keys
  "history", "emotion", "causes" (all torch.Tensor).
- clone() now deep-copies the CausalGraph as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional
import torch
import torch.nn.functional as F

from esc.causal_graph import CausalGraph


# Type alias for the optional encoder hook
EncoderFn = Callable[[list[str]], dict[str, torch.Tensor]]


@dataclass
class ESCState:
    """
    Causal dialogue state for the ESC MDP.

    s_t = concat(H̄_t, C̄_t, e_t, p_t) ∈ ℝ^{D_H + D_C + D_E + 3}

    Note on phase_embedding dimension
    ----------------------------------
    Phase is a 3-element softmax probability vector over the three ESC
    phases (Exploration / Comforting / Action Planning).  D_P = 3 so
    the dimension is exact and argmax() is meaningful.

    Attributes
    ----------
    history_embeddings : [K, D_H] — recent dialogue turn encodings
    causal_graph       : CausalGraph — cause nodes, edges, resolution ρ_t
    emotion_vector     : [D_E]    — current emotional state
    phase_embedding    : [3]      — soft phase distribution (sums to 1)
    turn_index         : int      — current turn (for horizon tracking)
    target_emotion     : [D_E]    — desired emotional endpoint (for R_emotion)
                                    Defaults to zeros (calm/neutral).
    """

    history_embeddings: torch.Tensor   # [K, D_H]
    causal_graph: CausalGraph
    emotion_vector: torch.Tensor       # [D_E]
    phase_embedding: torch.Tensor      # [3]  — soft phase probs
    turn_index: int = 0
    # Target emotion: what "resolved" looks like emotionally.
    # Zero vector = calm/neutral in a learned embedding space.
    target_emotion: torch.Tensor = field(default_factory=lambda: torch.zeros(ESCState.D_E))

    # ------------------------------------------------------------------
    # Class-level dimension constants (ClassVar → excluded from __init__)
    # ------------------------------------------------------------------
    K_HISTORY_WINDOW: ClassVar[int] = 5     # Recent turns kept in window
    D_H: ClassVar[int] = 768                # Qwen-9B hidden size
    D_C: ClassVar[int] = 384               # Cause embedding dimension
    D_E: ClassVar[int] = 128               # Emotion vector dimension
    D_P: ClassVar[int] = 3                  # Phase dimension (FIXED: 3 phases)
    N_C: ClassVar[int] = 4                  # Max causes tracked

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Validate shapes at construction time to catch mismatches early
        K, D_H = self.K_HISTORY_WINDOW, self.D_H
        if self.history_embeddings.shape != (K, D_H):
            raise ValueError(
                f"history_embeddings must be [{K}, {D_H}], "
                f"got {list(self.history_embeddings.shape)}"
            )
        if self.emotion_vector.shape != (self.D_E,):
            raise ValueError(
                f"emotion_vector must be [{self.D_E}], "
                f"got {list(self.emotion_vector.shape)}"
            )
        if self.phase_embedding.shape != (3,):
            raise ValueError(
                f"phase_embedding must be [3] (soft phase probs), "
                f"got {list(self.phase_embedding.shape)}"
            )
        if self.target_emotion.shape != (self.D_E,):
            raise ValueError(
                f"target_emotion must be [{self.D_E}], "
                f"got {list(self.target_emotion.shape)}"
            )

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensor(self) -> torch.Tensor:
        """
        Flatten state to a single vector for the policy/value networks.

        Pooling:
          H̄_t = mean(H_t, dim=0)    → [D_H]
          C̄_t = mean(C_t, dim=0)    → [D_C]  (from causal graph)

        Returns
        -------
        torch.Tensor of shape [D_H + D_C + D_E + 3]
        """
        h_pooled = self.history_embeddings.mean(dim=0)                    # [D_H]
        c_matrix = self.causal_graph.cause_embedding_matrix(self.N_C, self.D_C)
        c_pooled = c_matrix.mean(dim=0)                                   # [D_C]

        return torch.cat([
            h_pooled,               # [D_H]
            c_pooled,               # [D_C]
            self.emotion_vector,    # [D_E]
            self.phase_embedding,   # [3]
        ], dim=0)

    @classmethod
    def get_state_dim(cls) -> int:
        """Total state vector dimension d = D_H + D_C + D_E + D_P."""
        return cls.D_H + cls.D_C + cls.D_E + cls.D_P

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dialogue(
        cls,
        turns: list[str],
        encoder: Optional[EncoderFn] = None,
        target_emotion: Optional[torch.Tensor] = None,
    ) -> "ESCState":
        """
        Build an ESCState from raw dialogue turns.

        When encoder is None (default), returns a zero-filled placeholder
        state that is structurally valid and safe to use throughout the
        pipeline.

        When encoder is provided, it is called as::

            result = encoder(turns)

        and must return a dict with keys:
          - "history"  : Tensor [K, D_H]   — turn encodings
          - "emotion"  : Tensor [D_E]       — current emotion vector
          - "causes"   : list[dict] where each dict has:
                           "label"     : str
                           "embedding" : Tensor [D_C]

        Parameters
        ----------
        turns:          List of dialogue strings (alternating speaker turns)
        encoder:        Optional callable that produces real embeddings.
                        Signature: encoder(turns) -> dict[str, Tensor]
        target_emotion: Desired emotional endpoint [D_E]; zeros if None.

        Returns
        -------
        ESCState with either real or zero-filled embeddings.
        """
        if target_emotion is None:
            target_emotion = torch.zeros(cls.D_E)

        if encoder is not None:
            result = encoder(turns)
            history = result["history"]                    # [K, D_H]
            emotion = result["emotion"]                    # [D_E]
            cause_dicts = result.get("causes", [])

            graph = CausalGraph(d_c=cls.D_C)
            for cd in cause_dicts[: cls.N_C]:
                graph.add_cause(
                    label=cd["label"],
                    embedding=cd["embedding"],
                )
            # Pad with anonymous causes if fewer than N_C were extracted
            while graph.num_causes < cls.N_C:
                graph.add_cause(
                    label=f"cause_{graph.num_causes}",
                    embedding=torch.zeros(cls.D_C),
                )
        else:
            # Placeholder: zeros everywhere, anonymous causes
            history = torch.zeros(cls.K_HISTORY_WINDOW, cls.D_H)
            emotion = torch.zeros(cls.D_E)
            graph = CausalGraph.placeholder(cls.N_C, cls.D_C)

        # Initial phase: estimated from conversation depth (number of turns
        # passed in), not a hardcoded uniform prior.
        #
        # 2026-09-12 bug fix: this used to be `torch.ones(3) / 3.0` -- an
        # exact 3-way tie -- for every real state, regardless of `turns`.
        # Since current_phase = phase_embedding.argmax(), and argmax()
        # deterministically resolves an exact tie to index 0, EVERY real
        # state (all 910 training states, all real eval states, at any
        # conversation depth from 2 to 59+ turns -- verified directly)
        # always had current_phase == 0. generate_candidate_actions() then
        # always filtered to PHASE_STRATEGY_MAP[0] == [Question,
        # Restatement, Reflection, Self-disclosure] -- the other 4
        # strategies (Affirmation, Providing Suggestions, Information,
        # Others) were structurally unreachable for every state, not just
        # eval-sampling bias (which eval/data.py's build_eval_instances fix
        # addresses separately -- that fix controls which gold labels get
        # compared against, not what the model's own state thinks its phase
        # is). This is very likely the true root cause of the mode collapse
        # both this run and the original pilot hit: with an unchanging
        # 4-strategy candidate set and highly similar encoded states (see
        # the conversation log), any policy converges to whichever single
        # option scores best on average.
        #
        # ESConv has no ground-truth phase labels, so this is a heuristic,
        # like the rest of this pipeline's state construction (frozen
        # random projections, seeker-turns-as-cause-spans) -- disclosed, not
        # hidden. Thresholds (11, 20 turns) are the 33rd/66th percentile of
        # turn-count over the real eligible eval-instance distribution
        # (measured 2026-09-12, n=2610, median 15.5, mean 17.2, range
        # 2-59), so each phase covers roughly a third of real conversations
        # rather than using round numbers picked without looking at data.
        # Only usable signal at inference time is turns seen so far, not
        # future conversation length, which this respects.
        n_turns = len(turns)
        if n_turns <= 11:
            phase_idx = 0
        elif n_turns <= 20:
            phase_idx = 1
        else:
            phase_idx = 2
        phase = torch.zeros(3)
        phase[phase_idx] = 1.0

        return cls(
            history_embeddings=history,
            causal_graph=graph,
            emotion_vector=emotion,
            phase_embedding=phase,
            turn_index=0,
            target_emotion=target_emotion,
        )

    def clone(self) -> "ESCState":
        """
        Deep copy this state for MCTS tree expansion.

        All tensors and the causal graph are cloned so that mutations in
        one branch of the search tree do not affect another.

        Returns
        -------
        New ESCState with independent tensor storage and a copied CausalGraph.
        """
        # Deep-copy the causal graph
        new_graph = CausalGraph(d_c=self.causal_graph.d_c)
        for i in range(self.causal_graph.num_causes):
            node = self.causal_graph.get_node(i)
            new_graph.add_cause(
                label=node.label,
                embedding=node.embedding.clone(),
                initial_resolution=node.resolution,
            )
            for dst in self.causal_graph.get_edges(i):
                try:
                    new_graph.add_edge(i, dst)
                except (IndexError, ValueError):
                    pass  # edge may already exist if graph has cycles

        return ESCState(
            history_embeddings=self.history_embeddings.clone(),
            causal_graph=new_graph,
            emotion_vector=self.emotion_vector.clone(),
            phase_embedding=self.phase_embedding.clone(),
            turn_index=self.turn_index,
            target_emotion=self.target_emotion.clone(),
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> int:
        """
        Discrete phase index (0=Exploration, 1=Comforting, 2=Action).

        Derived from argmax of the soft phase distribution.  Because
        phase_embedding is always a proper softmax output, this is
        always well-defined.
        """
        return int(self.phase_embedding.argmax().item())

    @property
    def resolution_tensor(self) -> torch.Tensor:
        """Per-cause resolution probabilities ρ_t, shape [n_c]."""
        return self.causal_graph.resolution_tensor()

    def __repr__(self) -> str:
        return (
            f"ESCState(turn={self.turn_index}, phase={self.current_phase}, "
            f"graph={self.causal_graph})"
        )
