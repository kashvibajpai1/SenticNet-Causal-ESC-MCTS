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

        Aggregation (changed 2026-09-15, revised same day -- see debug spec
        Steps 1b/2 in the conversation log):

        Original: mean-pooled across all K history rows and all N_C cause
        rows. Diagnosed as the cause of severe state-vector collapse
        (pairwise cosine similarity ~0.94-0.97 on genuinely diverse real
        conversations, vs ~0.6 for individual turn embeddings) --
        history_embeddings' window mixes topic-specific seeker turns with
        generic, similarly-phrased supporter turns, and averaging lets the
        cross-conversation-similar supporter turns dilute the
        conversation-specific seeker content; similarly for cause spans of
        uneven salience.

        First revision (select-only, superseded): replaced the mean with
        selecting a single row (most recent history turn, most salient
        cause) -- H̄_t = H_t[-1], C̄_t = C_t[0]. This measurably reduced
        similarity (mean state-vector cosine 0.958 -> 0.866) but discards
        K-1 history rows and N_C-1 cause rows entirely, which was not the
        intended fix. Its probe results are preserved in
        results/diagnostics/step1b_multiturn_averaging.json and
        step2_supervised_probe.json (both pre-concatenation) as a
        comparison point.

        Concatenation (superseded 2026-09-21): kept every row distinct
        instead of reducing to one -- H̄_t = flatten(H_t), C̄_t =
        flatten(C_t). Reasonable while `encode_dialogue` still produced K
        genuinely different history rows and N_C genuinely different
        cause rows. It no longer is: `encode_dialogue` was switched to
        whole-window encoding (join the K turns / N_C cause spans into one
        text block, pool once via Qwen), and `from_dialogue` repeats that
        single result K (resp. N_C) times purely to keep this class's
        [K, D_H] / [N_C, D_C] field shapes compatible with esc/env.py's
        sliding window and the transition model -- see from_dialogue's
        docstring. Concatenating those now-identical rows produces a
        5507-dim vector where large contiguous blocks are exact
        duplicates of each other: no additional signal, just wasted
        dimensions (bigger networks, more parameters, slower, for
        nothing).

        Current (single-row extraction): H̄_t = H_t[-1], C̄_t = C_t[0].
        Since whole-window makes every row within a group identical, this
        is mathematically equivalent to using the raw pre-repeat
        [D_H]/[D_C] vectors directly -- confirmed by exact reproduction in
        scripts/diagnose_step4_verify_option_b.py against the diagnostic
        scripts that never went through the repeat step at all. This is
        NOT "select" from the pre-whole-window comparison (which extracted
        one of K genuinely *different* rows, discarding real information
        from the other K-1); it's a no-op flatten of duplicate rows,
        recovering the original D_H+D_C+D_E+3 = 1283 state_dim.

        Returns
        -------
        torch.Tensor of shape [D_H + D_C + D_E + 3]
        """
        h = self.history_embeddings[-1]                                   # [D_H]
        c_matrix = self.causal_graph.cause_embedding_matrix(self.N_C, self.D_C)
        c = c_matrix[0]                                                   # [D_C]

        return torch.cat([
            h,                      # [D_H]
            c,                      # [D_C]
            self.emotion_vector,    # [D_E]
            self.phase_embedding,   # [3]
        ], dim=0)

    @classmethod
    def get_state_dim(cls) -> int:
        """Total state vector dimension: d = D_H + D_C + D_E + D_P = 1283.

        Reverted 2026-09-21 from the brief K*D_H + N_C*D_C + D_E + D_P
        (5507) concatenation-era formula -- see to_tensor()'s docstring:
        whole-window encoding makes every one of the K (resp. N_C)
        repeated rows identical, so concatenating them added zero signal,
        only wasted dimensions.
        """
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

        and must return a dict with keys (2026-09-20, whole-window encoding
        -- see models/backbone_qwen.py::encode_dialogue's docstring):
          - "history"        : Tensor [D_H]   — single whole-window turn embedding
          - "emotion"        : Tensor [D_E]   — current emotion vector
          - "cause_embedding": Tensor [D_C]   — single whole-window cause embedding
          - "cause_label"    : str            — truncated text excerpt

        The single history/cause vectors are repeated to fill the
        [K, D_H] / [N_C, D_C] matrix shapes the rest of the pipeline
        (esc/env.py's sliding window, the transition model) still expects
        -- Option B from the debug spec: interface-compatible, not a full
        architecture change. All K (resp. N_C) rows are identical.

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
            history_vec = result["history"]                 # [D_H], whole-window
            emotion = result["emotion"]                     # [D_E]
            cause_vec = result["cause_embedding"]            # [D_C], whole-window
            cause_label = result.get("cause_label") or "cause"

            # Repeat to preserve the [K, D_H] / [N_C, D_C] shapes downstream
            # code expects (Option B -- see this method's docstring). All
            # rows are identical since whole-window collapses the group to
            # one vector before this point.
            history = history_vec.unsqueeze(0).repeat(cls.K_HISTORY_WINDOW, 1)

            graph = CausalGraph(d_c=cls.D_C)
            for _ in range(cls.N_C):
                graph.add_cause(label=cause_label, embedding=cause_vec)
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
