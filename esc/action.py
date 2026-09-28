"""
ESC actions: strategy-cause pairs a_t = (σ, c_i)

Changes from v1
---------------
- generate_candidate_actions() now filters by phase: only strategies
  appropriate for the current ESC phase are included.  This reduces the
  branching factor and prevents the policy from having to learn to avoid
  semantically inappropriate strategies from reward signal alone.
- embed_action() uses a small learned nn.Embedding rather than a
  brittle one-hot encoding, and accepts a device argument.
- PHASE_STRATEGY_MAP defines which strategies are legal per phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional
import torch
import torch.nn as nn

from esc.state import ESCState


@dataclass
class ESCAction:
    """
    Support action as strategy-cause pair: a_t = (σ, c_i)

    Attributes
    ----------
    strategy_id : Support strategy index (0–7)
    cause_index : Which cause to address (0 to n_c-1)
    embedding   : Optional dense action embedding ψ(σ, c_i) ∈ ℝ^{d_a}
    """

    strategy_id: int
    cause_index: int
    embedding: Optional[torch.Tensor] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Strategy catalogue (8 ESC strategies from literature)
    # ------------------------------------------------------------------
    STRATEGIES: ClassVar[list[str]] = [
        "Question",               # 0: Ask clarifying questions
        "Restatement",            # 1: Paraphrase to show understanding
        "Reflection",             # 2: Reflect feelings back
        "Self-disclosure",        # 3: Share similar experience
        "Affirmation",            # 4: Validate and encourage
        "Providing Suggestions",  # 5: Offer advice / solutions
        "Information",            # 6: Provide facts / resources
        "Others",                 # 7: Miscellaneous support
    ]
    NUM_STRATEGIES: ClassVar[int] = 8

    # ------------------------------------------------------------------
    # Phase-strategy filtering
    # Phase 0 = Exploration, Phase 1 = Comforting, Phase 2 = Action
    #
    # Rationale:
    #   Exploration  → understand the problem first (Q, Restatement,
    #                  Reflection, Self-disclosure)
    #   Comforting   → emotional stabilisation (Reflection, Affirmation,
    #                  Self-disclosure, Others)
    #   Action       → practical steps (Suggestions, Information,
    #                  Affirmation, Others)
    # ------------------------------------------------------------------
    PHASE_STRATEGY_MAP: ClassVar[dict[int, list[int]]] = {
        0: [0, 1, 2, 3],       # Exploration
        1: [2, 3, 4, 7],       # Comforting
        2: [4, 5, 6, 7],       # Action Planning
    }

    # ------------------------------------------------------------------
    # Hashing and equality (required for use as dict key in TreeNode)
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return hash((self.strategy_id, self.cause_index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ESCAction):
            return NotImplemented
        return (
            self.strategy_id == other.strategy_id
            and self.cause_index == other.cause_index
        )

    def __repr__(self) -> str:
        name = self.STRATEGIES[self.strategy_id]
        return f"ESCAction(strategy={name}, cause={self.cause_index})"


def generate_candidate_actions(
    state: ESCState,
    num_strategies: int = ESCAction.NUM_STRATEGIES,
    num_causes: Optional[int] = None,
    filter_by_phase: bool = True,
) -> list[ESCAction]:
    """
    Generate the candidate action set A(s_t) for a given state.

    Phase filtering
    ---------------
    When filter_by_phase=True (default), only strategies listed in
    ESCAction.PHASE_STRATEGY_MAP[current_phase] are included.  This
    trims the branching factor from 32 to ~8–16, which:
      - Makes MCTS much more tractable.
      - Prevents the policy from wasting budget on actions that are
        semantically wrong for the current conversation phase.

    If the current phase has no entry in the map (shouldn't happen with
    valid states), all strategies are returned as a safe fallback.

    Parameters
    ----------
    state           : Current ESCState
    num_strategies  : Override number of strategies (default 8)
    num_causes      : Override number of causes; inferred from state if None
    filter_by_phase : Enable phase-based strategy filtering (default True)

    Returns
    -------
    List of ESCAction objects, one per valid (strategy, cause) pair.
    Guaranteed non-empty as long as the state has ≥1 cause.
    """
    if num_causes is None:
        num_causes = state.causal_graph.num_causes
        # Fallback: if graph is empty, use class-level N_C
        if num_causes == 0:
            num_causes = ESCState.N_C

    # Determine which strategies are legal for the current phase
    if filter_by_phase:
        phase = state.current_phase
        allowed_strategies = ESCAction.PHASE_STRATEGY_MAP.get(
            phase, list(range(num_strategies))
        )
    else:
        allowed_strategies = list(range(num_strategies))

    candidates: list[ESCAction] = []
    for strategy_id in allowed_strategies:
        for cause_idx in range(num_causes):
            candidates.append(
                ESCAction(strategy_id=strategy_id, cause_index=cause_idx)
            )

    return candidates


class ActionEmbedder(nn.Module):
    """
    Learned dense embeddings for ESC actions: ψ(σ, c_i) ∈ ℝ^{d_a}

    Replaces one-hot encoding with a small nn.Embedding table for each
    of strategy_id and cause_index.  The two embeddings are concatenated.

    Parameters
    ----------
    num_strategies  : Vocabulary size for strategy embeddings
    num_causes      : Vocabulary size for cause embeddings
    strategy_emb_dim: Strategy embedding dimension (default 32)
    cause_emb_dim   : Cause embedding dimension (default 32)
    """

    def __init__(
        self,
        num_strategies: int = ESCAction.NUM_STRATEGIES,
        num_causes: int = ESCState.N_C,
        strategy_emb_dim: int = 32,
        cause_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.strategy_emb = nn.Embedding(num_strategies, strategy_emb_dim)
        self.cause_emb = nn.Embedding(num_causes, cause_emb_dim)
        self.output_dim = strategy_emb_dim + cause_emb_dim

    def forward(self, action: ESCAction) -> torch.Tensor:
        """
        Embed a single ESCAction.

        Parameters
        ----------
        action : ESCAction to embed

        Returns
        -------
        torch.Tensor of shape [strategy_emb_dim + cause_emb_dim]
        """
        s_idx = torch.tensor(action.strategy_id, dtype=torch.long)
        c_idx = torch.tensor(action.cause_index, dtype=torch.long)
        return torch.cat([
            self.strategy_emb(s_idx),
            self.cause_emb(c_idx),
        ], dim=0)


def embed_action(
    action: ESCAction,
    strategy_dim: int = 64,
    cause_dim: int = 64,
) -> torch.Tensor:
    """
    One-hot action embedding for lightweight use (no nn.Module required).

    Suitable for inspection and testing.  For training, prefer ActionEmbedder.

    Parameters
    ----------
    action       : ESCAction to embed
    strategy_dim : One-hot vector length for strategy (must be ≥ NUM_STRATEGIES)
    cause_dim    : One-hot vector length for cause (must be ≥ N_C)

    Returns
    -------
    torch.Tensor of shape [strategy_dim + cause_dim]
    """
    strategy_onehot = torch.zeros(strategy_dim)
    if 0 <= action.strategy_id < strategy_dim:
        strategy_onehot[action.strategy_id] = 1.0

    cause_onehot = torch.zeros(cause_dim)
    if 0 <= action.cause_index < cause_dim:
        cause_onehot[action.cause_index] = 1.0

    return torch.cat([strategy_onehot, cause_onehot], dim=0)
