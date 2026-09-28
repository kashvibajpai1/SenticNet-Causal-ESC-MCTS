"""FlowMCTS-ablation trainer from synthetic state batches.

NOTE: this is an in-house GFlowNet-style flow-matching ablation (trains the
policy/value networks with flow-consistency + ranking losses, no MCTS search
and no transition model) — it is NOT a reproduction of the published AFlow
system (Zou et al.). See flow/README.md.
"""

from __future__ import annotations

from typing import Any

import torch

from flow import (
    compute_edge_flow,
    compute_state_flow,
    flow_consistency_loss,
    ranking_loss,
)
from models.policy import PolicyNetwork
from models.value import ValueNetwork


class FlowAblationTrainer:
    """Trains π_θ and V_φ with flow consistency and a lightweight ranking term on values."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        config: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.value = value
        self.config = config
        lr = float(config.get("learning_rate", config.get("learningrate", 1e-4)))
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()),
            lr=lr,
        )

    def train_step(self, states: torch.Tensor) -> dict[str, float]:
        """
        One optimization step on a batch of flat state vectors.

        Parameters
        ----------
        states : tensor (batch_size, state_dim)
        """
        policy_probs = self.policy(states)
        values = self.value(states)

        num_actions = policy_probs.shape[-1]
        V = values.expand(-1, num_actions)
        Q_b = policy_probs
        F_s = compute_state_flow(Q_b, V)
        F_edges = compute_edge_flow(F_s, policy_probs.detach())
        flow_from_policy = compute_edge_flow(
            F_s.detach(), policy_probs.detach()
        ).detach()
        flow_loss = flow_consistency_loss(F_edges, flow_from_policy)

        margin = float(
            self.config.get("gamma_margin", self.config.get("gammamargin", 1.0))
        )
        v_winner = values.mean()
        rank_loss = ranking_loss(
            v_winner,
            torch.zeros_like(v_winner),
            margin=margin,
        )

        w_flow = float(
            self.config.get("flow_loss_weight", self.config.get("flowlossweight", 1.0))
        )
        w_rank = float(
            self.config.get(
                "ranking_loss_weight", self.config.get("rankinglossweight", 1.0)
            )
        )
        loss = w_flow * flow_loss + w_rank * rank_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "flow_loss": float(flow_loss.item()),
            "rank_loss": float(rank_loss.item()),
        }
