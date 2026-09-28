"""Flow-matching state flow :math:`F(s)=Q_b(s)V_\\phi(s)` and edge factors (in-house ablation, not AFlow)."""

import torch


def compute_state_flow(Q_b: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    if Q_b.shape != V.shape:
        raise ValueError(
            f"Q_b and V must have same shape, got {Q_b.shape} and {V.shape}"
        )
    return Q_b - V


def compute_edge_flow(F_s: torch.Tensor, policy_probs: torch.Tensor) -> torch.Tensor:
    if F_s.ndim == policy_probs.ndim:
        return F_s * policy_probs
    return F_s.unsqueeze(-1) * policy_probs
