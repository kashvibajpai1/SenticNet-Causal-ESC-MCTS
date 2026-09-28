"""Flow-matching losses :math:`L_{\\mathrm{flow}}` and ranking :math:`L_{\\mathrm{rank}}` (in-house ablation, not AFlow)."""

import torch
import torch.nn.functional as F


def flow_consistency_loss(
    flow_edges: torch.Tensor, flow_from_policy: torch.Tensor
) -> torch.Tensor:
    if flow_edges.shape != flow_from_policy.shape:
        raise ValueError(
            "flow_edges and flow_from_policy must have same shape, "
            f"got {flow_edges.shape} and {flow_from_policy.shape}"
        )
    return F.mse_loss(flow_edges, flow_from_policy)


def ranking_loss(
    v_winner: torch.Tensor, v_loser: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    diff = v_winner - v_loser
    return torch.clamp(margin - diff, min=0.0).mean()
