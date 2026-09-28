"""Policy π_θ(a | s): state vector → action probability distribution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SharedTrunk(nn.Module):
    """Shared feature extractor trunk."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """
    Policy network π_θ(a | s).

    Maps a flat state vector s ∈ ℝ^{state_dim} to a probability
    distribution over |action_dim| candidate actions.

    Architecture: SharedTrunk → Linear(hidden_dim, action_dim) → Softmax

    Parameters
    ----------
    state_dim  : Flat state vector dimension
    action_dim : Number of candidate actions |A(s)|
    hidden_dim : Hidden layer width (default 256)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self._trunk = _SharedTrunk(state_dim, hidden_dim)
        self._head = nn.Linear(hidden_dim, action_dim)
        self.num_actions = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : Flat state tensor, shape [state_dim] or [B, state_dim]

        Returns
        -------
        Probability distribution, shape [..., action_dim], sums to 1
        """
        h = self._trunk(state)
        return F.softmax(self._head(h), dim=-1)
