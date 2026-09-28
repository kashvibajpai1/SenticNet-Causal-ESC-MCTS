"""Value network V_θ(s): state vector → scalar value estimate."""

from __future__ import annotations

import torch
from torch import nn


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


class ValueNetwork(nn.Module):
    """
    Value network V_θ(s).

    Maps a flat state vector s ∈ ℝ^{state_dim} to a scalar value
    estimate in [0, ∞) via ReLU output activation.

    Used for MCTS leaf evaluation — replaces high-variance random rollout.

    Architecture: SharedTrunk → Linear(hidden_dim, 1) → ReLU

    2026-09-21: tanh replaced with ReLU to eliminate the vanishing-gradient
    saturation trap diagnosed under whole-window encoding -- 5 of 6 training
    runs (all seeds, both causal_mcts and flow_ablation) drove tanh's
    pre-activation to values far past its useful range (mean 5.0-6.0),
    saturating post-tanh to exactly 1.0000 within ~50 steps regardless of
    input representation or training objective -- see the conversation log
    for the saturation logs and static check that preceded this change.

    Two known risks this change introduces, being watched for empirically
    rather than pre-emptively patched (see the conversation log):
    - PUCT's exploration term in mcts/node.py::select_child is bounded to
      roughly [0, c_puct] (c_puct=1.5); Q was comparably bounded to [-1,1]
      under tanh. ReLU's unbounded Q could make the exploration term
      negligible once Q exceeds ~1.5, biasing search toward pure
      exploitation.
    - esc/reward.py's R_cause/R_emotion components can be *negative* (a
      bad action). ReLU cannot represent a negative target at all -- this
      is a real range mismatch, not just a tuning question.

    2026-09-24: rank_loss weight was investigated as a separate lever
    (rankinglossweight in config/train_shared.yaml). Setting it to 0.0
    eliminated the value network's residual near-ceiling collapse (Pearson
    r vs real reward rose from ~0.4-0.6 to 0.66-0.78) but made causal_mcts's
    downstream MCTS strategy accuracy significantly worse (-4.8pp vs
    random_floor, p=0.0218) -- rank_loss=1.0 was reverted to and kept as the
    shipped config despite the value head still showing collapsed-looking
    output (value_mean 0.93-0.99, std ~0.01-0.05, Pearson r 0.39-0.73 across
    seeds in the final run) precisely BECAUSE it produces better measured
    accuracy. This is a pragmatic choice, not a resolved root cause: the
    mechanism diagnosed here (rank_loss's batch-uniform gradient pulling
    every output toward its margin) is still active in the shipped config.
    See INVESTIGATION.md for the full rank_loss/c_puct exploration.

    Parameters
    ----------
    state_dim  : Flat state vector dimension
    hidden_dim : Hidden layer width (default 256)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self._trunk = _SharedTrunk(state_dim, hidden_dim)
        self._head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : Flat state tensor, shape [state_dim] or [B, state_dim]

        Returns
        -------
        Scalar value estimate, shape [..., 1], range [0, ∞)
        """
        h = self._trunk(state)
        return torch.relu(self._head(h))

    def value(self, state_tensor: torch.Tensor) -> float:
        """Convenience wrapper returning a Python float (no grad)."""
        with torch.no_grad():
            return float(self.forward(state_tensor).item())
