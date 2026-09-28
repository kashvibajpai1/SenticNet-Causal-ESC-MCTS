"""
Learned transition model f_θ(s_t, a_t) → s_{t+1} components.

Outputs three tensors consumed by ESCEnv.step():
  next_emotion      ∈ ℝ^{d_e}  — updated emotional state
  next_phase_logits ∈ ℝ^{3}    — logits over 3 ESC phases
  delta_resolution  ∈ ℝ^{n_c}  — per-cause resolution delta ∈ (-1, 1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from esc.state import ESCState
from esc.action import ESCAction


@dataclass
class TransitionOutput:
    """
    Structured output from a TransitionModel forward pass.

    Attributes
    ----------
    next_emotion       : Predicted emotion vector for s_{t+1}, shape [d_e]
    next_phase_logits  : Raw logits over 3 ESC phases, shape [3]
    delta_resolution   : Per-cause resolution change in (-1, 1), shape [n_c]
    """
    next_emotion: torch.Tensor        # [d_e]
    next_phase_logits: torch.Tensor   # [3]
    delta_resolution: torch.Tensor    # [n_c]


class TransitionModel(ABC):
    """
    Abstract base for ESC transition models f_θ(s_t, a_t) → TransitionOutput.

    Subclasses implement forward().  ESCEnv.step() calls forward() and
    uses the output to construct the next ESCState.
    """

    @abstractmethod
    def forward(self, state: ESCState, action: ESCAction) -> TransitionOutput:
        """Predict next-state components given current state and action."""
        ...


class LinearTransitionModel(TransitionModel, nn.Module):
    """
    Lightweight learned transition model (MLP backbone).

    Architecture
    ------------
    input = concat(state_vec, action_onehot) ∈ ℝ^{state_dim + action_dim}
    h     = ReLU(LayerNorm(Linear(input, hidden_dim)))
            ReLU(LayerNorm(Linear(h, hidden_dim)))

    next_emotion      = tanh(Linear(h, d_e))       ∈ (-1, 1)^{d_e}
    next_phase_logits = Linear(h, 3)               ∈ ℝ^3
    delta_resolution  = tanh(Linear(h, n_causes))  ∈ (-1, 1)^{n_c}

    Parameters
    ----------
    state_dim    : ESCState.get_state_dim()
    n_causes     : ESCState.N_C
    d_e          : ESCState.D_E
    n_strategies : ESCAction.NUM_STRATEGIES
    hidden_dim   : Hidden layer width (default 256)
    """

    def __init__(
        self,
        state_dim: int,
        n_causes: int,
        d_e: int,
        n_strategies: int,
        hidden_dim: int = 256,
    ) -> None:
        nn.Module.__init__(self)
        action_dim = n_strategies + n_causes
        input_dim = state_dim + action_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.emotion_head = nn.Linear(hidden_dim, d_e)
        self.phase_head = nn.Linear(hidden_dim, 3)
        self.resolution_head = nn.Linear(hidden_dim, n_causes)

        self._n_strategies = n_strategies
        self._n_causes = n_causes

    def _encode_action(self, action: ESCAction) -> torch.Tensor:
        strat = torch.zeros(self._n_strategies)
        if 0 <= action.strategy_id < self._n_strategies:
            strat[action.strategy_id] = 1.0
        cause = torch.zeros(self._n_causes)
        if 0 <= action.cause_index < self._n_causes:
            cause[action.cause_index] = 1.0
        return torch.cat([strat, cause], dim=0)

    def forward(self, state: ESCState, action: ESCAction) -> TransitionOutput:  # type: ignore[override]
        state_vec = state.to_tensor()
        action_vec = self._encode_action(action)
        x = torch.cat([state_vec, action_vec], dim=0)
        h = self.trunk(x)
        return TransitionOutput(
            next_emotion=torch.tanh(self.emotion_head(h)),
            next_phase_logits=self.phase_head(h),
            delta_resolution=torch.tanh(self.resolution_head(h)),
        )


class RandomTransitionModel(TransitionModel):
    """
    Randomised transition model for testing and ablation baselines.

    Produces small random perturbations so rewards are non-zero without
    a trained model.  Not for real training — only for pipeline validation.

    Parameters
    ----------
    d_e                    : Emotion dimension
    n_causes               : Number of causes
    emotion_noise_scale    : Std of Gaussian noise added to emotion (default 0.05)
    resolution_delta_scale : Max absolute resolution delta (default 0.1)
    seed                   : Optional RNG seed for reproducibility
    """

    def __init__(
        self,
        d_e: int,
        n_causes: int,
        emotion_noise_scale: float = 0.05,
        resolution_delta_scale: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self._d_e = d_e
        self._n_causes = n_causes
        self._emotion_noise_scale = emotion_noise_scale
        self._resolution_delta_scale = resolution_delta_scale
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def forward(self, state: ESCState, action: ESCAction) -> TransitionOutput:
        noise = torch.randn(self._d_e, generator=self._rng) * self._emotion_noise_scale
        next_emotion = (state.emotion_vector + noise).clamp(-1.0, 1.0)
        phase_logits = torch.randn(3, generator=self._rng) * 0.1
        delta_res = (
            torch.rand(self._n_causes, generator=self._rng)
            * self._resolution_delta_scale
        )
        return TransitionOutput(
            next_emotion=next_emotion,
            next_phase_logits=phase_logits,
            delta_resolution=delta_res,
        )
