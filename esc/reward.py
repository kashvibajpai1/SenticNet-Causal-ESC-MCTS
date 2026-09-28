"""
Reward computation: R(s_t, a_t) = α·R_cause + β·R_emotion + γ·R_phase

Three fixes from v1
-------------------
1. R_emotion: measures *progress toward* target emotion, not raw distance.
   Positive when the agent moves the user closer to the target state.

2. R_cause: uses the *delta* in resolution probabilities (Δρ), not the
   cumulative sum.  Only the improvement caused by this action earns
   reward; already-resolved causes contribute nothing.

3. Normalisation: each component is scaled to approximately [-1, 1] before
   weighting so that α, β, γ are comparable and the user-specified weights
   have predictable effect.

Normalisation approach
----------------------
  R_cause  ∈ [−1, 1]  via dividing Σ Δρ by n_c  (mean delta per cause)
  R_emotion ∈ [−1, 1] via dividing the distance improvement by d_e^0.5
                        (expected L2 norm scale for unit Gaussian vectors)
  R_phase  ∈ {0, 1}   binary; no change needed
"""

from __future__ import annotations

import math
from typing import Optional
import torch

from esc.state import ESCState


def compute_reward(
    state: ESCState,
    next_state: ESCState,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """
    Compute the decomposed ESC reward.

    R(s_t, a_t) = α·R_cause + β·R_emotion + γ·R_phase

    Component details
    -----------------
    R_cause (cause resolution progress)
        = mean(ρ_{t+1} − ρ_t) over all causes, divided by 1 (already
          mean per-cause, so ∈ [−1, 1]).
        Positive when causes become more resolved; negative if any regress.

    R_emotion (movement toward target)
        = (d(e_t, e*) − d(e_{t+1}, e*)) / sqrt(D_E)
        Positive when the next emotion is closer to target_emotion than
        the current emotion was.  Negative when moving away.

    R_phase (phase advancement)
        = 1.0  if current_phase(s_{t+1}) > current_phase(s_t)
        = 0.0  otherwise

    All three are normalised to approximately [−1, 1] so that the
    weights α, β, γ carry comparable influence.

    Parameters
    ----------
    state       : Current state s_t
    next_state  : Next state s_{t+1} after taking action a_t
    alpha       : Weight for cause-resolution reward (default 1.0)
    beta        : Weight for emotion-progress reward (default 1.0)
    gamma       : Weight for phase-progression reward (default 1.0)

    Returns
    -------
    Scalar total reward as a Python float.
    """
    R_cause = _r_cause(state, next_state)
    R_emotion = _r_emotion(state, next_state)
    R_phase = _r_phase(state, next_state)

    return alpha * R_cause + beta * R_emotion + gamma * R_phase


# ------------------------------------------------------------------
# Individual reward components (public so tests can target each one)
# ------------------------------------------------------------------

def _r_cause(state: ESCState, next_state: ESCState) -> float:
    """
    Mean per-cause resolution delta: mean(ρ_{t+1} − ρ_t).

    ∈ [−1, 1] because each Δρ_i ∈ [−1, 1] and we take the mean.

    Positive  → agent helped resolve causes this step.
    Negative  → agent made causes worse (regression).
    Zero      → no change.
    """
    rho_t = state.causal_graph.resolution_tensor()          # [n_c]
    rho_next = next_state.causal_graph.resolution_tensor()  # [n_c]

    if rho_t.numel() == 0:
        return 0.0

    delta = rho_next - rho_t                                # [n_c]
    return float(delta.mean().item())


def _r_emotion(state: ESCState, next_state: ESCState) -> float:
    """
    Normalised emotional progress toward the target state.

    dist_t    = ||e_t    − e*||_2
    dist_next = ||e_{t+1} − e*||_2
    R_emotion = (dist_t − dist_next) / sqrt(D_E)

    Positive  → moved closer to target (good).
    Negative  → moved further from target (bad).
    Divided by sqrt(D_E) to normalise for dimensionality.
    """
    target = state.target_emotion
    norm_factor = math.sqrt(ESCState.D_E) + 1e-8  # avoid /0

    dist_t = torch.norm(state.emotion_vector - target, p=2).item()
    dist_next = torch.norm(next_state.emotion_vector - target, p=2).item()

    return (dist_t - dist_next) / norm_factor


def _r_phase(state: ESCState, next_state: ESCState) -> float:
    """
    Binary reward for advancing the conversation phase.

    Returns 1.0 if phase index strictly increased, 0.0 otherwise.
    Phase regression is not penalised here (it is implicitly punished
    via the opportunity cost of not earning +1).
    """
    return 1.0 if next_state.current_phase > state.current_phase else 0.0


# ------------------------------------------------------------------
# Convenience re-export for external callers that need individual components
# ------------------------------------------------------------------

def reward_components(
    state: ESCState,
    next_state: ESCState,
) -> dict[str, float]:
    """
    Return all three reward components as a labelled dict.

    Useful for logging and debugging per-component contributions.

    Returns
    -------
    {"R_cause": float, "R_emotion": float, "R_phase": float}
    """
    return {
        "R_cause": _r_cause(state, next_state),
        "R_emotion": _r_emotion(state, next_state),
        "R_phase": _r_phase(state, next_state),
    }
