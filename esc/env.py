"""
ESC MDP environment: M = (S, A, P, R)

Changes from v1
---------------
- Accepts a pluggable TransitionModel; defaults to RandomTransitionModel
  so the pipeline produces non-trivial, differentiable next states
  immediately (no identity copy any more).
- step() uses TransitionOutput to update emotion, phase, and cause
  resolution in the new state rather than just cloning.
- History window is slid: the oldest turn is dropped and a placeholder
  new-turn embedding is appended (until backbone encoder is wired in).
- Exposes render() for lightweight debugging.
"""

from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn.functional as F

from esc.state import ESCState
from esc.action import ESCAction
from esc.reward import compute_reward, reward_components
from esc.causal_graph import CausalGraph
from models.transition import TransitionModel, RandomTransitionModel


class ESCEnv:
    """
    Emotional Support Conversation MDP environment.

    Interface
    ---------
    reset([initial_turns]) → ESCState s_0
    step(s_t, a_t)         → (s_{t+1}, reward, done, info)

    Parameters
    ----------
    transition_model : TransitionModel instance (default: RandomTransitionModel)
                       Swap in LinearTransitionModel once trained.
    max_horizon      : Episode length in turns (default 20)
    reward_weights   : (alpha, beta, gamma) for the three reward components
    device           : Torch device string (also used as default target device
                       for new tensors when the state is not yet on device)
    config           : Optional flat dict; may supply ``max_horizon`` /
                       ``maxhorizon`` and ``device`` to override the kwargs above
    """

    def __init__(
        self,
        transition_model: Optional[TransitionModel] = None,
        max_horizon: int = 20,
        reward_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: str = "cpu",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._transition = transition_model or RandomTransitionModel(
            d_e=ESCState.D_E,
            n_causes=ESCState.N_C,
        )
        self._max_horizon = max_horizon
        self._alpha, self._beta, self._gamma = reward_weights
        self._device = device
        if config is not None:
            self._max_horizon = int(
                config.get("max_horizon", config.get("maxhorizon", self._max_horizon))
            )
            self._device = str(config.get("device", self._device))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_horizon(self) -> int:
        return self._max_horizon

    @property
    def k_hist(self) -> int:
        return ESCState.K_HISTORY_WINDOW

    @property
    def d_h(self) -> int:
        return ESCState.D_H

    @property
    def n_causes(self) -> int:
        return ESCState.N_C

    @property
    def d_c(self) -> int:
        return ESCState.D_C

    @property
    def d_e(self) -> int:
        return ESCState.D_E

    @property
    def d_p(self) -> int:
        return ESCState.D_P

    # ------------------------------------------------------------------
    # MDP interface
    # ------------------------------------------------------------------

    def reset(
        self,
        initial_turns: Optional[list[str]] = None,
        target_emotion: Optional[torch.Tensor] = None,
    ) -> ESCState:
        """
        Reset the environment and return the initial state s_0.

        Parameters
        ----------
        initial_turns   : Optional dialogue strings to seed the state.
                          Until the backbone encoder is wired in, this
                          sets turn_index=0 but embeddings remain zero.
        target_emotion  : Desired emotional endpoint [D_E].
                          Defaults to zero vector (calm/neutral).

        Returns
        -------
        ESCState s_0
        """
        return ESCState.from_dialogue(
            turns=initial_turns or [],
            encoder=None,               # plug real encoder here later
            target_emotion=target_emotion,
        )

    def step(
        self,
        state: ESCState,
        action: ESCAction,
    ) -> tuple[ESCState, float, bool, dict]:
        """
        Execute one MDP step: (s_t, a_t) → (s_{t+1}, r_t, done, info).

        Transition dynamics
        -------------------
        1. Run TransitionModel → TransitionOutput
        2. Clone state and apply updates:
           a. Slide history window (drop oldest turn, append zero placeholder)
           b. Update emotion_vector from TransitionOutput.next_emotion
           c. Update phase_embedding via softmax(next_phase_logits)
           d. Apply delta_resolution to causal graph (clamped to [0,1])
        3. Compute reward using the corrected reward function.
        4. Check done flag.

        Parameters
        ----------
        state  : Current state s_t
        action : Action a_t = (strategy_id, cause_index)

        Returns
        -------
        next_state : ESCState s_{t+1}
        reward     : float — R(s_t, a_t)
        done       : bool — True when horizon reached
        info       : dict — diagnostic details
        """
        # 1. Run transition model
        t_out = self._transition.forward(state, action)

        # 2. Build next state
        next_state = state.clone()
        next_state.turn_index = state.turn_index + 1

        # 2a. Slide history window: drop row 0, append zeros for new turn
        #     (placeholder; backbone encoder would provide the real embedding)
        dev = state.history_embeddings.device
        new_turn_emb = torch.zeros(1, self.d_h, device=dev)
        next_state.history_embeddings = torch.cat(
            [state.history_embeddings[1:], new_turn_emb], dim=0
        )  # still [K, D_H]

        # 2b. Update emotion
        next_state.emotion_vector = t_out.next_emotion.detach()

        # 2c. Update phase: softmax over 3 logits → valid probability vector
        next_state.phase_embedding = F.softmax(
            t_out.next_phase_logits.detach(), dim=0
        )

        # 2d. Apply resolution deltas to causal graph (clamp each to [0,1])
        old_res = state.causal_graph.resolution_tensor()             # [n_c]
        new_res = (old_res + t_out.delta_resolution.detach()).clamp(0.0, 1.0)
        next_state.causal_graph.update_resolution(new_res)

        # 3. Compute reward
        reward = compute_reward(
            state, next_state,
            alpha=self._alpha,
            beta=self._beta,
            gamma=self._gamma,
        )

        # 4. Termination
        done = next_state.turn_index >= self._max_horizon

        # 5. Info dict (for logging / debugging)
        components = reward_components(state, next_state)
        info: dict = {
            "turn": next_state.turn_index,
            "strategy_used": action.strategy_id,
            "cause_targeted": action.cause_index,
            "horizon_reached": done,
            "phase": next_state.current_phase,
            "R_cause": components["R_cause"],
            "R_emotion": components["R_emotion"],
            "R_phase": components["R_phase"],
            "resolution": next_state.causal_graph.resolution_tensor().tolist(),
        }

        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def render(self, state: ESCState) -> str:
        """
        Return a human-readable summary of the current state.

        Not called during training; useful for interactive debugging.
        """
        phase_names = ["Exploration", "Comforting", "Action Planning"]
        phase_name = phase_names[state.current_phase]
        res = state.causal_graph.resolution_tensor().tolist()
        res_str = ", ".join(f"{r:.2f}" for r in res)
        emo_norm = float(state.emotion_vector.norm().item())
        target_dist = float(
            torch.norm(state.emotion_vector - state.target_emotion).item()
        )
        return (
            f"Turn {state.turn_index} | Phase: {phase_name} | "
            f"Emotion norm: {emo_norm:.3f} | "
            f"Dist-to-target: {target_dist:.3f} | "
            f"Cause resolutions: [{res_str}]"
        )
