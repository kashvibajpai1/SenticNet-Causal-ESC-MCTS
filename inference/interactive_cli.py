"""Interactive ESC session: MCTS plan + Qwen backbone stub for the reply."""

from __future__ import annotations

import os
from typing import List

from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from mcts.mcts import MCTS
from models.backbone_qwen import QwenBackbone
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.utils import load_merged_config
from utils.seed import set_global_seed


def run_interactive_session() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_merged_config(root)

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    dialogue: List[str] = []

    state_dim = ESCState.get_state_dim()
    transition_model = LinearTransitionModel(
        state_dim=state_dim,
        n_causes=ESCState.N_C,
        d_e=ESCState.D_E,
        n_strategies=ESCAction.NUM_STRATEGIES,
    )
    env = ESCEnv(
        transition_model=transition_model,
        config=config,
    )

    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C
    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    planner = MCTS(
        env=env,
        policy_network=policy,
        value_network=value,
        config=config,
    )

    backbone_name = str(
        config.get("backbone_model_name", "Qwen/Qwen-1.5-9B-Chat")
    )
    backbone = QwenBackbone(model_name=backbone_name)

    state = env.reset(initial_turns=dialogue)

    print("Starting interactive ESC session. Type 'quit' to exit.")
    while True:
        user = input("User: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        dialogue.append(user)

        action = planner.search(state)
        prompt = f"User: {user}\nChosen strategy: {action}\nAssistant:"
        reply = backbone.generate_response(prompt)
        print(f"Assistant: {reply}")

        state, _reward, done, _info = env.step(state, action)
        if done:
            print("Episode finished at max horizon. Resetting.")
            state = env.reset(initial_turns=dialogue)


def main() -> None:
    run_interactive_session()


if __name__ == "__main__":
    main()
