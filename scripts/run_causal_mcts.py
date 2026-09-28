"""Script entry for causal MCTS training with ESCEnv root states."""

from __future__ import annotations

import argparse
import json
import os

import torch

from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.train_data import ESCStateBundleDataset
from train.trainer_causal_mcts import CausalMCTSTrainer
from train.utils import load_merged_config
from utils.seed import set_global_seed

_SYNTHETIC_WARNING = (
    "WARNING: Running in synthetic smoke-test mode. "
    "This does not reproduce the paper experiments."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_merged_config(root)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_global_seed(seed)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

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

    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    planner = MCTS(
        env=env,
        policy_network=policy,
        value_network=value,
        config=config,
    )

    trainer = CausalMCTSTrainer(
        policy,
        value,
        transition_model,
        planner,
        config=config,
    )

    batch_size = int(config.get("batch_size", config.get("batchsize", 4)))
    max_steps = int(config.get("max_steps", config.get("maxsteps", 10)))

    bundles_path = os.path.join(root, "artifacts", "states", "train.pt")
    bundle_ds: ESCStateBundleDataset | None = None
    if os.path.isfile(bundles_path):
        try:
            cand = ESCStateBundleDataset(bundles_path)
            if len(cand) > 0:
                bundle_ds = cand
        except OSError:
            bundle_ds = None

    if bundle_ds is None:
        print(_SYNTHETIC_WARNING)

    last_metrics: dict[str, float] = {}
    for step in range(max_steps):
        if bundle_ds is not None:
            idx = torch.randint(0, len(bundle_ds), (batch_size,))
            states = [bundle_ds.get_state(int(i)) for i in idx]
        else:
            states = [env.reset(initial_turns=None) for _ in range(batch_size)]
        last_metrics = trainer.train_step(states)
        print(f"[CausalMCTS] seed={seed} step={step} loss={last_metrics['loss']:.4f}")

    ckpt_dir = os.path.join(root, "checkpoints", "causal_mcts", f"seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
    torch.save(value.state_dict(), os.path.join(ckpt_dir, "value.pt"))
    torch.save(transition_model.state_dict(), os.path.join(ckpt_dir, "transition.pt"))
    with open(os.path.join(ckpt_dir, "final_metrics.json"), "w") as f:
        json.dump({"seed": seed, "max_steps": max_steps, **last_metrics}, f, indent=2)
    print(f"[CausalMCTS] seed={seed} checkpoint saved -> {ckpt_dir}")


if __name__ == "__main__":
    main()
