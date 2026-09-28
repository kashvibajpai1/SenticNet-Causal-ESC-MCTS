"""Script entry for FlowMCTS-ablation training (artifacts or synthetic batches).

NOTE: "FlowMCTS-ablation" is an in-house flow-matching ablation, not a
reproduction of the published AFlow system (Zou et al.). See flow/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import cycle

import torch
from torch.utils.data import DataLoader

from data.collate import collate_state_tensors
from esc.action import ESCAction
from esc.state import ESCState
from models.policy import PolicyNetwork
from models.value import ValueNetwork
from train.train_data import ESCStateTensorDataset
from train.trainer_flow_ablation import FlowAblationTrainer
from train.utils import load_merged_config
from utils.seed import set_global_seed

_SYNTHETIC_WARNING = (
    "WARNING: Running in synthetic smoke-test mode. "
    "This does not reproduce the paper experiments."
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_merged_config(root)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_global_seed(seed)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    trainer = FlowAblationTrainer(policy, value, config=config)

    batch_size = int(config.get("batch_size", config.get("batchsize", 4)))
    max_steps = int(config.get("max_steps", config.get("maxsteps", 10)))

    states_path = os.path.join(root, "artifacts", "states", "train.pt")
    loader_cycle = None
    if os.path.isfile(states_path):
        ds = ESCStateTensorDataset(states_path)
        if len(ds) > 0:
            bs = min(batch_size, len(ds))
            loader = DataLoader(
                ds,
                batch_size=bs,
                shuffle=True,
                collate_fn=collate_state_tensors,
            )
            loader_cycle = cycle(loader)

    if loader_cycle is None:
        print(_SYNTHETIC_WARNING)

    last_metrics: dict[str, float] = {}
    for step in range(max_steps):
        if loader_cycle is not None:
            states = next(loader_cycle)
        else:
            states = torch.randn(batch_size, state_dim)
        last_metrics = trainer.train_step(states)
        print(f"[FlowAblation] seed={seed} step={step} loss={last_metrics['loss']:.4f}")

    ckpt_dir = os.path.join(root, "checkpoints", "flow_ablation", f"seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
    torch.save(value.state_dict(), os.path.join(ckpt_dir, "value.pt"))
    with open(os.path.join(ckpt_dir, "final_metrics.json"), "w") as f:
        json.dump({"seed": seed, "max_steps": max_steps, **last_metrics}, f, indent=2)
    print(f"[FlowAblation] seed={seed} checkpoint saved -> {ckpt_dir}")


if __name__ == "__main__":
    main()
