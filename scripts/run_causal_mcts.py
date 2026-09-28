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

SATURATION_LOG_INTERVAL = 50
PEARSON_LOG_INTERVAL = 100


def _log_value_saturation(value: ValueNetwork, states: list[ESCState], step: int) -> dict:
    """Snapshot the value network's pre-activation/post-activation output
    distribution on the current batch, without affecting training (no
    grad, separate forward pass). Added 2026-09-21 to track whether the
    ~0.9999-saturated value network diagnosed under tanh (both under the
    old collapsed mean-pooling representation AND under whole-window
    inputs -- see the conversation log) persists after switching the
    output activation to ReLU.

    IMPORTANT: this calls value(S) -- the network's own forward() -- for
    post-activation, rather than manually reapplying a hardcoded
    activation function. An earlier version of this helper manually
    computed `torch.tanh(pre_activation)`, which would have silently kept
    logging fake tanh-based numbers after the network itself switched to
    ReLU. Field names are activation-agnostic (pre/post_activation) since
    which activation is live is now a model property, not something this
    logger should assume."""
    with torch.no_grad():
        S = torch.stack([s.to_tensor() for s in states])
        h = value._trunk(S)
        pre_activation = value._head(h).squeeze(-1)
        post_activation = value(S).squeeze(-1)
    return {
        "step": step,
        "pre_activation_mean": float(pre_activation.mean()),
        "pre_activation_std": float(pre_activation.std()),
        "pre_activation_min": float(pre_activation.min()),
        "pre_activation_max": float(pre_activation.max()),
        "post_activation_mean": float(post_activation.mean()),
        "post_activation_std": float(post_activation.std()),
        "post_activation_min": float(post_activation.min()),
        "post_activation_max": float(post_activation.max()),
    }


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
    saturation_log: list[dict] = []
    pearson_log: list[dict] = []
    for step in range(max_steps):
        if bundle_ds is not None:
            idx = torch.randint(0, len(bundle_ds), (batch_size,))
            states = [bundle_ds.get_state(int(i)) for i in idx]
        else:
            states = [env.reset(initial_turns=None) for _ in range(batch_size)]
        last_metrics = trainer.train_step(states)
        print(f"[CausalMCTS] seed={seed} step={step} loss={last_metrics['loss']:.4f}")

        if step % PEARSON_LOG_INTERVAL == 0 or step == max_steps - 1:
            pearson_snap = {
                "step": step,
                "pearson_r": last_metrics["pearson_r"],
                "reward_mean": last_metrics["reward_mean"],
                "reward_std": last_metrics["reward_std"],
                "value_mean": last_metrics["value_mean"],
                "value_std": last_metrics["value_std"],
            }
            pearson_log.append(pearson_snap)
            print(
                f"[CausalMCTS] seed={seed} step={step} pearson_r={pearson_snap['pearson_r']:.4f} "
                f"reward(mean={pearson_snap['reward_mean']:.4f} std={pearson_snap['reward_std']:.4f}) "
                f"value(mean={pearson_snap['value_mean']:.4f} std={pearson_snap['value_std']:.4f})"
            )

        if step % SATURATION_LOG_INTERVAL == 0 or step == max_steps - 1:
            snap = _log_value_saturation(value, states, step)
            saturation_log.append(snap)
            print(
                f"[CausalMCTS] seed={seed} step={step} value_saturation "
                f"pre_act(mean={snap['pre_activation_mean']:.4f} std={snap['pre_activation_std']:.4f}) "
                f"post_act(mean={snap['post_activation_mean']:.4f} std={snap['post_activation_std']:.4f} "
                f"min={snap['post_activation_min']:.4f} max={snap['post_activation_max']:.4f})"
            )

    ckpt_dir = os.path.join(root, "checkpoints", "causal_mcts", f"seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
    torch.save(value.state_dict(), os.path.join(ckpt_dir, "value.pt"))
    torch.save(transition_model.state_dict(), os.path.join(ckpt_dir, "transition.pt"))
    with open(os.path.join(ckpt_dir, "final_metrics.json"), "w") as f:
        json.dump({"seed": seed, "max_steps": max_steps, **last_metrics}, f, indent=2)
    with open(os.path.join(ckpt_dir, "value_saturation_log.json"), "w") as f:
        json.dump(saturation_log, f, indent=2)

    diagnostics_dir = os.path.join(root, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    pearson_path = os.path.join(diagnostics_dir, f"value_pearson_r_seed{seed}.json")
    with open(pearson_path, "w") as f:
        json.dump(pearson_log, f, indent=2)
    print(f"[CausalMCTS] seed={seed} checkpoint saved -> {ckpt_dir}")
    print(f"[CausalMCTS] seed={seed} pearson_r log -> {pearson_path}")


if __name__ == "__main__":
    main()
