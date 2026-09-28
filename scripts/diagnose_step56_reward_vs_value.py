"""Deep diagnostic Steps 5+6+8: real reward distribution vs trained value
network's predictions, on the actual seed=42 (ReLU) checkpoint.

For 500 sampled training-bundle states, uses the TRAINED policy+value+
transition to pick an action the same way training does (mcts.search_with_
policy -> env.step), records the real reward R(s,a) from esc/reward.py, and
the trained value network's prediction V(s) on the pre-action state. Reports
reward distribution, prediction distribution, Pearson r, and MSE between
them. Read-only: loads the seed42 checkpoint, no training, no writes to
checkpoints/.
"""

from __future__ import annotations

import json
import os
import statistics

import torch

from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.train_data import ESCStateBundleDataset
from train.utils import load_merged_config
from utils.seed import set_global_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SAMPLES = 500


def main() -> None:
    config = load_merged_config(ROOT)
    set_global_seed(42)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    ckpt_dir = os.path.join(ROOT, "checkpoints", "causal_mcts", "seed42")
    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    transition = LinearTransitionModel(
        state_dim=state_dim, n_causes=ESCState.N_C, d_e=ESCState.D_E,
        n_strategies=ESCAction.NUM_STRATEGIES,
    )
    policy.load_state_dict(torch.load(os.path.join(ckpt_dir, "policy.pt"), map_location="cpu"))
    value.load_state_dict(torch.load(os.path.join(ckpt_dir, "value.pt"), map_location="cpu"))
    transition.load_state_dict(torch.load(os.path.join(ckpt_dir, "transition.pt"), map_location="cpu"))
    policy.eval()
    value.eval()
    transition.eval()

    env = ESCEnv(transition_model=transition, config=config)
    mcts = MCTS(env=env, policy_network=policy, value_network=value, config=config)

    bundles_path = os.path.join(ROOT, "artifacts", "states", "train.pt")
    bundle_ds = ESCStateBundleDataset(bundles_path)
    n = min(N_SAMPLES, len(bundle_ds))
    idx = torch.randperm(len(bundle_ds))[:n]

    rewards: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for i in idx:
            state = bundle_ds.get_state(int(i))
            action = mcts.search(state)
            _next_state, reward, _done, _info = env.step(state, action)
            pred = float(value(state.to_tensor()).item())
            rewards.append(float(reward))
            predictions.append(pred)

    def stats(xs: list[float]) -> dict:
        return {
            "min": min(xs), "mean": statistics.mean(xs), "median": statistics.median(xs),
            "std": statistics.pstdev(xs), "max": max(xs),
        }

    r_mean = statistics.mean(rewards)
    p_mean = statistics.mean(predictions)
    cov = sum((r - r_mean) * (p - p_mean) for r, p in zip(rewards, predictions)) / n
    r_std = statistics.pstdev(rewards)
    p_std = statistics.pstdev(predictions)
    pearson_r = cov / (r_std * p_std) if r_std > 0 and p_std > 0 else float("nan")
    mse = sum((r - p) ** 2 for r, p in zip(rewards, predictions)) / n

    reward_bins = [(-2.0, -1.0), (-1.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 10.0)]
    reward_hist = []
    for lo, hi in reward_bins:
        c = sum(1 for r in rewards if lo <= r < hi)
        reward_hist.append({"range": f"[{lo},{hi})", "count": c})

    summary = {
        "n_samples": n,
        "checkpoint": ckpt_dir,
        "reward_stats": stats(rewards),
        "prediction_stats": stats(predictions),
        "pearson_r": pearson_r,
        "mse": mse,
        "reward_histogram": reward_hist,
        "rewards": rewards,
        "predictions": predictions,
    }

    out_dir = os.path.join(ROOT, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reward_vs_value.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[reward_vs_value] wrote {out_path}")
    print(f"[reward_vs_value] n={n}")
    print(f"[reward_vs_value] REWARD  stats: {stats(rewards)}")
    print(f"[reward_vs_value] PREDICT stats: {stats(predictions)}")
    print(f"[reward_vs_value] Pearson r = {pearson_r:.4f}, MSE = {mse:.4f}")
    for b in reward_hist:
        print(f"[reward_vs_value] reward {b['range']:>14}: {b['count']:4d} {'#'*b['count']}")


if __name__ == "__main__":
    main()
