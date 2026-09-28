"""Seed43 diagnostic: per-seed PUCT tree stats + eval-time value distribution.

Loads each seed's trained checkpoint (current c_puct=0.5 config, no
retraining), runs MCTS.search() with the PUCT_DEBUG hook (mcts/node.py) over
N_STATES sampled states from the existing training bundle
(artifacts/states/train.pt -- already-encoded, so no LLM calls needed), and
records: every select_child() candidate's Q/exploration_term/puct, plus the
raw value network output V(s) on each pre-search state. Read-only w.r.t.
checkpoints; writes diagnostics/seed43_puct_and_value_seed{N}.json per seed.
"""

from __future__ import annotations

import json
import os
import statistics

import torch

from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from mcts import node as node_mod
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.train_data import ESCStateBundleDataset
from train.utils import load_merged_config
from utils.seed import set_global_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_STATES = 40
SEEDS = [42, 43, 44]


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"min": None, "mean": None, "max": None, "std": None}
    return {
        "min": min(xs),
        "mean": statistics.mean(xs),
        "max": max(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def main() -> None:
    config = load_merged_config(ROOT)
    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C
    bundles_path = os.path.join(ROOT, "artifacts", "states", "train.pt")
    bundle_ds = ESCStateBundleDataset(bundles_path)

    all_summaries = {}
    for seed in SEEDS:
        set_global_seed(seed)
        ckpt_dir = os.path.join(ROOT, "checkpoints", "causal_mcts", f"seed{seed}")
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

        idx = torch.randperm(len(bundle_ds))[:N_STATES]
        value_outputs: list[float] = []

        node_mod.PUCT_DEBUG_LOG.clear()
        node_mod.PUCT_DEBUG = True
        try:
            with torch.no_grad():
                for i in idx:
                    state = bundle_ds.get_state(int(i))
                    value_outputs.append(float(value(state.to_tensor()).item()))
                    mcts.search(state)
        finally:
            node_mod.PUCT_DEBUG = False

        log = list(node_mod.PUCT_DEBUG_LOG)
        all_q = [c["Q"] for entry in log for c in entry["candidates"]]
        all_u = [c["exploration_term"] for entry in log for c in entry["candidates"]]
        all_puct = [c["puct"] for entry in log for c in entry["candidates"]]
        depths = [entry["depth"] for entry in log]
        n_candidates_per_call = [len(entry["candidates"]) for entry in log]

        q_stats = stats(all_q)
        u_stats = stats(all_u)
        ratio = (u_stats["mean"] / q_stats["mean"]) if q_stats["mean"] not in (None, 0) else None

        summary = {
            "seed": seed,
            "checkpoint": ckpt_dir,
            "c_puct": mcts._c_puct,
            "num_simulations": mcts._num_simulations,
            "n_states_sampled": N_STATES,
            "n_select_child_calls": len(log),
            "value_output_stats": stats(value_outputs),
            "Q_stats": q_stats,
            "exploration_term_stats": u_stats,
            "puct_stats": stats(all_puct),
            "ratio_exploration_over_Q": ratio,
            "max_depth_seen": max(depths) if depths else None,
            "mean_candidates_per_call": statistics.mean(n_candidates_per_call) if n_candidates_per_call else None,
        }
        all_summaries[f"seed{seed}"] = summary

        print(f"[seed{seed}] value_output: {summary['value_output_stats']}")
        print(f"[seed{seed}] Q: {q_stats}  exploration: {u_stats}  ratio(expl/Q)={ratio}")
        print(f"[seed{seed}] n_select_child_calls={len(log)} max_depth={summary['max_depth_seen']}")

        out_path = os.path.join(ROOT, "diagnostics", f"seed43_puct_and_value_seed{seed}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[seed{seed}] wrote {out_path}")

    combined_path = os.path.join(ROOT, "diagnostics", "seed43_puct_and_value_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[combined] wrote {combined_path}")


if __name__ == "__main__":
    main()
