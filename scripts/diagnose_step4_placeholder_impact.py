"""Debug spec Step 4: audit the simulated-state placeholder's impact on search.

esc/env.py::step() slides the history window and appends a zero placeholder
for the new turn on every transition -- real content only ever appears in
the root state. This script checks: for a fixed real root state, does the
value network's output at one-ply-deep simulated children actually vary by
which candidate action was taken, or is it nearly constant regardless of
action (which would mean search has little more than the transition
model's small emotion/phase/resolution deltas to work with, not the
dominant history component of the state vector)?

Usage: python -m scripts.diagnose_step4_placeholder_impact --seed 42
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from esc.action import ESCAction, generate_candidate_actions
from esc.env import ESCEnv
from esc.state import ESCState
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.train_data import ESCStateBundleDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-roots", type=int, default=5, help="How many distinct root states to test.")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints", "causal_mcts", f"seed{args.seed}")

    state_dim = ESCState.get_state_dim()
    value = ValueNetwork(state_dim=state_dim)
    transition = LinearTransitionModel(
        state_dim=state_dim,
        n_causes=ESCState.N_C,
        d_e=ESCState.D_E,
        n_strategies=ESCAction.NUM_STRATEGIES,
    )
    value.load_state_dict(torch.load(os.path.join(ckpt_dir, "value.pt"), map_location="cpu"))
    transition.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "transition.pt"), map_location="cpu")
    )
    value.eval()
    transition.eval()

    env = ESCEnv(transition_model=transition, config={})

    bundles_path = os.path.join(root, "artifacts", "states", "train.pt")
    ds = ESCStateBundleDataset(bundles_path)

    results = []
    idxs = torch.linspace(0, len(ds) - 1, args.n_roots).long().tolist()
    for root_idx in idxs:
        root_state = ds.get_state(int(root_idx))
        candidates = generate_candidate_actions(root_state)

        root_value = value.value(root_state.to_tensor())
        child_values = []
        for action in candidates:
            child_state, reward, _done, _info = env.step(root_state, action)
            v = value.value(child_state.to_tensor())
            child_values.append(
                {
                    "strategy": ESCAction.STRATEGIES[action.strategy_id],
                    "cause_index": action.cause_index,
                    "child_value": v,
                    "reward": reward,
                }
            )

        vals = torch.tensor([c["child_value"] for c in child_values])
        entry = {
            "root_idx": root_idx,
            "phase": root_state.current_phase,
            "root_value": root_value,
            "n_candidates": len(candidates),
            "child_value_mean": float(vals.mean()),
            "child_value_std": float(vals.std() if len(vals) > 1 else 0.0),
            "child_value_min": float(vals.min()),
            "child_value_max": float(vals.max()),
            "child_value_range": float(vals.max() - vals.min()),
            "children": child_values,
        }
        results.append(entry)
        print(
            f"[step4] root_idx={root_idx} phase={entry['phase']} "
            f"root_value={root_value:.4f} n_candidates={entry['n_candidates']} "
            f"child_value: mean={entry['child_value_mean']:.4f} "
            f"std={entry['child_value_std']:.6f} "
            f"range={entry['child_value_range']:.6f} "
            f"(min={entry['child_value_min']:.4f}, max={entry['child_value_max']:.4f})"
        )

    out_path = os.path.join(root, "results", "diagnostics", "step4_placeholder_impact.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[step4] wrote {out_path}")

    overall_range = max(r["child_value_range"] for r in results)
    print(
        f"\n[step4] max child_value_range across all tested roots: {overall_range:.6f} "
        "(near-zero => one-ply values are essentially action-independent)"
    )


if __name__ == "__main__":
    main()
