"""Step 3 diagnostic: collect real Q-value / PUCT data from ONE MCTS episode.

Loads causal_mcts seed=42's trained checkpoint (no retraining), runs
MCTS.search() once on a single real eval instance, and dumps every
select_child() decision (via mcts/node.py's temporary PUCT_DEBUG hook) to
diagnostics/puct_tree_analysis.json. Read-only w.r.t. checkpoints; does not
touch training state or write to checkpoints/.
"""

from __future__ import annotations

import json
import os

import torch

from data import encoder_adapter
from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from eval.data import build_eval_instances
from models.backbone_qwen import QwenBackbone
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from mcts import node as node_mod
from mcts.mcts import MCTS
from train.utils import load_merged_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    config = load_merged_config(ROOT)
    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    ckpt_dir = os.path.join(ROOT, "checkpoints", "causal_mcts", "seed42")
    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    transition = LinearTransitionModel(
        state_dim=state_dim,
        n_causes=ESCState.N_C,
        d_e=ESCState.D_E,
        n_strategies=ESCAction.NUM_STRATEGIES,
    )
    policy.load_state_dict(torch.load(os.path.join(ckpt_dir, "policy.pt"), map_location="cpu"))
    value.load_state_dict(torch.load(os.path.join(ckpt_dir, "value.pt"), map_location="cpu"))
    transition.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "transition.pt"), map_location="cpu")
    )
    policy.eval()
    value.eval()
    transition.eval()

    env = ESCEnv(transition_model=transition, config=config)
    mcts = MCTS(env=env, policy_network=policy, value_network=value, config=config)
    print(f"[diagnose_step3] c_puct={mcts._c_puct} num_simulations={mcts._num_simulations}")

    backbone_model_name = config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    backbone = QwenBackbone(backbone_model_name, seed=42)
    backbone.load()
    encoder = encoder_adapter(backbone)

    instances = build_eval_instances(ROOT, n=1)
    assert instances, "No eval instances found"
    inst = instances[0]
    state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)

    node_mod.PUCT_DEBUG_LOG.clear()
    node_mod.PUCT_DEBUG = True
    try:
        action = mcts.search(state)
    finally:
        node_mod.PUCT_DEBUG = False

    log = node_mod.PUCT_DEBUG_LOG
    all_q = [c["Q"] for entry in log for c in entry["candidates"]]
    all_u = [c["exploration_term"] for entry in log for c in entry["candidates"]]
    all_puct = [c["puct"] for entry in log for c in entry["candidates"]]

    summary = {
        "checkpoint": ckpt_dir,
        "c_puct": mcts._c_puct,
        "num_simulations": mcts._num_simulations,
        "conversation_id": inst["conversation_id"],
        "turn_index": inst["turn_index"],
        "gold_strategy": inst["gold_strategy"],
        "selected_action": repr(action),
        "n_select_child_calls": len(log),
        "Q_min": min(all_q) if all_q else None,
        "Q_mean": (sum(all_q) / len(all_q)) if all_q else None,
        "Q_max": max(all_q) if all_q else None,
        "exploration_term_min": min(all_u) if all_u else None,
        "exploration_term_mean": (sum(all_u) / len(all_u)) if all_u else None,
        "exploration_term_max": max(all_u) if all_u else None,
        "puct_min": min(all_puct) if all_puct else None,
        "puct_mean": (sum(all_puct) / len(all_puct)) if all_puct else None,
        "puct_max": max(all_puct) if all_puct else None,
        "full_log": log,
    }

    out_dir = os.path.join(ROOT, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "puct_tree_analysis.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[diagnose_step3] wrote {out_path}")
    print(f"[diagnose_step3] n_select_child_calls={len(log)}")
    print(f"[diagnose_step3] Q: min={summary['Q_min']:.4f} mean={summary['Q_mean']:.4f} max={summary['Q_max']:.4f}")
    print(
        f"[diagnose_step3] exploration_term: min={summary['exploration_term_min']:.4f} "
        f"mean={summary['exploration_term_mean']:.4f} max={summary['exploration_term_max']:.4f}"
    )
    print(
        f"[diagnose_step3] puct: min={summary['puct_min']:.4f} "
        f"mean={summary['puct_mean']:.4f} max={summary['puct_max']:.4f}"
    )


if __name__ == "__main__":
    main()
