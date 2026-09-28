"""Bounded latency comparison: learned-transition MCTS vs. an LLM-rollout MCTS proxy.

The paper's core latency claim -- that replacing LLM rollouts with a learned
transition model speeds up MCTS decisions -- had zero validation anywhere in
the repository. This is a small, bounded, real measurement of it:

  - learned_transition: MCTS.search() using the trained LinearTransitionModel
    (no LLM calls at all during search -- confirmed by inspection of
    esc/env.py::step and mcts/mcts.py::_simulate).
  - llm_rollout_proxy: for the same root states, call the real Qwen backbone's
    generate_response() once per top-3 policy-prior candidate action (a bounded
    proxy for what an LLM-rollout MCTS simulation would need to do to evaluate
    candidates at a node -- not a full multi-level LLM-rollout tree search,
    which would be far too slow on CPU to bound).

Capped at 10 decisions, <= 3 LLM calls each (<=30 total), to keep this phase
short. Writes results/latency/latency_baseline.json.
"""

from __future__ import annotations

import json
import os
import time

import torch

from data import encoder_adapter
from esc.action import ESCAction, generate_candidate_actions
from esc.env import ESCEnv
from esc.state import ESCState
from eval.data import build_eval_instances
from mcts.mcts import MCTS
from models.backbone_qwen import QwenBackbone
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.utils import load_merged_config
from utils.seed import set_global_seed

N_DECISIONS = 10
TOP_K_CANDIDATES = 3
SEED = 42


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_merged_config(root)
    set_global_seed(SEED)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    backbone_model_name = config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    backbone = QwenBackbone(backbone_model_name, seed=SEED)
    backbone.load()
    encoder = encoder_adapter(backbone)

    ckpt_dir = os.path.join(root, "checkpoints", "causal_mcts", f"seed{SEED}")
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

    instances = build_eval_instances(root, n=N_DECISIONS)
    states = [
        ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        for inst in instances
    ]

    learned_times: list[float] = []
    llm_times: list[float] = []
    llm_calls_total = 0

    for state in states:
        t0 = time.perf_counter()
        mcts.search(state)
        learned_times.append(time.perf_counter() - t0)

        candidates = generate_candidate_actions(state)
        with torch.no_grad():
            priors = policy(state.to_tensor())
        # priors is sized to the network's action_dim; align length defensively
        k = min(TOP_K_CANDIDATES, len(candidates))
        order = list(range(len(candidates)))
        if priors.shape[0] == len(candidates):
            order = sorted(order, key=lambda i: -float(priors[i]))
        top_candidates = [candidates[i] for i in order[:k]]

        t0 = time.perf_counter()
        for action in top_candidates:
            strategy_name = ESCAction.STRATEGIES[action.strategy_id]
            prompt = f"As the Supporter, using the '{strategy_name}' strategy, what do you say next?"
            backbone.generate_response(prompt)
            llm_calls_total += 1
        llm_times.append(time.perf_counter() - t0)

    def _stats(xs: list[float]) -> dict[str, float]:
        return {
            "mean_seconds": sum(xs) / len(xs) if xs else 0.0,
            "min_seconds": min(xs) if xs else 0.0,
            "max_seconds": max(xs) if xs else 0.0,
        }

    learned_stats = _stats(learned_times)
    llm_stats = _stats(llm_times)
    speedup = (
        llm_stats["mean_seconds"] / learned_stats["mean_seconds"]
        if learned_stats["mean_seconds"] > 0
        else float("inf")
    )

    result = {
        "n_decisions": len(states),
        "learned_transition_mcts": {
            "num_simulations": config.get("num_simulations"),
            **learned_stats,
            "per_decision_seconds": learned_times,
        },
        "llm_rollout_proxy": {
            "top_k_candidates_per_decision": TOP_K_CANDIDATES,
            "total_llm_calls": llm_calls_total,
            **llm_stats,
            "per_decision_seconds": llm_times,
        },
        "speedup_llm_over_learned": speedup,
        "note": (
            "llm_rollout_proxy times a bounded proxy (top-3 candidate generate_response "
            "calls at the root), not a full LLM-rollout MCTS tree search -- that would be "
            "far too slow to bound on CPU. See eval/latency_baseline.py docstring."
        ),
    }

    out_dir = os.path.join(root, "results", "latency")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latency_baseline.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[latency] learned_transition mean={learned_stats['mean_seconds']:.4f}s")
    print(f"[latency] llm_rollout_proxy mean={llm_stats['mean_seconds']:.4f}s")
    print(f"[latency] speedup={speedup:.1f}x, wrote {out_path}")


if __name__ == "__main__":
    main()
