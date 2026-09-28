"""Run evaluation for one system + seed: strategy selection, real generation, metrics.

Systems compared:
  - causal_mcts    : trained policy/value/transition + MCTS search
  - flow_ablation   : trained policy only, argmax action (no MCTS search) --
                      the renamed "AFlow baseline" (in-house flow-matching
                      ablation, not a reproduction of published AFlow)
  - random_floor   : uniformly random legal strategy (lower bound)

All three use the SAME real Qwen backbone to generate the final reply text,
conditioned on the strategy each system selects -- isolating strategy-
selection quality as the source of any metric differences.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from data import encoder_adapter
from esc.action import ESCAction, generate_candidate_actions
from esc.env import ESCEnv
from esc.state import ESCState
from eval.data import build_eval_instances
from eval.metrics import (
    bertscore_f1,
    corpus_bleu,
    distinct_2,
    per_example_rougeL,
    rouge_scores,
    sentence_bleu_scores,
    strategy_accuracy,
)
from mcts.mcts import MCTS
from models.backbone_qwen import QwenBackbone
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.utils import load_merged_config
from utils.seed import set_global_seed

STRATEGY_NAMES = ESCAction.STRATEGIES


def _build_prompt(context: list[str], strategy_name: str) -> str:
    lines = []
    for i, t in enumerate(context):
        speaker = "Seeker" if i % 2 == 0 else "Supporter"
        lines.append(f"{speaker}: {t}")
    convo = "\n".join(lines)
    return (
        f"Conversation so far:\n{convo}\n\n"
        f"As the Supporter, respond using the '{strategy_name}' strategy. "
        "Give one supportive, on-topic reply (1-3 sentences)."
    )


def _select_strategy_causal_mcts(config, ckpt_dir, state_dim, num_actions):
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

    def _select(state: ESCState) -> ESCAction:
        return mcts.search(state)

    return _select


def _select_strategy_flow_ablation(ckpt_dir, state_dim, num_actions):
    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    policy.load_state_dict(torch.load(os.path.join(ckpt_dir, "policy.pt"), map_location="cpu"))
    policy.eval()

    def _select(state: ESCState) -> ESCAction:
        with torch.no_grad():
            probs = policy(state.to_tensor())
        idx = int(probs.argmax().item())
        n_c = ESCState.N_C
        return ESCAction(strategy_id=idx // n_c, cause_index=idx % n_c)

    return _select


def _select_strategy_random(seed: int):
    rng = torch.Generator().manual_seed(seed)

    def _select(state: ESCState) -> ESCAction:
        candidates = generate_candidate_actions(state)
        idx = int(torch.randint(0, len(candidates), (1,), generator=rng).item())
        return candidates[idx]

    return _select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system", required=True, choices=["causal_mcts", "flow_ablation", "random_floor"]
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_merged_config(root)
    set_global_seed(args.seed)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    backbone_model_name = config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    backbone = QwenBackbone(backbone_model_name, seed=args.seed)
    backbone.load()
    encoder = encoder_adapter(backbone)

    if args.system == "causal_mcts":
        ckpt_dir = os.path.join(root, "checkpoints", "causal_mcts", f"seed{args.seed}")
        select_fn = _select_strategy_causal_mcts(config, ckpt_dir, state_dim, num_actions)
    elif args.system == "flow_ablation":
        ckpt_dir = os.path.join(root, "checkpoints", "flow_ablation", f"seed{args.seed}")
        select_fn = _select_strategy_flow_ablation(ckpt_dir, state_dim, num_actions)
    else:
        select_fn = _select_strategy_random(args.seed)

    instances = build_eval_instances(root, n=args.n)
    print(f"[eval] system={args.system} seed={args.seed} n_instances={len(instances)}")

    predictions: list[str] = []
    references: list[str] = []
    pred_strategies: list[str] = []
    gold_strategies: list[str] = []

    for i, inst in enumerate(instances):
        state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        action = select_fn(state)
        strategy_name = STRATEGY_NAMES[action.strategy_id]
        prompt = _build_prompt(inst["context"], strategy_name)
        reply = backbone.generate_response(prompt)

        predictions.append(reply)
        references.append(inst["gold_reply"])
        pred_strategies.append(strategy_name)
        gold_strategies.append(inst["gold_strategy"])
        print(
            f"[eval] {args.system} seed={args.seed} {i + 1}/{len(instances)} "
            f"strategy={strategy_name} gold={inst['gold_strategy']}"
        )

    bleu = corpus_bleu(predictions, references)
    sent_bleu = sentence_bleu_scores(predictions, references)
    rouge = rouge_scores(predictions, references)
    rougeL_per_ex = per_example_rougeL(predictions, references)
    bert_f1 = bertscore_f1(predictions, references)
    dist2 = distinct_2(predictions)
    strat_acc = strategy_accuracy(pred_strategies, gold_strategies)

    result = {
        "system": args.system,
        "seed": args.seed,
        "n_instances": len(instances),
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bertscore_f1_mean": (sum(bert_f1) / len(bert_f1)) if bert_f1 else 0.0,
        "distinct_2": dist2,
        "strategy_accuracy": strat_acc,
        "per_example": {
            "sentence_bleu": sent_bleu,
            "rougeL": rougeL_per_ex,
            "bertscore_f1": bert_f1,
            "predicted_strategy": pred_strategies,
            "gold_strategy": gold_strategies,
            "prediction": predictions,
            "reference": references,
        },
    }

    out_dir = os.path.join(root, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.system}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
