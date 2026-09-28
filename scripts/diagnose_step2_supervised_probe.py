"""Debug spec Step 2: does the raw representation even contain the answer?

The cheapest, most decisive experiment: no MCTS, no policy network, no
training loop. Builds (state vector, gold strategy label) pairs from real
ESConv training-split turns, then trains a plain linear classifier on two
separate inputs against the same labels:

  (a) the raw pooled Qwen embedding (pre-projection) of the dialogue
      context so far -- the same text used to build "emotion" in
      encode_dialogue, so (a) and (b) describe the same information.
  (b) the final projected state vector -- the actual
      ESCState.to_tensor() the policy network sees.

Reports held-out accuracy for both against random-guessing and
majority-class floors.

Usage: python -m scripts.diagnose_step2_supervised_probe
"""

from __future__ import annotations

import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import encoder_adapter, iter_jsonl
from esc.action import ESCAction
from esc.state import ESCState
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config

N_INSTANCES = 300
MIN_CONTEXT = 2
TRAIN_FRAC = 0.8
PROBE_EPOCHS = 300
PROBE_LR = 0.05
SEED = 0


def _build_train_split_instances(root: str, n: int) -> list[dict]:
    """Same shape as eval.data.build_eval_instances, but from the TRAIN
    split (per spec's request for "real ESConv training states"), and
    collecting every eligible turn per conversation (matching the B1 fix's
    approach) so multiple phases are represented, not just phase 0."""
    jsonl_path = os.path.join(root, "artifacts", "processed", "conversations.jsonl")
    instances: list[dict] = []
    for rec in iter_jsonl(jsonl_path):
        if rec.metadata.get("split") != "train":
            continue
        strategies = rec.annotations.get("turn_strategies") or []
        for t in range(MIN_CONTEXT, len(rec.turns)):
            if t >= len(rec.speaker_roles) or rec.speaker_roles[t] != "supporter":
                continue
            gold_strategy = strategies[t] if t < len(strategies) else None
            if gold_strategy is None:
                continue
            instances.append(
                {
                    "conversation_id": rec.conversation_id,
                    "context": list(rec.turns[:t]),
                    "gold_strategy": gold_strategy,
                }
            )
            if len(instances) >= n:
                return instances
    return instances


def _train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
) -> float:
    """Plain linear classifier (logistic regression), full-batch gradient descent."""
    torch.manual_seed(SEED)
    dim = x_train.shape[1]
    # Standardize features (zero mean, unit std per dim) using train stats only.
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    clf = nn.Linear(dim, num_classes)
    optimizer = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        logits = clf(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = clf(x_test).argmax(dim=-1)
        acc = float((preds == y_test).float().mean().item())
    return acc


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instances = _build_train_split_instances(root, N_INSTANCES)
    print(f"[step2] built {len(instances)} (context, gold_strategy) instances from train split")

    strategy_to_idx = {s: i for i, s in enumerate(ESCAction.STRATEGIES)}
    labels = [strategy_to_idx[i["gold_strategy"]] for i in instances]
    import collections

    label_counts = collections.Counter(labels)
    print(f"[step2] gold label distribution: "
          f"{ {ESCAction.STRATEGIES[k]: v for k, v in label_counts.items()} }")

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    raw_vecs = []
    proj_vecs = []
    for i, inst in enumerate(instances):
        full_text = " ".join(inst["context"])
        raw = backbone._pooled_embedding(full_text)
        raw_vecs.append(raw)

        state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        proj_vecs.append(state.to_tensor())
        if (i + 1) % 50 == 0:
            print(f"[step2] encoded {i + 1}/{len(instances)}")

    raw_x = torch.stack(raw_vecs)
    proj_x = torch.stack(proj_vecs)
    y = torch.tensor(labels, dtype=torch.long)

    n = len(instances)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * TRAIN_FRAC)
    train_idx = torch.tensor(idx[:n_train])
    test_idx = torch.tensor(idx[n_train:])

    num_classes = len(ESCAction.STRATEGIES)
    random_floor = 1.0 / num_classes
    majority_floor = max(collections.Counter(labels[i] for i in idx[n_train:]).values()) / len(test_idx)

    raw_acc = _train_probe(raw_x[train_idx], y[train_idx], raw_x[test_idx], y[test_idx], num_classes)
    proj_acc = _train_probe(proj_x[train_idx], y[train_idx], proj_x[test_idx], y[test_idx], num_classes)

    print(f"\n[step2] n_train={len(train_idx)} n_test={len(test_idx)} num_classes={num_classes}")
    print(f"[step2] random floor (1/{num_classes}): {random_floor:.4f}")
    print(f"[step2] majority-class floor (test split): {majority_floor:.4f}")
    print(f"[step2] RAW pooled embedding probe accuracy:       {raw_acc:.4f}")
    print(f"[step2] PROJECTED state-vector probe accuracy:     {proj_acc:.4f}")

    out = {
        "n_instances": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "num_classes": num_classes,
        "random_floor": random_floor,
        "majority_floor": majority_floor,
        "raw_probe_accuracy": raw_acc,
        "projected_probe_accuracy": proj_acc,
        "label_distribution": {ESCAction.STRATEGIES[k]: v for k, v in label_counts.items()},
    }
    out_path = os.path.join(root, "results", "diagnostics", "step2_supervised_probe.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[step2] wrote {out_path}")


if __name__ == "__main__":
    main()
