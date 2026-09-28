"""Verify Step 2's byte-identical before/after numbers were a genuine
statistical coincidence, not a caching bug -- by computing BOTH the
old-style (mean-pooled) and new-style (selected) projected vectors from
the SAME encoded states in ONE process, eliminating any run-to-run
confound (stale bytecode, non-determinism, etc).

Usage: python -m scripts.diagnose_step2b_verify_aggregation_change
"""

from __future__ import annotations

import collections
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
                {"conversation_id": rec.conversation_id, "context": list(rec.turns[:t]), "gold_strategy": gold_strategy}
            )
            if len(instances) >= n:
                return instances
    return instances


def _train_probe(x_train, y_train, x_test, y_test, num_classes):
    torch.manual_seed(SEED)
    dim = x_train.shape[1]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    clf = nn.Linear(dim, num_classes)
    optimizer = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        loss = F.cross_entropy(clf(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = clf(x_test).argmax(dim=-1)
        acc = float((preds == y_test).float().mean().item())
    return acc, preds


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instances = _build_train_split_instances(root, N_INSTANCES)
    print(f"[step2b] built {len(instances)} instances")

    strategy_to_idx = {s: i for i, s in enumerate(ESCAction.STRATEGIES)}
    labels = [strategy_to_idx[i["gold_strategy"]] for i in instances]

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    new_vecs = []   # current to_tensor(): selection-based
    old_vecs = []   # manually reconstructed: mean-based (pre-fix behavior)
    for i, inst in enumerate(instances):
        state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        new_vecs.append(state.to_tensor())

        h_mean = state.history_embeddings.mean(dim=0)
        c_matrix = state.causal_graph.cause_embedding_matrix(ESCState.N_C, ESCState.D_C)
        c_mean = c_matrix.mean(dim=0)
        old_vecs.append(torch.cat([h_mean, c_mean, state.emotion_vector, state.phase_embedding], dim=0))

        if (i + 1) % 50 == 0:
            print(f"[step2b] encoded {i + 1}/{len(instances)}")

    new_x = torch.stack(new_vecs)
    old_x = torch.stack(old_vecs)
    y = torch.tensor(labels, dtype=torch.long)

    # Sanity: confirm the two feature matrices actually differ.
    max_abs_diff = (new_x - old_x).abs().max().item()
    identical = torch.allclose(new_x, old_x)
    print(f"\n[step2b] new_x vs old_x max abs diff: {max_abs_diff:.6f}  identical: {identical}")

    n = len(instances)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * TRAIN_FRAC)
    train_idx = torch.tensor(idx[:n_train])
    test_idx = torch.tensor(idx[n_train:])
    num_classes = len(ESCAction.STRATEGIES)
    majority_floor = max(collections.Counter(labels[i] for i in idx[n_train:]).values()) / len(test_idx)

    old_acc, old_preds = _train_probe(old_x[train_idx], y[train_idx], old_x[test_idx], y[test_idx], num_classes)
    new_acc, new_preds = _train_probe(new_x[train_idx], y[train_idx], new_x[test_idx], y[test_idx], num_classes)

    preds_identical = torch.equal(old_preds, new_preds)
    print(f"\n[step2b] majority floor: {majority_floor:.4f}")
    print(f"[step2b] OLD (mean-pooled) probe accuracy: {old_acc:.4f}")
    print(f"[step2b] NEW (selection)   probe accuracy: {new_acc:.4f}")
    print(f"[step2b] test-set predictions identical between old/new: {preds_identical}")
    if preds_identical:
        print("[step2b] old_preds:", old_preds.tolist())
        print("[step2b] new_preds:", new_preds.tolist())

    out = {
        "max_abs_feature_diff": max_abs_diff,
        "features_identical": identical,
        "old_mean_probe_accuracy": old_acc,
        "new_selection_probe_accuracy": new_acc,
        "majority_floor": majority_floor,
        "predictions_identical": preds_identical,
    }
    out_path = os.path.join(root, "results", "diagnostics", "step2b_verify_aggregation_change.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[step2b] wrote {out_path}")


if __name__ == "__main__":
    main()
