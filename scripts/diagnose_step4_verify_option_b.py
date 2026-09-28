"""Verify Option B (whole-window with repeated-row [K,D_H]/[N_C,D_C]
compatibility shim, wired into models/backbone_qwen.py::encode_dialogue
and esc/state.py::from_dialogue) against the same 800-train/1700-test
split used in scripts/diagnose_step3_significance_test.py.

"select" here is recomputed via the OLD manual per-turn encoding (the
production encode_dialogue no longer offers that path -- it's whole-window
now by construction), matching how it was computed before, to stay a fair
comparison point. "whole_window" is computed via the ACTUAL PRODUCTION
code path (ESCState.from_dialogue with the real encoder), not a manual
bypass -- this is the real thing Step 2 needs to verify.

Usage: python -m scripts.diagnose_step4_verify_option_b
"""

from __future__ import annotations

import collections
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest

from data import encoder_adapter, iter_jsonl
from esc.action import ESCAction
from esc.causal_graph import CausalGraph
from esc.state import ESCState
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config

# Must match scripts/diagnose_step3_significance_test.py exactly.
N_TRAIN = 800
N_TEST = 1700
N_TOTAL = N_TRAIN + N_TEST
MIN_CONTEXT = 2
PROBE_EPOCHS = 300
PROBE_LR = 0.05
SEED = 0


def _build_instances(root: str, n: int) -> list[dict]:
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
            instances.append({"context": list(rec.turns[:t]), "gold_strategy": gold_strategy})
            if len(instances) >= n:
                return instances
    return instances


def _train_linear_probe(x_train, y_train, x_test, y_test):
    torch.manual_seed(SEED)
    dim = x_train.shape[1]
    num_classes = int(y_train.max().item()) + 1
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train_n = (x_train - mean) / std
    x_test_n = (x_test - mean) / std
    clf = nn.Linear(dim, num_classes)
    optimizer = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        loss = F.cross_entropy(clf(x_train_n), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = clf(x_test_n).argmax(dim=-1)
    return preds


def _mcnemar_exact(method_correct, baseline_correct) -> dict:
    b = int(((baseline_correct == 1) & (method_correct == 0)).sum().item())
    c = int(((baseline_correct == 0) & (method_correct == 1)).sum().item())
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    result = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": float(result.pvalue)}


def _paired_bootstrap_diff(method_correct, baseline_correct, seed=42, n_resamples=10000) -> dict:
    diffs = (method_correct.float() - baseline_correct.float())
    n = len(diffs)
    rng = random.Random(seed)
    obs_mean = float(diffs.mean())
    diffs_list = diffs.tolist()
    resample_means = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        resample_means.append(sum(diffs_list[i] for i in idx) / n)
    resample_means.sort()
    lo = resample_means[int(0.025 * n_resamples)]
    hi = resample_means[min(int(0.975 * n_resamples), n_resamples - 1)]
    p_value = 2 * min(
        sum(1 for m in resample_means if m <= 0) / n_resamples,
        sum(1 for m in resample_means if m >= 0) / n_resamples,
    )
    return {"mean_diff": obs_mean, "ci_lo": lo, "ci_hi": hi, "bootstrap_p": min(p_value, 1.0)}


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_start = time.perf_counter()

    instances = _build_instances(root, N_TOTAL)
    print(f"[step4v] built {len(instances)} instances (target {N_TOTAL})")

    strategy_to_idx = {s: i for i, s in enumerate(ESCAction.STRATEGIES)}
    labels = [strategy_to_idx[i["gold_strategy"]] for i in instances]

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    D_H, D_C = ESCState.D_H, ESCState.D_C
    K, N_C = ESCState.K_HISTORY_WINDOW, ESCState.N_C

    select_vecs, whole_window_vecs = [], []
    t0 = time.perf_counter()
    for i, inst in enumerate(instances):
        # "select" -- OLD manual per-turn encoding (production no longer does this).
        recent = inst["context"][-K:]
        history_matrix = torch.zeros(K, D_H)
        start_row = K - len(recent)
        for j, turn_text in enumerate(recent):
            history_matrix[start_row + j] = backbone._proj_history(backbone._pooled_embedding(turn_text))
        seeker_turns = [t for j, t in enumerate(inst["context"]) if j % 2 == 0 and t.strip()]
        candidate_spans = list(reversed(seeker_turns))[:N_C]
        old_cause_graph = CausalGraph(d_c=D_C)
        for span in candidate_spans:
            old_cause_graph.add_cause(label=span[:60], embedding=backbone._proj_cause(backbone._pooled_embedding(span)))
        while old_cause_graph.num_causes < N_C:
            old_cause_graph.add_cause(label="pad", embedding=torch.zeros(D_C))
        old_c_matrix = old_cause_graph.cause_embedding_matrix(N_C, D_C)

        full_text = " ".join(inst["context"]) if inst["context"] else ""
        emotion = torch.tanh(backbone._proj_emotion(backbone._pooled_embedding(full_text)))

        # Need phase too -- build via from_dialogue with encoder=None then override,
        # simplest: replicate the turn-count heuristic directly.
        n_turns = len(inst["context"])
        phase = torch.zeros(3)
        phase[0 if n_turns <= 11 else (1 if n_turns <= 20 else 2)] = 1.0

        select_vecs.append(torch.cat([history_matrix[-1], old_c_matrix[0], emotion, phase], dim=0))

        # "whole_window" -- ACTUAL PRODUCTION CODE PATH.
        state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        ww_c_matrix = state.causal_graph.cause_embedding_matrix(N_C, D_C)
        whole_window_vecs.append(
            torch.cat([state.history_embeddings[-1], ww_c_matrix[0], state.emotion_vector, state.phase_embedding], dim=0)
        )

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[step4v] encoded {i + 1}/{len(instances)}  ({elapsed / (i + 1):.3f}s/instance, "
                  f"elapsed {elapsed:.0f}s)")

    encode_elapsed = time.perf_counter() - t0
    print(f"[step4v] encoding done in {encode_elapsed:.1f}s ({encode_elapsed / 60:.1f} min)")

    select_x = torch.stack(select_vecs)
    whole_window_x = torch.stack(whole_window_vecs)
    y = torch.tensor(labels, dtype=torch.long)

    n = len(instances)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    train_idx = torch.tensor(idx[:N_TRAIN])
    test_idx = torch.tensor(idx[N_TRAIN:N_TRAIN + N_TEST])
    print(f"[step4v] n_train={len(train_idx)} n_test={len(test_idx)}")

    y_test = y[test_idx]
    test_label_counts = collections.Counter(y_test.tolist())
    majority_class, majority_count = test_label_counts.most_common(1)[0]
    majority_floor = majority_count / len(test_idx)
    majority_correct = (torch.full_like(y_test, majority_class) == y_test).long()
    print(f"[step4v] majority class: {ESCAction.STRATEGIES[majority_class]}  floor: {majority_floor:.4f}")

    results = {}
    for name, x in [("select", select_x), ("whole_window", whole_window_x)]:
        preds = _train_linear_probe(x[train_idx], y[train_idx], x[test_idx], y_test)
        correct = (preds == y_test).long()
        acc = float(correct.float().mean())
        mcnemar = _mcnemar_exact(correct, majority_correct)
        bootstrap = _paired_bootstrap_diff(correct, majority_correct)
        results[name] = {"accuracy": acc, "gap": acc - majority_floor, "mcnemar": mcnemar, "bootstrap": bootstrap}
        print(f"\n[step4v] {name}: acc={acc:.4f} gap={acc - majority_floor:+.4f}")
        print(f"[step4v]   McNemar: p={mcnemar['p_value']:.4f}")
        print(f"[step4v]   Bootstrap: 95% CI=[{bootstrap['ci_lo']:+.4f}, {bootstrap['ci_hi']:+.4f}] p={bootstrap['bootstrap_p']:.4f}")

    # Direct select vs whole_window comparison too.
    sel_correct = (_train_linear_probe(select_x[train_idx], y[train_idx], select_x[test_idx], y_test) == y_test).long()
    ww_correct = (_train_linear_probe(whole_window_x[train_idx], y[train_idx], whole_window_x[test_idx], y_test) == y_test).long()
    head_to_head = _mcnemar_exact(ww_correct, sel_correct)
    print(f"\n[step4v] whole_window vs select head-to-head McNemar: p={head_to_head['p_value']:.4f} "
          f"(b={head_to_head['b']} c={head_to_head['c']})")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[step4v] TOTAL wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    out = {
        "n_instances": n, "n_train": len(train_idx), "n_test": len(test_idx),
        "majority_floor": majority_floor, "majority_class": ESCAction.STRATEGIES[majority_class],
        "results": results, "head_to_head_mcnemar": head_to_head, "total_wall_seconds": total_elapsed,
    }
    out_path = os.path.join(root, "results", "diagnostics", "step4_verify_option_b.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[step4v] wrote {out_path}")


if __name__ == "__main__":
    main()
