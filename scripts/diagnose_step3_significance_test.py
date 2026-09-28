"""Is select's / whole-window's ~22.5% probe accuracy real, or noise from
a small test set?

The prior 5-candidate comparison used n_test=200 (noise band ~2.8 points
at the ~20.5% base rate) -- too wide to tell a 2-point gap over the
majority-class floor apart from chance. This script grows the TEST set
specifically (linear probes don't need much training data) to n_test=1700
(~1 point noise band), keeps n_train=800, and runs McNemar's exact test
plus a paired bootstrap CI for each method against the majority-class
predictor on the SAME test examples.

Only tests select and whole-window (concatenate, max-pool, attention-pool
already reasonably ruled out per the 5-candidate comparison).

Usage: python -m scripts.diagnose_step3_significance_test
"""

from __future__ import annotations

import collections
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from torch import nn

from data import encoder_adapter, iter_jsonl
from esc.action import ESCAction
from esc.state import ESCState
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config

N_TRAIN = 800
N_TEST = 1700
N_TOTAL = N_TRAIN + N_TEST
MIN_CONTEXT = 2
PROBE_EPOCHS = 300
PROBE_LR = 0.05
SEED = 0
N_BOOTSTRAP = 10000


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
            instances.append(
                {"context": list(rec.turns[:t]), "gold_strategy": gold_strategy}
            )
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


def _mcnemar_exact(method_correct: torch.Tensor, baseline_correct: torch.Tensor) -> dict:
    """Exact McNemar's test on paired binary correctness vectors."""
    b = int(((baseline_correct == 1) & (method_correct == 0)).sum().item())  # baseline right, method wrong
    c = int(((baseline_correct == 0) & (method_correct == 1)).sum().item())  # method right, baseline wrong
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    result = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": float(result.pvalue)}


def _paired_bootstrap_diff(method_correct: torch.Tensor, baseline_correct: torch.Tensor, seed: int = 42, n_resamples: int = N_BOOTSTRAP) -> dict:
    diffs = (method_correct.float() - baseline_correct.float())
    n = len(diffs)
    rng = random.Random(seed)
    obs_mean = float(diffs.mean())
    resample_means = []
    diffs_list = diffs.tolist()
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
    print(f"[step3] built {len(instances)} instances (target {N_TOTAL})")

    strategy_to_idx = {s: i for i, s in enumerate(ESCAction.STRATEGIES)}
    labels = [strategy_to_idx[i["gold_strategy"]] for i in instances]
    print("[step3] label distribution:", {ESCAction.STRATEGIES[k]: v for k, v in collections.Counter(labels).items()})

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    D_C = ESCState.D_C
    K, N_C = ESCState.K_HISTORY_WINDOW, ESCState.N_C

    select_vecs, whole_window_vecs = [], []
    t0 = time.perf_counter()
    for i, inst in enumerate(instances):
        state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        h_selected = state.history_embeddings[-1]
        c_matrix = state.causal_graph.cause_embedding_matrix(N_C, D_C)
        c_selected = c_matrix[0]
        select_vecs.append(torch.cat([h_selected, c_selected, state.emotion_vector, state.phase_embedding], dim=0))

        joined_history = " ".join(inst["context"][-K:])
        wh = backbone._proj_history(backbone._pooled_embedding(joined_history))
        seeker_turns = [t for j, t in enumerate(inst["context"]) if j % 2 == 0 and t.strip()]
        candidate_spans = list(reversed(seeker_turns))[:N_C]
        joined_causes = " ".join(candidate_spans) if candidate_spans else ""
        wc = backbone._proj_cause(backbone._pooled_embedding(joined_causes))
        whole_window_vecs.append(torch.cat([wh, wc, state.emotion_vector, state.phase_embedding], dim=0))

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[step3] encoded {i + 1}/{len(instances)}  ({elapsed / (i + 1):.3f}s/instance, "
                  f"elapsed {elapsed:.0f}s)")

    encode_elapsed = time.perf_counter() - t0
    print(f"[step3] encoding done in {encode_elapsed:.1f}s ({encode_elapsed / 60:.1f} min)")

    select_x = torch.stack(select_vecs)
    whole_window_x = torch.stack(whole_window_vecs)
    y = torch.tensor(labels, dtype=torch.long)

    n = len(instances)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    train_idx = torch.tensor(idx[:N_TRAIN])
    test_idx = torch.tensor(idx[N_TRAIN:N_TRAIN + N_TEST])
    print(f"[step3] n_train={len(train_idx)} n_test={len(test_idx)}")

    y_test = y[test_idx]
    test_label_counts = collections.Counter(y_test.tolist())
    majority_class, majority_count = test_label_counts.most_common(1)[0]
    majority_floor = majority_count / len(test_idx)
    majority_preds = torch.full_like(y_test, majority_class)
    majority_correct = (majority_preds == y_test).long()
    print(f"[step3] majority class: {ESCAction.STRATEGIES[majority_class]}  floor: {majority_floor:.4f}")

    results = {}
    for name, x in [("select", select_x), ("whole_window", whole_window_x)]:
        preds = _train_linear_probe(x[train_idx], y[train_idx], x[test_idx], y_test)
        correct = (preds == y_test).long()
        acc = float(correct.float().mean())
        mcnemar = _mcnemar_exact(correct, majority_correct)
        bootstrap = _paired_bootstrap_diff(correct, majority_correct)
        results[name] = {
            "accuracy": acc,
            "majority_floor": majority_floor,
            "gap": acc - majority_floor,
            "mcnemar": mcnemar,
            "bootstrap": bootstrap,
        }
        print(f"\n[step3] {name}: acc={acc:.4f} gap={acc - majority_floor:+.4f}")
        print(f"[step3]   McNemar: b={mcnemar['b']} c={mcnemar['c']} "
              f"n_discordant={mcnemar['n_discordant']} p={mcnemar['p_value']:.4f}")
        print(f"[step3]   Bootstrap: mean_diff={bootstrap['mean_diff']:+.4f} "
              f"95% CI=[{bootstrap['ci_lo']:+.4f}, {bootstrap['ci_hi']:+.4f}] p={bootstrap['bootstrap_p']:.4f}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[step3] TOTAL wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    out = {
        "n_instances": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "majority_floor": majority_floor,
        "majority_class": ESCAction.STRATEGIES[majority_class],
        "results": results,
        "total_wall_seconds": total_elapsed,
    }
    out_path = os.path.join(root, "results", "diagnostics", "step3_significance_test.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[step3] wrote {out_path}")


if __name__ == "__main__":
    main()
