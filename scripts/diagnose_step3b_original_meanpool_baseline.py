"""Where did the ORIGINAL (broken) mean-pooling state vector stand at the
same properly-powered test size (n_test=1700) used in step3's significance
test for select/whole-window?

Reuses the IDENTICAL deterministic instance-selection and train/test split
as scripts/diagnose_step3_significance_test.py (same N_TRAIN, N_TEST,
MIN_CONTEXT, SEED) so this is a true paired comparison against the same
majority-class floor and the same test examples -- not a new sample.

Only needs the standard encoding path (mean-pooling reads the same raw
per-turn/per-cause matrices "select" does; no whole-window pass needed).

Usage: python -m scripts.diagnose_step3b_original_meanpool_baseline
"""

from __future__ import annotations

import collections
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

from data import encoder_adapter, iter_jsonl
from esc.action import ESCAction
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
    from scipy.stats import binomtest
    b = int(((baseline_correct == 1) & (method_correct == 0)).sum().item())
    c = int(((baseline_correct == 0) & (method_correct == 1)).sum().item())
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    result = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": float(result.pvalue)}


def _paired_bootstrap_diff(method_correct: torch.Tensor, baseline_correct: torch.Tensor, seed: int = 42, n_resamples: int = 10000) -> dict:
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
    print(f"[step3b] built {len(instances)} instances (target {N_TOTAL}) -- must match step3's {N_TOTAL}")

    strategy_to_idx = {s: i for i, s in enumerate(ESCAction.STRATEGIES)}
    labels = [strategy_to_idx[i["gold_strategy"]] for i in instances]

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    N_C, D_C = ESCState.N_C, ESCState.D_C

    meanpool_vecs = []
    t0 = time.perf_counter()
    for i, inst in enumerate(instances):
        state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        h_mean = state.history_embeddings.mean(dim=0)
        c_matrix = state.causal_graph.cause_embedding_matrix(N_C, D_C)
        c_mean = c_matrix.mean(dim=0)
        meanpool_vecs.append(torch.cat([h_mean, c_mean, state.emotion_vector, state.phase_embedding], dim=0))

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[step3b] encoded {i + 1}/{len(instances)}  ({elapsed / (i + 1):.3f}s/instance, "
                  f"elapsed {elapsed:.0f}s)")

    encode_elapsed = time.perf_counter() - t0
    print(f"[step3b] encoding done in {encode_elapsed:.1f}s ({encode_elapsed / 60:.1f} min)")

    meanpool_x = torch.stack(meanpool_vecs)
    y = torch.tensor(labels, dtype=torch.long)

    n = len(instances)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    train_idx = torch.tensor(idx[:N_TRAIN])
    test_idx = torch.tensor(idx[N_TRAIN:N_TRAIN + N_TEST])
    print(f"[step3b] n_train={len(train_idx)} n_test={len(test_idx)} (must match step3's split exactly)")

    y_test = y[test_idx]
    test_label_counts = collections.Counter(y_test.tolist())
    majority_class, majority_count = test_label_counts.most_common(1)[0]
    majority_floor = majority_count / len(test_idx)
    majority_preds = torch.full_like(y_test, majority_class)
    majority_correct = (majority_preds == y_test).long()
    print(f"[step3b] majority class: {ESCAction.STRATEGIES[majority_class]}  floor: {majority_floor:.4f}  "
          f"(must match step3's 'Others' / 0.1729)")

    preds = _train_linear_probe(meanpool_x[train_idx], y[train_idx], meanpool_x[test_idx], y_test)
    correct = (preds == y_test).long()
    acc = float(correct.float().mean())
    mcnemar = _mcnemar_exact(correct, majority_correct)
    bootstrap = _paired_bootstrap_diff(correct, majority_correct)

    print(f"\n[step3b] mean_pool (original): acc={acc:.4f} gap={acc - majority_floor:+.4f}")
    print(f"[step3b]   McNemar: b={mcnemar['b']} c={mcnemar['c']} "
          f"n_discordant={mcnemar['n_discordant']} p={mcnemar['p_value']:.4f}")
    print(f"[step3b]   Bootstrap: mean_diff={bootstrap['mean_diff']:+.4f} "
          f"95% CI=[{bootstrap['ci_lo']:+.4f}, {bootstrap['ci_hi']:+.4f}] p={bootstrap['bootstrap_p']:.4f}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[step3b] TOTAL wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    out = {
        "n_instances": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "majority_floor": majority_floor,
        "majority_class": ESCAction.STRATEGIES[majority_class],
        "original_meanpool": {
            "accuracy": acc,
            "gap": acc - majority_floor,
            "mcnemar": mcnemar,
            "bootstrap": bootstrap,
        },
        "total_wall_seconds": total_elapsed,
    }
    out_path = os.path.join(root, "results", "diagnostics", "step3b_original_meanpool_baseline.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[step3b] wrote {out_path}")


if __name__ == "__main__":
    main()
