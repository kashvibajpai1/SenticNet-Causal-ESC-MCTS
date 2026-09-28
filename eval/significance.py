"""Cross-seed variance + paired significance testing.

Library functions (paired_bootstrap, wilcoxon_test) plus a CLI
(`python -m eval.significance`) that aggregates results/metrics/*.json
into results/metrics/summary.json: mean +/- std per system across seeds,
and pairwise paired-bootstrap / Wilcoxon tests on the seed=42 per-example
scores (same held-out instances across systems, so the pairing is exact).
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics
from collections.abc import Sequence

from scipy.stats import binomtest, ttest_rel, wilcoxon

SYSTEMS = ["causal_mcts", "flow_ablation", "random_floor"]
SEEDS = [42, 43, 44]
HEADLINE_METRICS = [
    "bleu",
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_f1_mean",
    "distinct_2",
    "strategy_accuracy",
]


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    assert len(scores_a) == len(scores_b) and len(scores_a) > 0
    n = len(scores_a)
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    obs_mean = sum(diffs) / n

    resample_means = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        resample_means.append(sum(diffs[i] for i in idx) / n)
    resample_means.sort()
    lo = resample_means[int(0.025 * n_resamples)]
    hi = resample_means[min(int(0.975 * n_resamples), n_resamples - 1)]
    p_value = 2 * min(
        sum(1 for m in resample_means if m <= 0) / n_resamples,
        sum(1 for m in resample_means if m >= 0) / n_resamples,
    )
    return {"mean_diff": obs_mean, "ci_lo": lo, "ci_hi": hi, "bootstrap_p": min(p_value, 1.0)}


def wilcoxon_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict[str, float]:
    try:
        stat, p = wilcoxon(scores_a, scores_b)
        return {"statistic": float(stat), "p_value": float(p)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0}


def mcnemar_test(correct_a: Sequence[float], correct_b: Sequence[float]) -> dict[str, float]:
    """Exact McNemar test on paired binary (correct/incorrect) outcomes.

    Only the discordant pairs (a right/b wrong, or a wrong/b right) carry
    information about which system tends to win; the exact binomial test on
    those counts is the textbook-correct significance test for this design
    (same instance evaluated by both systems), and is far more sensitive
    than an unpaired/pooled proportions test since it cancels out
    per-instance difficulty as a nuisance variable.
    """
    assert len(correct_a) == len(correct_b) and len(correct_a) > 0
    b = sum(1 for a, c in zip(correct_a, correct_b) if a and not c)
    c = sum(1 for a, c in zip(correct_a, correct_b) if not a and c)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    p_value = binomtest(b, n, 0.5).pvalue
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(p_value)}


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_dir = os.path.join(root, "results", "metrics")

    per_system: dict[str, dict[int, dict]] = {s: {} for s in SYSTEMS}
    for path in glob.glob(os.path.join(metrics_dir, "*_seed*.json")):
        with open(path) as f:
            data = json.load(f)
        sys_name = data["system"]
        seed = data["seed"]
        if sys_name in per_system:
            per_system[sys_name][seed] = data

    summary: dict = {"per_system_mean_std": {}, "pairwise_significance": {}}

    for sys_name in SYSTEMS:
        runs = per_system[sys_name]
        if not runs:
            continue
        entry = {}
        for metric in HEADLINE_METRICS:
            vals = [runs[s][metric] for s in runs if metric in runs[s]]
            if not vals:
                continue
            entry[metric] = {
                "mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
                "values": vals,
            }
        summary["per_system_mean_std"][sys_name] = entry

    seed_for_sig = 42
    if all(seed_for_sig in per_system[s] for s in SYSTEMS):
        for a, b in [("causal_mcts", "flow_ablation"), ("causal_mcts", "random_floor")]:
            data_a = per_system[a][seed_for_sig]["per_example"]
            data_b = per_system[b][seed_for_sig]["per_example"]
            pair_key = f"{a}_vs_{b}"
            summary["pairwise_significance"][pair_key] = {}
            for metric_key in ["rougeL", "bertscore_f1"]:
                sa = data_a[metric_key]
                sb = data_b[metric_key]
                if len(sa) != len(sb) or not sa:
                    continue
                summary["pairwise_significance"][pair_key][metric_key] = {
                    "bootstrap": paired_bootstrap(sa, sb),
                    "wilcoxon": wilcoxon_test(sa, sb),
                }
            strat_a = [
                1.0 if p == g else 0.0
                for p, g in zip(data_a["predicted_strategy"], data_a["gold_strategy"])
            ]
            strat_b = [
                1.0 if p == g else 0.0
                for p, g in zip(data_b["predicted_strategy"], data_b["gold_strategy"])
            ]
            summary["pairwise_significance"][pair_key]["strategy_correct"] = {
                "bootstrap_seed42_only": paired_bootstrap(strat_a, strat_b),
            }

    # Strategy-accuracy significance, pooled across ALL seeds (not just 42).
    # Instances are paired 1:1 within each seed (same eval instance order
    # for every system at that seed -- verified against gold_strategy
    # sequences), so McNemar's exact test on the discordant pairs is the
    # textbook-correct test here, and is run per-seed AND pooled (summing
    # discordant-pair counts across seeds). The pooled paired_bootstrap
    # (resampling over all 1500 concatenated per-instance diffs) is also
    # reported as a cross-check; both are more informative than the old
    # seed42-only bootstrap above, which is kept only for continuity.
    common_seeds = [s for s in SEEDS if all(s in per_system[sys_] for sys_ in SYSTEMS)]
    for a, b in [("causal_mcts", "flow_ablation"), ("causal_mcts", "random_floor")]:
        pair_key = f"{a}_vs_{b}"
        if pair_key not in summary["pairwise_significance"]:
            summary["pairwise_significance"][pair_key] = {}
        per_seed_mcnemar = {}
        pooled_correct_a: list[float] = []
        pooled_correct_b: list[float] = []
        pooled_b_count = 0
        pooled_c_count = 0
        for s in common_seeds:
            data_a = per_system[a][s]["per_example"]
            data_b = per_system[b][s]["per_example"]
            correct_a = [
                1.0 if p == g else 0.0
                for p, g in zip(data_a["predicted_strategy"], data_a["gold_strategy"])
            ]
            correct_b = [
                1.0 if p == g else 0.0
                for p, g in zip(data_b["predicted_strategy"], data_b["gold_strategy"])
            ]
            if len(correct_a) != len(correct_b) or not correct_a:
                continue
            mc = mcnemar_test(correct_a, correct_b)
            per_seed_mcnemar[str(s)] = mc
            pooled_b_count += mc["b"]
            pooled_c_count += mc["c"]
            pooled_correct_a.extend(correct_a)
            pooled_correct_b.extend(correct_b)

        pooled_n = pooled_b_count + pooled_c_count
        pooled_mcnemar = (
            {
                "b": pooled_b_count,
                "c": pooled_c_count,
                "n_discordant": pooled_n,
                "p_value": float(binomtest(pooled_b_count, pooled_n, 0.5).pvalue),
                "note": "naive pooling across seeds -- treats all instances as i.i.d., "
                        "ignoring between-seed (between-model) clustering",
            }
            if pooled_n > 0
            else {"b": pooled_b_count, "c": pooled_c_count, "n_discordant": 0, "p_value": 1.0}
        )

        summary["pairwise_significance"][pair_key]["strategy_correct_all_seeds"] = {
            "seeds_used": common_seeds,
            "mcnemar_per_seed": per_seed_mcnemar,
            "mcnemar_pooled": pooled_mcnemar,
            "bootstrap_pooled": (
                paired_bootstrap(pooled_correct_a, pooled_correct_b)
                if pooled_correct_a
                else None
            ),
        }

        # Seed-level paired test: treats each seed's accuracy as one
        # observation (n=len(common_seeds)) -- the statistically honest unit
        # if the claim is "the method beats the baseline in general" rather
        # than "on this specific pool of instances". Very low power at
        # n=3 seeds regardless of true effect size; reported for transparency.
        if len(common_seeds) >= 2:
            seed_acc_a = [per_system[a][s]["strategy_accuracy"] for s in common_seeds]
            seed_acc_b = [per_system[b][s]["strategy_accuracy"] for s in common_seeds]
            seed_diffs = [x - y for x, y in zip(seed_acc_a, seed_acc_b)]
            mean_diff = statistics.mean(seed_diffs)
            std_diff = statistics.stdev(seed_diffs) if len(seed_diffs) > 1 else 0.0
            t_stat, t_p = ttest_rel(seed_acc_a, seed_acc_b)
            summary["pairwise_significance"][pair_key]["strategy_correct_seed_level"] = {
                "seeds_used": common_seeds,
                "per_seed_accuracy_a": seed_acc_a,
                "per_seed_accuracy_b": seed_acc_b,
                "per_seed_diff": seed_diffs,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "paired_ttest": {"t_statistic": float(t_stat), "p_value": float(t_p)},
                "note": f"n={len(common_seeds)} seeds -- very low power regardless of true effect size",
            }

    out_path = os.path.join(metrics_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[significance] wrote {out_path}")
    for sys_name, entry in summary["per_system_mean_std"].items():
        print(f"[significance] {sys_name}: " + ", ".join(
            f"{k}={v['mean']:.4f}+/-{v['std']:.4f}" for k, v in entry.items()
        ))
    for pair_key, entry in summary["pairwise_significance"].items():
        all_seeds = entry.get("strategy_correct_all_seeds")
        if all_seeds:
            mc = all_seeds["mcnemar_pooled"]
            print(
                f"[significance] {pair_key} strategy_correct McNemar "
                f"(pooled, seeds={all_seeds['seeds_used']}): "
                f"b={mc['b']} c={mc['c']} p={mc['p_value']:.4f}"
            )
        seed_level = entry.get("strategy_correct_seed_level")
        if seed_level:
            print(
                f"[significance] {pair_key} strategy_correct seed-level paired t-test: "
                f"mean_diff={seed_level['mean_diff']:.4f} "
                f"p={seed_level['paired_ttest']['p_value']:.4f} (n={len(seed_level['seeds_used'])} seeds)"
            )


if __name__ == "__main__":
    main()
