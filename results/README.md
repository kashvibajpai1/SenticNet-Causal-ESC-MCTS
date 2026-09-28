# Results: real experiment run

This directory holds the output of an actual, real, bounded experiment run
(`scripts/run_full_experiment.sh` and, for this update, a split
preprocess+train / eval sequence — see **Reproducing this run** below),
produced in response to peer review that found the paper's empirical
validation not credible: the reported config was a documented "smoke test"
(`maxsteps=10`), no evaluation code existed for any cited metric, no
variance/significance testing was done, no qualitative analysis was
included, and the "AFlow baseline" was unclear.

Investigating the codebase to fix this surfaced problems worse than the
reviews assumed — see **What was actually broken** below. This document
reports what changed, the real numbers this run produced (including
unflattering ones), and every known limitation, plainly.

## 2026-09-13 update: the missing policy loss, a deeper phase bug, and a larger eval

A first pass at this pilot (300 steps, 15 simulations, 100 conversations/
split, 40 eval instances — numbers preserved below for comparison) trained
`causal_mcts` with no term in its loss tying the policy network to what MCTS
search actually preferred, and sampled eval instances in a way that put
100% of them in ESC phase 0. This update fixes both, and one more bug found
while investigating the result:

1. **Wired in the missing policy loss** (`train/trainer_causal_mcts.py`):
   `train_step` now calls `mcts.search_with_policy()` instead of
   `mcts.search()`, and adds a cross-entropy term between the policy
   network's (candidate-restricted) output and MCTS's visit-count
   distribution, plus a small entropy bonus against collapse. New config
   keys: `policylossweight` (1.0), `entropybonus` (0.01).
2. **Fixed eval's phase-0 sampling bias** (`eval/data.py`):
   `build_eval_instances` no longer takes only the first eligible supporter
   turn per conversation — it collects every eligible turn, so later-phase
   gold labels (Comforting, Action Planning) are represented too, not just
   Exploration.
3. **Found and fixed a deeper bug this exposed** (`esc/state.py`): even with
   (1) and (2) in place, `causal_mcts` re-evaluated at n=150 still predicted
   a single strategy 150/150 times. Root cause: `from_dialogue()` set
   `phase_embedding` to an exact 3-way tie (`[1/3, 1/3, 1/3]`) for *every*
   real state. `current_phase = phase_embedding.argmax()` resolves an exact
   tie to index 0 deterministically, so **every real state — all 910
   training states, and every eval instance, at any conversation depth from
   2 to 59+ turns — always had `current_phase == 0`.** Since candidate
   actions are filtered by `PHASE_STRATEGY_MAP[current_phase]`, 4 of the 8
   real strategies (Affirmation, Providing Suggestions, Information, Others)
   were structurally unreachable for every input, not just under the
   eval-sampling bias (2) fixed. This is very likely the true root cause of
   the mode collapse, more fundamental than (1): the policy loss correctly
   learned to match MCTS's own target, but MCTS itself never had a chance to
   prefer anything outside the same frozen 4-strategy set. Fixed by
   estimating phase from real conversation depth (turn count) instead of a
   hardcoded tie — see the fix's commit message and inline comment in
   `esc/state.py` for the exact thresholds and how they were chosen (data
   percentiles, not round numbers).

**This was outside the task spec's stated scope** (the spec's B1 only asked
to fix eval's turn-sampling, not the model's own phase computation) and was
confirmed with the user before implementing, since it required redoing
preprocessing, training, and eval from scratch — the bug was baked into the
serialized `artifacts/states/*.pt` bundles, so fixing the code alone
couldn't retroactively fix already-built states.

### What this bought, honestly

- The global mode collapse is gone: `causal_mcts` now predicts **3
  different strategies** (not 1) across the 150-instance eval set, and
  which one varies by phase in a way that lines up exactly with
  `PHASE_STRATEGY_MAP` (e.g. `Information` — a phase-2-only strategy — only
  ever appears for deeper-conversation instances).
- **Raw strategy accuracy did not improve**, and `causal_mcts` is now
  **significantly *worse* than `random_floor`** on strategy accuracy
  (bootstrap p=0.021 — see Significance below). The structural bug capping
  reachable strategies is fixed, but 800 training steps on real-encoded
  states that are still highly similar to each other (pairwise cosine
  similarity 0.97-0.998 on the actual state tensors the networks see — see
  the conversation log) isn't enough signal for the network to discriminate
  *within* a phase yet. It converges to "the single best default per
  phase" rather than a genuinely context-sensitive choice. That's a
  separate, deeper problem than either bug fixed here, and is not resolved
  by this pass.
- Two things this pass explicitly did **not** do, both by explicit user
  decision, not oversight: **B3** (reproducing or dropping the published
  CauESC/GDP-Zero/DG-MCTS/AFlow baselines) was skipped — no such comparison
  table or paper draft with one exists anywhere in this repo, only a
  3-page methodology-only proposal PDF with no results section, so there
  was nothing to fix. **Part C** (pairwise LLM-as-judge) has a built,
  tested harness (`eval/llm_judge.py`, `tests/test_llm_judge.py`) but was
  never executed — no `ANTHROPIC_API_KEY` was available in this
  environment.

## 2026-09-24 update: value-network collapse, rank_loss/c_puct investigation, eval scaled to n=500, significance-pipeline fix

Three changes since the 2026-09-13 update, in order of how they were found:

1. **A value-network output collapse was diagnosed and investigated.**
   `models/value.py`'s output was found to converge to a narrow band near a
   fixed ceiling (mean ~0.98, std ~0.017) regardless of the actual
   per-instance reward, traced to `rank_loss`'s batch-uniform gradient
   (`flow.py::ranking_loss`) pulling every output toward its margin
   independent of instance-level signal. tanh was replaced with ReLU
   (2026-09-21, to rule out tanh-specific saturation as the sole cause; see
   `models/value.py`'s docstring). A hypothesis-driven investigation then
   tested whether removing rank_loss's contribution (`rankinglossweight
   1.0 -> 0.0`) and rebalancing PUCT's exploration constant (`cpuct 1.5 ->
   0.5 -> 0.3`) would fix both the value network *and* downstream strategy
   accuracy. It fixed the value network (Pearson r vs. real reward rose from
   ~0.36 to 0.66-0.75) but made accuracy significantly *worse*
   (-4.8pp vs. random_floor, p=0.0218); rebalancing c_puct partially but
   incompletely recovered it, and pushing further in the same direction
   regressed one seed further rather than continuing to improve. **The
   original config (`rankinglossweight: 1.0`, `cpuct: 1.5`) was reverted to
   and reproduced its original numbers on a fresh training run** — this is
   the config reflected in the tables below. The value network under this
   config still shows the collapsed-looking output pattern; it is being kept
   because every attempted fix measurably hurt accuracy, not because the
   collapse is resolved. Full hypothesis-by-hypothesis writeup, including
   every config tried and why each was rejected:
   [`../INVESTIGATION.md`](../INVESTIGATION.md).
2. **Eval set scaled from 150 to 500 instances/seed.** At n=150, per earlier
   analysis, instance-level noise (~±3.14pp binomial SE) was close to the
   size of observed effects, limiting statistical power. n=500 (~±1.88pp)
   was judged the more useful use of compute than further architectural
   changes at the time. All tables below reflect n=500.
3. **Found and fixed a significance-pipeline bug**: `eval/significance.py`'s
   `strategy_correct` significance test was computed on **seed 42 alone**
   while its output (`mean_diff`, `bootstrap_p`) read as if it summarized
   all 3 seeds — silently understating (or overstating) every strategy-
   accuracy significance result reported before this fix. Added McNemar's
   exact test (the correct test for this design: causal_mcts and
   random_floor/flow_ablation are scored on identical per-seed instance
   sets, i.e. paired binary data), computed per-seed and pooled across all
   3 seeds, plus a pooled paired-bootstrap and a seed-level paired t-test.
   The old seed42-only test is kept in `summary.json` for continuity
   (`bootstrap_seed42_only`) but is no longer the headline number — see
   **Significance testing** below. This fix applies only to
   `strategy_correct`; the ROUGE-L/BERTScore pairwise tests below are still
   computed on seed42 only (not yet extended to pooled McNemar/bootstrap,
   since those are continuous, not binary, metrics).

## What was actually broken (found in the original pilot pass)

- **The "Qwen-9B" backbone was a complete stub.** `models/backbone_qwen.py`'s
  `encode_dialogue()` returned all-zero tensors and `generate_response()`
  returned the literal string `"[Qwen-9B stub reply]"`. No real dialogue
  encoding or generation ever happened, at any `maxsteps`.
- **No evaluation code existed anywhere** for BLEU, ROUGE, BERTScore,
  Distinct-2, or strategy accuracy — a repo-wide search found zero matches.
- **The ESConv data loader had never been run against the live dataset.**
  `esconv_row_to_record()` assumed a flat row schema; the real
  `thu-coai/esconv` dataset wraps each row as a JSON string under a
  `"text"` key. Every row silently produced 0 turns. Fixed in
  `data/dataset_sources.py`.
- **The flat-YAML config parser didn't strip quotes**, so
  `backbone_model_name: "Qwen/..."` was read as the literal string
  `'"Qwen/..."'` (quotes included) — this went unnoticed before because
  nothing had ever actually consumed that config key. Fixed in
  `train/utils.py`.
- **`--max-esconv` capped records globally across HF splits**, not per
  split, so any cap value exhausted entirely within `train` and left
  `validation`/`test` empty — which would have made the entire eval harness
  silently operate on zero instances. Fixed in `scripts/preprocess_datasets.py`
  (now capped per split).
- **Neither trainer ever saved a checkpoint** (`torch.save`) — added to
  both `scripts/run_causal_mcts.py` and `scripts/run_flow_ablation.py`.
- **The "AFlow baseline" was an in-house GFlowNet-style flow-matching
  ablation**, not a reproduction of the published AFlow system (Zou et al.).
  Renamed to **FlowMCTS-ablation** throughout the repo, with an explicit
  disclaimer everywhere it's introduced.

## What replaced the stub backbone

**`Qwen/Qwen2.5-0.5B-Instruct`**, run locally, CPU-only (this project has no
GPU access) — downsized from the paper's stated "Qwen-9B," which was never
actually feasible here regardless of the stub. This is a real, honest
substitution, not a claim of matching "Qwen-9B" quality.

- Backbone weights are **frozen** throughout — only the policy/value/
  transition MLPs downstream are trained, exactly as the original
  architecture already assumed (the backbone was always meant to be a
  fixed encoder, per `models/README.md`).
- Qwen's hidden size (896) doesn't match the fixed `D_H=768`/`D_C=384`/
  `D_E=128` dimensions hardcoded across the policy/value/transition
  networks. Real Qwen hidden states are projected down with **frozen,
  seeded random linear projections** (Johnson-Lindenstrauss-style) rather
  than retrofitting those dimensions everywhere. This is a real
  dimensionality reduction of real semantic features, not a hidden hack —
  see `models/backbone_qwen.py`'s module docstring. **This is also the
  likely reason encoded states remain highly similar to each other even
  after the phase fix** (see "What this bought, honestly" above) — a JL
  projection approximately preserves distances, but doesn't guarantee the
  *specific* distinctions the policy network needs are preserved well
  enough to act on with limited training.
- ESConv has no ground-truth "cause span" labels. `encode_dialogue()` uses
  a heuristic: the seeker's turns are treated as candidate distress-cause
  spans, most recent first. This is real and text-grounded, not zero-fill,
  but it is a heuristic, not a principled extraction model.
- **`esc/env.py`'s per-step history embedding remains a zero-padded
  placeholder** (`step()` appends a zero vector for the new turn during
  MCTS simulation, rather than a live Qwen call). This was **not** fixed,
  deliberately: doing so would mean calling Qwen inside every MCTS
  simulation step, reintroducing exactly the LLM-rollout latency problem
  this project's architecture exists to avoid (see the latency results
  below). Root states passed into search are real (built from real
  encoded context); only the simulated-forward states inside the tree
  search itself use this placeholder. This is a known, disclosed
  limitation, not a hidden one.

## Config actually used

Not the old `maxsteps=10` smoke test, and not an unverified paper-scale
claim. Full resolved config + git commit + library versions:
[`configs/resolved_config.json`](configs/resolved_config.json).

| Key | Latest run (2026-09-24) | Previous update (2026-09-13) | Original pilot | Task spec's literal ask |
|---|---|---|---|---|
| Value activation | ReLU | tanh | tanh | (unspecified) |
| `rankinglossweight` | 1.0 | 1.0 | 1.0 | (unspecified) |
| `cpuct` | 1.5 | 1.5 | 1.5 | (unspecified) |
| `maxsteps` | 800 | 800 | 300 | 3000–5000 |
| `num_simulations` | 30 | 30 | 15 | 30–50 |
| `batchsize` | 16 | 16 | 16 | (unchanged) |
| Dataset | full ESConv (1300 conversations: 910/195/195 train/valid/test) | full ESConv | 100 conversations/split | full dataset |
| Seeds | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 | 42–44 min, +45/46 if budget allows |
| Eval set | **500 instances/seed** | 150 instances/seed | 40 (all phase-0, pre-B1-fix) | "most/all" of the eligible pool |

`rankinglossweight` and `cpuct` are both the original, unchanged defaults —
both were investigated as alternatives during the 2026-09-24 update (values
0.0 and 0.5/0.3 respectively) but reverted; see
[`../INVESTIGATION.md`](../INVESTIGATION.md) for why.

`maxsteps` and eval `n` are below the spec's literal ask — an explicit,
user-approved tradeoff after calibration showed the literal values
extrapolate to 30+ hours on this machine (8GB RAM, CPU-only), against a
requested 3-4 hour budget. `num_simulations` and the full dataset **are**
at spec. See `config/train_shared.yaml`'s inline comment for the exact
numbers needed to run the literal full-scope version, and the eligible-pool
math (B1's fix surfaces ~13.8 eligible instances/test-conversation, so
"most/all" is ~2,700 instances, not the 500 used here).

## Systems compared

- **`causal_mcts`** — trained policy/value/transition networks + MCTS search (the paper's main system)
- **`flow_ablation`** — trained policy only, argmax action, no MCTS search (FlowMCTS-ablation; **not** AFlow)
- **`random_floor`** — uniformly random legal strategy (lower bound)

All three use the **same** real Qwen backbone to generate the final reply
text, conditioned on whichever strategy each system selects — isolating
strategy-selection quality as the source of any metric differences.

## Headline results (mean ± std over 3 seeds, n=500 eval instances/seed)

| Metric | causal_mcts | flow_ablation | random_floor |
|---|---|---|---|
| BLEU | 0.163 ± 0.015 | 0.185 ± 0.013 | 0.223 ± 0.044 |
| ROUGE-1 | 0.130 ± 0.000 | 0.130 ± 0.002 | 0.132 ± 0.001 |
| ROUGE-2 | 0.0089 ± 0.0001 | 0.0093 ± 0.0001 | 0.0099 ± 0.0003 |
| ROUGE-L | 0.0924 ± 0.0004 | 0.0934 ± 0.0015 | 0.0942 ± 0.0002 |
| BERTScore F1 | 0.7034 ± 0.0004 | 0.7028 ± 0.0012 | 0.7029 ± 0.0003 |
| Distinct-2 | 0.345 ± 0.006 | 0.330 ± 0.002 | 0.347 ± 0.001 |
| Strategy accuracy | **0.1667 ± 0.0151** | 0.1267 ± 0.0392 | 0.1427 ± 0.0093 |

Full per-seed values and per-example scores: [`metrics/summary.json`](metrics/summary.json), `metrics/{system}_seed{N}.json`. Config: ReLU value activation, `rankinglossweight=1.0`, `cpuct=1.5` (see [`../INVESTIGATION.md`](../INVESTIGATION.md) for why these were kept over the alternatives tried).

### Significance testing

**Strategy accuracy (binary correct/incorrect, paired per-instance across systems within a seed) — McNemar's exact test, the statistically correct test for this design, pooled across all 3 seeds:**

| Comparison | b (A right/B wrong) | c (A wrong/B right) | McNemar pooled p | Pooled bootstrap p (cross-check) | Seed-level paired t-test p (n=3, low power) |
|---|---|---|---|---|---|
| causal_mcts vs flow_ablation | 235 | 175 | **0.0035** | 0.0032 | 0.359 |
| causal_mcts vs random_floor | 181 | 145 | **0.0524** | 0.0488 | 0.295 |

Per-seed McNemar p-values, causal_mcts vs random_floor: seed42=0.920, seed43=0.010, seed44=0.699 — the pooled result is driven substantially by seed43; seeds 42 and 44 are not individually significant. causal_mcts vs flow_ablation is significant on 2 of 3 seeds individually (seed43 p<0.001, seed44 p=0.0095) and pooled.

**ROUGE-L / BERTScore F1 (continuous metrics, paired bootstrap + Wilcoxon, seed=42 instances only — not yet extended to all 3 seeds):**

| Comparison | ROUGE-L diff (bootstrap p / Wilcoxon p) | BERTScore diff (bootstrap p / Wilcoxon p) |
|---|---|---|
| causal_mcts vs flow_ablation | +0.0015 (0.398 / 0.172) | +0.0004 (0.657 / 0.730) |
| causal_mcts vs random_floor | -0.0018 (0.277 / 0.583) | +0.0013 (0.095 / 0.074) |

**Read plainly: causal_mcts beats random_floor at strategy selection by
2.4 percentage points (16.67% vs. 14.27%), and the pooled, statistically
appropriate test (McNemar) puts this at p=0.0524 — borderline, just short of
the conventional p<0.05 threshold, not a clean significant win.** It clears
significance against flow_ablation (p=0.0035). Text-generation metrics
(BLEU/ROUGE/BERTScore) show no significant differences between any pair of
systems on the seed42-only test available for them, consistent with all
three sharing the same backbone for reply generation regardless of which
strategy gets picked.

This is a materially different picture than the 2026-09-13 update's n=150
result, where `causal_mcts` was *significantly worse* than `random_floor`
(p=0.021) under a test that, in hindsight, was silently computed on seed42
alone rather than pooled across seeds (see the 2026-09-24 update above) —
comparing that number directly to today's pooled p=0.0524 is not an
apples-to-apples comparison of the same statistic. Both the config (see
[`../INVESTIGATION.md`](../INVESTIGATION.md)) and the significance
methodology changed since that update, and both changes are documented
rather than silently folded into a "things got better" narrative.

## Latency comparison (the paper's core claim, previously unvalidated)

Bounded, real measurement on 10 held-out decisions — learned-transition
MCTS (the trained `LinearTransitionModel`, no LLM calls during search) vs.
a bounded LLM-rollout proxy (real Qwen `generate_response()` calls for the
top-3 policy-prior candidates per decision). Full data:
[`latency/latency_baseline.json`](latency/latency_baseline.json).

| | Mean seconds/decision |
|---|---|
| Learned-transition MCTS (30 simulations) | 0.160s |
| LLM-rollout proxy (3 real Qwen calls) | 6.433s |
| **Speedup** | **40.2x** |

Still the one clearly positive, robust result across every run of this
experiment (71x at 15 simulations in the original pilot, 37x at 30
simulations in the 2026-09-13 update, 40.2x here — run-to-run variation at
the same simulation count reflects real machine-load noise, not a
methodology change; still nowhere near the cost of real LLM rollouts). This
is a faithful test of the paper's central latency argument: MCTS search
using a learned transition model is dramatically cheaper than one that
queries an LLM per candidate, independent of how well the policy itself has
learned to choose strategies.

## Qualitative examples

18 side-by-side examples (dialogue context, predicted vs. gold strategy,
generated vs. gold reply, per-example metrics), from this run's
`causal_mcts` seed=42:
[`qualitative/examples.md`](qualitative/examples.md), [`qualitative/examples.jsonl`](qualitative/examples.jsonl).

## HF cache cleanup (as requested — nothing left on disk afterward)

Downloaded weights (Qwen2.5-0.5B-Instruct, 1.9GB; distilbert-base-uncased
for BERTScore, 512MB) were deleted after the latency baseline (their last
use), leaving only pre-existing, unrelated dataset caches untouched. Full
before/after proof: [`cleanup_log.txt`](cleanup_log.txt).

## Reproducing this run

```bash
bash scripts/redo_preprocess_and_train.sh   # preprocessing + training only
# then, per-combo (see scripts/resume_from_eval.sh for the full loop):
python -m eval.run_eval --system causal_mcts --seed 42 --n 150
# ...repeat for all (system, seed) pairs, then:
python -m eval.significance
python -m eval.latency_baseline
python -m eval.qualitative
python -m scripts.cleanup_hf_cache
```

(`scripts/run_full_experiment.sh` runs the whole thing — preprocess, train,
eval loop, significance, latency, qualitative, cleanup, tests — in one
shot, and is what the split scripts above were derived from; they exist
because this run was executed in stages while debugging.)

**Real timing, measured, not estimated:** preprocessing ~32-46 min (full
1300-conversation corpus, varies with HF cache warmth); training ~30
min/seed for `causal_mcts` (worse-case one seed took 74 min for unclear
reasons — see the conversation log's disk-cache-warmup theory — the rest
matched calibration), seconds/seed for `flow_ablation`; eval ~14-15
min/combo at n=150 (860s measured directly, timed with `time`), i.e. roughly
2 hours for all 9 (system, seed) eval combinations. Two real operational
hazards hit during this run and are worth knowing about if you rerun this:
(1) **memory pressure** — this ran on an 8GB RAM machine with other apps
open; eval failed outright twice before enough RAM was free, and the
original per-combo timeout (900s) had to be raised to 2400s after direct
measurement showed real per-instance cost (encoding + generation) has much
higher tail variance (1.4s-6.2s) than a small calibration sample suggested;
(2) **the machine went to sleep** for a long stretch mid-training (lid
closed/idle sleep), pausing (not corrupting) a background job for ~21 wall
hours while it accumulated only ~20 minutes of real CPU time — run under
`caffeinate -i -d -s` for any unattended multi-hour stretch.

**2026-09-24 update**: the tables above come from `scripts/run_final_retrain.sh`,
which runs the same preprocess-skip / fresh-train / eval-at-n=500 sequence
as above with `--n 500` in place of `--n 150`, against the reverted
`rankinglossweight=1.0`, `cpuct=1.5` config. Each intermediate config tried
during the investigation (`rankinglossweight=0.0`, `cpuct=0.5`, `cpuct=0.3`)
has its own `scripts/run_*_retrain.sh` variant, listed in
[`../INVESTIGATION.md`](../INVESTIGATION.md) and [`../CHECKPOINTS.md`](../CHECKPOINTS.md).

## Full list of known limitations

1. Backbone downsized from "Qwen-9B" to Qwen2.5-0.5B-Instruct (no GPU available).
2. Frozen random projections bridge Qwen's hidden size to the fixed state dimensions (not a trained reduction) — likely why real encoded states remain highly similar to each other (cosine similarity 0.97-0.998) even after the phase fix, which is the most likely reason strategy accuracy hasn't improved.
3. Cause-span extraction is a heuristic (seeker turns), not a principled model — ESConv has no ground-truth cause-span labels.
4. `esc/env.py::step()`'s in-tree simulated history embedding remains a zero-padded placeholder (root states are real; simulated states inside search are not) — this is what keeps MCTS search cheap, and is the reason the latency speedup result is legitimate rather than trivial.
5. `esc/state.py::from_dialogue()`'s phase estimate (added 2026-09-13) is a turn-count heuristic, not a learned or annotated phase classifier — ESConv has no ground-truth phase labels. It fixes a real determinism bug (every state stuck at phase 0) but is itself a heuristic, disclosed like the others here.
6. Config is "reduced-but-real" (800 steps, 30 simulations, full dataset) for CPU wall-clock feasibility, not paper-scale (spec asks 3000-5000 steps) — an explicit, budget-driven, user-approved deviation.
7. Eval set is 500 instances/seed (scaled up from 150 on 2026-09-24, from 40 in the original pilot), but still far short of the ~2,700 instances now structurally eligible after the B1 fix, for the same wall-clock budget reason as (6).
8. FlowMCTS-ablation is an in-house flow-matching regularizer inspired by AFlow/GFlowNet ideas, not a reproduction of the published AFlow system.
9. Strategy accuracy is computed against ESConv's own annotated strategy labels via `data/preprocess.py`'s normalization — a proxy for "good strategy choice," not a guarantee that the gold label was itself optimal.
10. Table 1's published-baseline comparison (CauESC, GDP-Zero, DG-MCTS, AFlow) was not addressed — no such table or paper draft with one exists in this repo to fix or drop.
11. Part C (pairwise LLM-as-judge) has a built, tested harness (`eval/llm_judge.py`) but was never executed — no API credentials were available in this environment.
12. **The value network's output still shows collapsed-looking behavior** (mean 0.93-0.99, std ~0.01-0.05 across seeds) under the shipped `rankinglossweight=1.0` config, despite this being the diagnosed root cause of an earlier accuracy problem. It is being kept because every attempted fix (`rankinglossweight=0.0`, alone or combined with `cpuct` retuning) measurably reduced strategy accuracy. This is an open, unresolved tension, not a settled fix — see [`../INVESTIGATION.md`](../INVESTIGATION.md).
13. The pooled McNemar/bootstrap significance fix (2026-09-24) applies only to `strategy_correct`. The ROUGE-L and BERTScore F1 pairwise significance tests above are still computed on seed42 alone, not pooled across all 3 seeds — the same limitation this update fixed for strategy accuracy has not yet been addressed for those continuous metrics.
14. FlowMCTS-ablation's value head has no working per-instance training signal regardless of `rankinglossweight`: `flow_loss = MSE(F_edges, flow_from_policy)` is computed from two tensors both derived from the same detached value, so it is always exactly 0 — a separate, confirmed, and as of this update still-unfixed bug (left alone deliberately, to isolate the rank_loss investigation to one variable at a time). flow_ablation's strategy-selection results above come entirely from its trained policy network, not its value network.
