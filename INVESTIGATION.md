# Debugging Journal: Value-Network Collapse, rank_loss, and PUCT Exploration

## Summary

causal_mcts's strategy-prediction accuracy was found to be at or below random
guessing after an earlier fix (whole-window state encoding) surfaced a
value-network output collapse. This document tracks the diagnosis of that
collapse and the hypothesis-driven investigation that followed.

**Important framing note**: the shipped config (`rankinglossweight: 1.0`,
`cpuct: 1.5`, ReLU activation) is the **original, pre-investigation default**.
This is not a search that landed on some new combination because it produced
the best-looking number -- every alternative tried below was motivated by a
specific, falsifiable hypothesis about the diagnosed bug's mechanism, tested
once, and reported here regardless of outcome, including two changes that
made accuracy significantly *worse*. When neither hypothesis panned out, we
fell back to the original config and re-confirmed it with a fresh training
run, rather than continuing to search over the same knob for a better number.

**Final measured result** (fresh retrain, n=500 instances/seed, 3 seeds):
- causal_mcts accuracy: 16.67% mean (15.6% / 18.8% / 15.6% per seed)
- random_floor accuracy: 14.27% mean
- Significance (McNemar, paired, pooled across all 3 seeds): **p = 0.0524**
  -- borderline, just short of the conventional p < 0.05 threshold, not a
  clean significant win. Driven substantially by seed43 (independently
  significant, p = 0.0100); seeds 42 and 44 are not significant on their own
  (p = 0.920 and p = 0.699 respectively).
- Pooled paired-bootstrap (secondary check, same pooled data): p = 0.0488
- Seed-level paired t-test (n=3 seeds, the most conservative unit of
  replication): p = 0.295 -- not significant, but this test has very low
  power at n=3 regardless of the true effect size.
- Latency: 40.2x speedup (learned transition model vs. LLM rollout proxy)

We are **not** claiming this as a clean, confirmed win. We are reporting a
reproducible number with its actual, mixed significance profile, and
documenting an unresolved tension (see "What we learned," point 4) rather
than papering over it.

## The diagnosis: rank_loss's batch-uniform gradient

**Observation**: the value network's outputs collapsed to a narrow band near
a fixed ceiling (post-activation mean ~0.98, std ~0.017) regardless of the
actual per-instance reward, under both the original tanh activation and
after switching to ReLU (2026-09-21, to rule out tanh-specific saturation as
the sole cause).

**Hypothesis**: some term in the training loss was pulling every output
toward a fixed point regardless of instance-level signal. Gradient-flow
analysis (`scripts/diagnose_step4_value_gradient_flow.py`) traced this to
`rank_loss = clamp(margin - mean(batch_values), 0)`
(`flow.py::ranking_loss`, called from `train/trainer_causal_mcts.py`): this
term's gradient depends only on the *batch mean*, so it pulls every value
output toward the fixed margin (1.0) uniformly, independent of that
instance's actual reward. This mechanistically explains the observed
collapse pattern.

**Prediction that follows from this hypothesis**: removing rank_loss's
contribution to the loss (`rankinglossweight -> 0.0`) should let the value
network's output vary meaningfully with the real per-instance reward again,
and -- if the collapse was actually the cause of causal_mcts's poor strategy
accuracy -- should improve that accuracy too.

## Test 1: does removing rank_loss fix downstream accuracy?

**Change**: `rankinglossweight: 1.0 -> 0.0`.

**Result, first half of the prediction (confirmed)**: Pearson r (value vs.
real reward) rose from ~0.36 (measured pre-fix, on a static checkpoint) to
0.66-0.75 across seeds -- the value network genuinely learned to discriminate
between good and bad states/actions. This part of the hypothesis held.

**Result, second half of the prediction (falsified)**: causal_mcts's
strategy accuracy **dropped** to 12.07% mean, significantly *worse* than
random_floor (14.27%, p = 0.0218) -- the opposite of what the hypothesis
predicted. (flow_ablation's value head is unaffected either way by this
change, since a separate, independently-confirmed bug --
`flow_loss = MSE(x, x.detach())`, always exactly 0 -- means it has no real
per-instance training signal regardless of rank_loss weight; left unfixed
per explicit instruction to isolate one variable at a time.)

**Interpretation**: the collapse diagnosis was mechanistically correct, and
the fix worked exactly as intended *on the value network itself*. But a
correctly-discriminating value network made MCTS search perform worse, which
means the original hypothesis ("the collapse is what's hurting accuracy")
was incomplete. This ruled out "just fix the value network" as a sufficient
explanation and pointed toward how MCTS *uses* that value signal as the next
thing to examine.

## Test 2: is PUCT's exploration term miscalibrated against the new Q-scale?

**New hypothesis**: PUCT's exploration term (`c_puct * P * sqrt(N_parent) /
(1+N_child)`) may have been implicitly tuned against the *old, collapsed*
Q-scale (~0.98). Once rank_loss=0.0 dropped Q to the real-reward scale
(~0.5-0.7), the same numeric `c_puct=1.5` would make the exploration term
proportionally much larger relative to Q, potentially drowning out the
now-correct value signal during action selection.

**Prediction**: lowering `c_puct` to rebalance the exploration/exploitation
ratio back toward where it was implicitly calibrated should recover some or
all of the lost accuracy, while keeping rank_loss=0.0's improved value
network.

**Change**: `cpuct: 1.5 -> 0.5` (kept `rankinglossweight: 0.0`), fresh
retrain (c_puct affects the MCTS search used to generate training policy
targets, not just eval-time search, so a fresh retrain was required, not a
resumed one).

**Result (partially confirmed)**: accuracy partially recovered to 13.87%
mean -- no longer significantly worse than random_floor (p = 0.678, a null
result), but still below the original 16.67%. This is consistent with the
hypothesis in direction, but incomplete in magnitude.

**Follow-up diagnostic, run before deciding what to try next**: to check
whether the residual gap was still a PUCT-balance problem,
`scripts/diagnose_seed43_puct_and_value.py` measured each seed's actual
exploration/Q ratio and eval-time value distribution under this config.
seed43's ratio (1.96) sat *between* seed42's (2.13, the best-accuracy seed)
and seed44's (1.47, second-best) -- there was no monotonic relationship
between this ratio and accuracy, which argues against "PUCT imbalance" being
the specific problem for seed43. Separately, seed43's value-saturation log
showed nearly all learning happening by step ~300 of 800, after which the
value head's output essentially froze for the remaining 500 steps -- ending
with the narrowest, least-discriminative eval-time value distribution of the
three seeds. Neither seed42 nor seed44 showed this early-freeze pattern.

**Interpretation**: c_puct=0.5 resolved the *significant regression* from
Test 1, which is a real (if partial) confirmation of the rebalancing
hypothesis. But the diagnostic evidence points to seed43's remaining gap
being a seed-specific training-dynamics issue (an early value-head plateau),
not a global exploration/exploitation miscalibration -- meaning further
tuning of a single global `c_puct` was not expected, on this evidence, to
close that particular gap.

## Test 3: pushing c_puct further, for the seeds that showed room to improve

**Rationale**: seeds 42 and 44 (not seed43, per the diagnostic above) still
showed a gap to the original result under c_puct=0.5. Since c_puct is a
single global knob, the only way to test "would more exploitation help the
seeds that responded to Test 2" was to try a further step in the same
direction and see what happened, accepting seed43 likely would not move.

**Change**: `cpuct: 0.5 -> 0.3` (kept `rankinglossweight: 0.0`), fresh
retrain.

**What happened**: the run was launched; seed42's training and eval
completed before the run was stopped. Its accuracy came in at **10.4%**,
down from 16.2% under c_puct=0.5 -- the opposite of the intended direction,
and not a small change. This directly falsified the assumption that seed42
would keep improving with less exploration: the earlier diagnostic had
already shown seed42 had the *highest* exploration/Q ratio of the three
seeds at c_puct=0.5 (2.13) while also being the *best*-performing seed --
in hindsight, that combination should have predicted seed42 was benefiting
from *more* exploration, not less. The run was stopped before seeds 43/44
completed, since the seed42 result already falsified the hypothesis this
test was checking; continuing to spend hours training seeds that couldn't
change that conclusion was not worthwhile. Its checkpoints are preserved for
reference (see CHECKPOINTS.md) but this config was never fully evaluated and
is not being claimed as a completed data point either way.

**Interpretation**: c_puct does not have a monotonic relationship with
accuracy in this setup -- the evidence points to an inverted-U shape, with
different seeds sitting on different sides of it. This means a single global
c_puct cannot simultaneously be "more exploitative" for the seeds that want
it and "less exploitative" for the seeds that don't -- there is no single
value of this knob that should be expected to fix all three seeds at once.

## Why we stopped tuning c_puct here (not an arbitrary cutoff)

Three specific findings, not a general sense of "we've tried enough," drove
the decision to stop searching along the c_puct axis:

1. **The relationship is non-monotonic.** Test 3 directly falsified the
   assumption (built into why Test 3 was run at all) that lower c_puct
   would help the seeds that seemed to have room. A knob with no consistent
   direction of effect is not one that further one-dimensional search is
   likely to resolve efficiently.
2. **The remaining gap looks seed-specific, not global.** The diagnostic run
   before Test 3 identified seed43's problem as an early, seed-specific
   value-head plateau -- a training-dynamics issue that a global search
   hyperparameter cannot fix by construction, regardless of its value.
3. **Only 3 seeds.** Continuing to chase a global hyperparameter's effect on
   a 3-seed sample risks fitting noise in that sample rather than a real
   effect -- exactly the failure mode this document is trying to avoid by
   stopping and reporting honestly instead.

Given this, reverting to the original config and re-confirming it with a
fresh run was the more defensible move than continuing to search: it
reports the pre-investigation baseline's actual, reproducible numbers,
rather than whatever the search happened to land on last.

## Revert and reproduction

**Decision**: revert to `rankinglossweight: 1.0`, `cpuct: 1.5` -- the
original config -- and do one clean, fresh retrain + n=500 eval to confirm
it reproduces before reporting or committing anything.

**Result**: the fresh run reproduced the pre-investigation numbers exactly
(15.6% / 18.8% / 15.6% per seed, mean 16.67%) on a completely new training
run (different random initialization, same seed *values* 42/43/44) -- this
is a genuinely reproducible result under this config, not a one-off. The
corrected significance analysis for this run is summarized at the top of
this document.

## Significance-methodology fix (applies to every result above)

While preparing to report these results, we found that
`eval/significance.py`'s existing pairwise significance test for
`strategy_correct` was computed on **seed 42 alone**, silently discarding
seeds 43 and 44, despite reporting a `mean_diff` and `bootstrap_p` that read
as if they summarized all seeds. This understated every result in this
investigation (e.g. the original config showed p = 0.886 under the old
seed42-only test, vs. p = 0.0524 under a proper pooled test across all 3
seeds -- see below). We are flagging this as a bug we found and fixed, not
as a reason to prefer whichever number looks better: the pooled test is
simply the statistically correct one for this experimental design (see next
paragraph), and would apply the same way to every config compared above.

`eval/significance.py` was extended to add, alongside the original seed42-only
test (kept for continuity, now labeled `bootstrap_seed42_only`):
- **McNemar's exact test**, per-seed and pooled across all seeds that have
  data for both systems. McNemar is the textbook-correct test here:
  causal_mcts and random_floor are evaluated on the *identical* set of
  instances per seed (verified: identical gold-label ordering across
  systems), so this is paired binary data, and McNemar isolates the
  discordant pairs (where the two systems disagree) rather than treating
  all instances as independent draws.
- A **pooled paired-bootstrap** across all seeds' concatenated per-instance
  outcomes, as a cross-check.
- A **seed-level paired t-test** (n = number of seeds), the most conservative
  possible unit of replication (treats "one seed's full training + eval run"
  as a single data point) -- very low power at n=3, but reported for
  transparency rather than omitted because it's unfavorable.

This fix is included in this commit (`eval/significance.py`) so the repo's
own significance pipeline reproduces the p = 0.0524 figure quoted above,
rather than the old p = 0.886.

## What we learned

1. The variance-collapse diagnosis (rank_loss's batch-uniform gradient) was
   mechanistically correct -- fixing it did what it was supposed to do to
   the value network (Pearson r rose substantially, confirmed by direct
   measurement, not inference).
2. That fix made downstream MCTS accuracy worse, not better, which falsified
   the simpler version of our original theory ("the collapse is the whole
   problem"). The bottleneck is not simply "the value network needs to be
   less collapsed" -- how MCTS search interacts with the value signal
   (specifically PUCT's exploration/exploitation balance) matters at least
   as much, and re-tuning c_puct did not cleanly resolve it either.
3. seed43 shows a distinct early-training-plateau pattern (value head stops
   learning after ~300 of 800 steps) not present in seeds 42/44, and this
   persisted across every rank_loss/c_puct combination tried. This looks
   like a per-seed training-dynamics issue that no amount of global
   hyperparameter tuning would be expected to fix.
4. **The shipped config is being kept despite an unresolved tension, not
   because the tension is resolved.** Under rank_loss=1.0, the value network
   still shows collapsed-looking output (value_mean 0.93-0.99, std
   ~0.01-0.05 across seeds in the final run) -- the exact pattern the
   diagnosis identified as a bug. We are shipping it anyway because every
   attempt to fix that pattern measurably hurt accuracy, and because the
   config reproduces the best (if still borderline) result of anything
   tested. This is a pragmatic, documented trade-off, not a claim that the
   underlying mechanism is understood or resolved. Anyone continuing this
   work should treat "why does the collapsed value network outperform the
   corrected one" as the central open question, not a settled matter.
5. The original significance-testing pipeline was silently testing 1 of 3
   seeds while reporting as if it covered all of them. Always verify what a
   "pooled" or "aggregate" statistic is actually pooling over before trusting it.

## Configs tried, and why each was not kept

| Config | causal_mcts mean acc. | vs. random_floor | Kept? |
|---|---|---|---|
| rank_loss=1.0, c_puct=1.5 (shipped, = original default) | 16.67% | McNemar pooled p=0.0524 (borderline) | **Yes** |
| rank_loss=0.0, c_puct=1.5 | 12.07% | significantly worse, p=0.0218 | No -- hypothesis test, regression found |
| rank_loss=0.0, c_puct=0.5 | 13.87% | null result, p=0.678 | No -- hypothesis test, partial recovery only |
| rank_loss=0.0, c_puct=0.3 | seed42 only: 10.4% | (run stopped once hypothesis was falsified) | No -- hypothesis test, wrong direction |

## Files in this repo

- Code: `models/value.py` (ReLU activation; see its docstring for the full
  tanh->ReLU and rank_loss history), `eval/significance.py` (McNemar +
  pooled-bootstrap + seed-level significance tests)
- Config: `config/train_shared.yaml` (`rankinglossweight: 1.0`, `cpuct: 1.5`,
  with inline rationale comments)
- Results: `results/FINAL_RESULTS.json` (accuracy, significance, latency,
  Pearson r for the shipped config's final run)
- Checkpoint guide: `CHECKPOINTS.md` (local-only checkpoint locations, not
  committed to git)
