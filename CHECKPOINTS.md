# Checkpoint Guide

Checkpoint files are large and are **not** committed to git (`checkpoints/`
is in `.gitignore`). This file is a map of what exists locally, for
reproducibility and for anyone picking this investigation back up.

## Shipped checkpoint (current `checkpoints/`)

- Path: `checkpoints/causal_mcts/seed{42,43,44}/`, `checkpoints/flow_ablation/seed{42,43,44}/`
- Config: ReLU value activation, `rankinglossweight: 1.0`, `cpuct: 1.5`
  (config/train_shared.yaml, as committed)
- causal_mcts accuracy: 16.67% mean (15.6% / 18.8% / 15.6% per seed), n=500/seed
- McNemar pooled p = 0.0524 vs. random_floor (borderline, see INVESTIGATION.md)
- Latency: 40.2x speedup
- Trained: fresh init, 2026-09-23/24 (the "final retrain" run referenced in
  INVESTIGATION.md's revert section)
- Reproducible: yes -- `utils.seed.set_global_seed` confirmed deterministic
  (re-initializing the same seed number twice gives bit-identical weights),
  and this exact config reproduced the original pre-investigation accuracy
  numbers to the decimal point on a completely fresh training run.

## To reproduce locally

```bash
# Full fresh retrain + eval (several hours on CPU-only hardware):
bash scripts/run_final_retrain.sh

# Eval only, if checkpoints/ already holds trained weights:
python -m eval.run_eval --system causal_mcts --seed 42 --n 500
python -m eval.significance   # aggregates results/metrics/*.json -> summary.json
```

## Backup checkpoints (local only, not in git)

These are snapshots of `checkpoints/` taken before each config change during
the investigation, kept for ablation reproducibility:

| Directory | Config at time of backup |
|---|---|
| `checkpoints_backup_pre_wholewindow/` | Before the whole-window state-encoding fix |
| `checkpoints_backup_pre_aggregation_fix/` | Before an earlier multi-turn aggregation fix |
| `checkpoints_backup_pre_relu/` | tanh activation, before the ReLU switch |
| `checkpoints_backup_relu_baseline/` | ReLU, n=150 eval era (before eval-set-size fix) |
| `checkpoints_backup_prerankloss_fix/` | ReLU, rank_loss=1.0, before Phase 1 (rank_loss=0.0 attempt) |
| `checkpoints_backup_cpuct_baseline/` | rank_loss=0.0, c_puct=1.5, before Phase 2 (c_puct=0.5 attempt) |
| `checkpoints_backup_cpuct_0p5/` | rank_loss=0.0, c_puct=0.5 (Phase 2 result), before Phase 3 |
| `checkpoints_backup_cpuct_0p3_abandoned/` | rank_loss=0.0, c_puct=0.3 -- Phase 3, abandoned mid-run (seed42 only completed: 10.4% accuracy) |

None of these are needed to reproduce the shipped result -- they exist so
each investigation phase's exact starting point can be re-examined if needed.
