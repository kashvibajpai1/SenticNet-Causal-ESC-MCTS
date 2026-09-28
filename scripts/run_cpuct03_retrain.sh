#!/usr/bin/env bash
# Retrain + eval under c_puct=0.3 (Phase 3), 2026-09-23.
#
# Context: Phase 2 (c_puct=0.5, rankinglossweight=0.0) brought causal_mcts
# strategy accuracy from a SIGNIFICANT regression vs random_floor
# (12.07% mean, p=0.0218, bad direction) to a null result (13.87% mean,
# p=0.678) -- no longer significantly worse than random, but not
# significantly better either. Per-seed diagnostic (diagnostics/
# seed43_puct_and_value_seed{42,43,44}.json) found seed43's PUCT
# exploration/Q ratio (1.96) sits BETWEEN seed42 (2.13) and seed44 (1.47)
# with no monotonic link to accuracy -- so further c_puct tuning is not
# expected to fix seed43 specifically (its issue is an early, seed-specific
# value-head plateau at step ~300, not PUCT imbalance). This run tests
# whether c_puct=0.3 (~40% lower than Phase 2) pushes seeds 42/44 --
# which DO show room per the diagnostic -- above the random_floor mean,
# while accepting seed43 likely stays flat (~10%).
#
# Training is FRESH (not resumed) for the same reason as Phase 2: c_puct
# affects the MCTS search used to generate policy targets during training
# itself, not just eval-time search. Phase 2's checkpoints (encoding
# c_puct=0.5-shaped targets) were backed up to checkpoints_backup_cpuct_0p5/
# before this script overwrites checkpoints/ in place.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[cpuct03_retrain] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[cpuct03_retrain] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[cpuct03_retrain] FAILED (exit $rc): $*" >&2
  fi
  return $rc
}

mkdir -p results/logs results/configs diagnostics

echo "=== 0. Record resolved config + environment ==="
python -c "
import json, os, subprocess, sys
os.chdir('$ROOT')
from train.utils import load_merged_config
cfg = load_merged_config('$ROOT')
try:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
except Exception:
    commit = 'unknown'
import torch, transformers
from esc.state import ESCState
record = {
    'resolved_config': cfg,
    'git_commit': commit + ' (plus uncommitted whole-window + ReLU + rankinglossweight=0.0 + cpuct=0.3 changes)',
    'state_dim': ESCState.get_state_dim(),
    'cpuct': cfg.get('cpuct', cfg.get('c_puct')),
    'rankinglossweight': cfg.get('rankinglossweight', cfg.get('ranking_loss_weight')),
    'eval_n_per_seed': 500,
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
}
with open('results/configs/resolved_config_cpuct03.json', 'w') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
"

echo "=== 1. Validate existing artifacts (preprocessing skipped -- reused) ==="
run_capped 60 python -m scripts.validate_data_pipeline 2>&1 | tee results/logs/validate_cpuct03.txt
if grep -q "skip train.pt" results/logs/validate_cpuct03.txt; then
  echo "[cpuct03_retrain] ABORT: train.pt missing." >&2
  exit 1
fi

SEEDS=(42 43 44)
echo "=== 2. Training (FRESH init): causal_mcts + flow_ablation, seeds ${SEEDS[*]} (cpuct=0.3, rankinglossweight=0.0) ==="
for seed in "${SEEDS[@]}"; do
  run_capped 2400 python -m scripts.run_causal_mcts --seed "$seed" \
    2>&1 | tee "results/logs/causal_mcts_seed${seed}_train_cpuct03.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/causal_mcts_seed${seed}_train_cpuct03.txt"; then
    echo "[cpuct03_retrain] ABORT: causal_mcts seed=$seed used synthetic fallback." >&2
    exit 1
  fi
  run_capped 300 python -m scripts.run_flow_ablation --seed "$seed" \
    2>&1 | tee "results/logs/flow_ablation_seed${seed}_train_cpuct03.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/flow_ablation_seed${seed}_train_cpuct03.txt"; then
    echo "[cpuct03_retrain] ABORT: flow_ablation seed=$seed used synthetic fallback." >&2
    exit 1
  fi
done

echo "=== 3. Eval timing probe ==="
python -c "
import time, os
os.chdir('$ROOT')
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config
cfg = load_merged_config('$ROOT')
bb = QwenBackbone(cfg.get('backbone_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'))
bb.load()
t0 = time.perf_counter()
for _ in range(5):
    bb.generate_response('As the Supporter, respond supportively to someone feeling anxious about work.')
elapsed = (time.perf_counter() - t0) / 5
print(f'PROBE_SECONDS_PER_CALL={elapsed:.3f}')
" 2>&1 | tee results/logs/timing_probe_cpuct03.txt

EVAL_N=500
echo "=== 4. Eval: causal_mcts / flow_ablation / random_floor x seeds ${SEEDS[*]} (n=$EVAL_N) ==="
for seed in "${SEEDS[@]}"; do
  for system in causal_mcts flow_ablation random_floor; do
    run_capped 6000 python -m eval.run_eval --system "$system" --seed "$seed" --n "$EVAL_N" \
      2>&1 | tee "results/logs/eval_${system}_seed${seed}_cpuct03.txt"
  done
done

echo "=== 5. Significance aggregation ==="
run_capped 120 python -m eval.significance 2>&1 | tee results/logs/significance_cpuct03.txt

echo "=== 6. Latency baseline ==="
run_capped 400 python -m eval.latency_baseline 2>&1 | tee results/logs/latency_baseline_cpuct03.txt

echo "=== 7. Qualitative extraction ==="
run_capped 60 python -m eval.qualitative 2>&1 | tee results/logs/qualitative_cpuct03.txt

echo "=== 8. HF cache cleanup ==="
run_capped 60 python -m scripts.cleanup_hf_cache 2>&1 | tee results/logs/cleanup_cpuct03.txt

echo "=== 9. Offline sanity checks ==="
run_capped 120 pytest -m "not integration" --tb=short -q 2>&1 | tee results/logs/pytest_cpuct03.txt
run_capped 60 ruff check . 2>&1 | tee results/logs/ruff_cpuct03.txt

echo "=== DONE ==="
