#!/usr/bin/env bash
# Eval-only re-run at 500 instances/seed (up from 150) to fix the statistical-
# power problem diagnosed 2026-09-21: at n=150, instance-level noise
# (+/-3.14pp binomial SE) was ~3x the observed effect size (+1.11pp),
# producing p=0.817 despite a real-looking effect. Noise at n=500 drops to
# ~+/-1.88pp (see diagnostics/ for the full PUCT/reward-sign/statistical-power
# writeup -- neither hypothesized architectural issue (PUCT exploration
# collapse, reward-sign mismatch) was actually found to be occurring).
#
# TRAINING IS SKIPPED: this run reuses the already-trained, already-verified
# ReLU checkpoints from checkpoints/{causal_mcts,flow_ablation}/seed{42,43,44}
# (backed up untouched to checkpoints_backup_relu_baseline/ before this
# script was written) and the existing artifacts/states/*.pt. Only the eval
# stage runs, at the larger instance count. num_simulations stays at 30
# (config/train_shared.yaml's cpuct=1.5, num_simulations=30 -- Option A from
# the diagnostic, eval-set size was judged the primary bottleneck, not
# search depth).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[relu_500instance_eval] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[relu_500instance_eval] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[relu_500instance_eval] FAILED (exit $rc): $*" >&2
  fi
  return $rc
}

mkdir -p results/logs results/configs

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
    'git_commit': commit + ' (plus uncommitted whole-window + ReLU changes; training NOT re-run this pass)',
    'state_dim': ESCState.get_state_dim(),
    'value_network_activation': 'ReLU (unchanged from 2026-09-21 run)',
    'eval_n_per_seed': 500,
    'num_simulations': cfg.get('num_simulations', cfg.get('numsimulations')),
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
}
with open('results/configs/resolved_config_500instance.json', 'w') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
"

echo "=== 1. Sanity-check existing checkpoints are present (no training this pass) ==="
SEEDS=(42 43 44)
for seed in "${SEEDS[@]}"; do
  for f in checkpoints/causal_mcts/seed${seed}/policy.pt checkpoints/causal_mcts/seed${seed}/value.pt \
           checkpoints/causal_mcts/seed${seed}/transition.pt \
           checkpoints/flow_ablation/seed${seed}/policy.pt checkpoints/flow_ablation/seed${seed}/value.pt; do
    if [ ! -f "$f" ]; then
      echo "[relu_500instance_eval] ABORT: missing checkpoint file $f" >&2
      exit 1
    fi
  done
done
echo "[relu_500instance_eval] all seed42/43/44 causal_mcts + flow_ablation checkpoints present"

echo "=== 2. Eval timing probe ==="
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
" 2>&1 | tee results/logs/timing_probe_500instance.txt

EVAL_N=500
echo "=== 3. Eval: causal_mcts / flow_ablation / random_floor x seeds ${SEEDS[*]} (n=$EVAL_N) ==="
for seed in "${SEEDS[@]}"; do
  for system in causal_mcts flow_ablation random_floor; do
    run_capped 6000 python -m eval.run_eval --system "$system" --seed "$seed" --n "$EVAL_N" \
      2>&1 | tee "results/logs/eval_${system}_seed${seed}_500instance.txt"
  done
done

echo "=== 4. Significance aggregation ==="
run_capped 120 python -m eval.significance 2>&1 | tee results/logs/significance_500instance.txt

echo "=== 5. Latency baseline ==="
run_capped 400 python -m eval.latency_baseline 2>&1 | tee results/logs/latency_baseline_500instance.txt

echo "=== 6. Qualitative extraction ==="
run_capped 60 python -m eval.qualitative 2>&1 | tee results/logs/qualitative_500instance.txt

echo "=== 7. HF cache cleanup ==="
run_capped 60 python -m scripts.cleanup_hf_cache 2>&1 | tee results/logs/cleanup_500instance.txt

echo "=== 8. Offline sanity checks ==="
run_capped 120 pytest -m "not integration" --tb=short -q 2>&1 | tee results/logs/pytest_500instance.txt
run_capped 60 ruff check . 2>&1 | tee results/logs/ruff_500instance.txt

echo "=== DONE ==="
