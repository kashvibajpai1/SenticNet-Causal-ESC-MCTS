#!/usr/bin/env bash
# Full retrain + eval under whole-window encoding (Option B), 2026-09-21.
#
# Preprocessing is now ~4x faster than the mean-pooling/concatenation eras
# (0.70s/conversation measured, vs ~3.0s/conversation before) -- fewer
# Qwen calls per conversation under whole-window (3 pooled_embedding calls
# vs up to 10). Training/eval compute cost is unaffected (state_dim is
# back to 1283, same as before whole-window; only state *content*
# changed). Calibrated total: ~3.2h (preprocess ~15min + train ~90min +
# eval ~77min + tail ~10min).
#
# Value-network saturation logging added to scripts/run_causal_mcts.py and
# scripts/run_flow_ablation.py (every 50 steps) -- see
# checkpoints/{system}/seed{N}/value_saturation_log.json after training.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[wholewindow_retrain] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[wholewindow_retrain] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[wholewindow_retrain] FAILED (exit $rc): $*" >&2
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
    commit = 'unknown (uncommitted whole-window changes)'
import torch, transformers
record = {
    'resolved_config': cfg,
    'git_commit': commit,
    'state_dim': __import__('esc.state', fromlist=['ESCState']).ESCState.get_state_dim(),
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
}
with open('results/configs/resolved_config.json', 'w') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
"

echo "=== 1. Preprocessing (whole-window encoding, full ESConv corpus) ==="
# Measured 0.70s/conversation, ~15.2min extrapolated for 1300 conversations.
# Cap gives generous headroom for network/HF variability.
run_capped 3600 python -m scripts.preprocess_datasets --skip-cornell \
  2>&1 | tee results/logs/preprocess.txt

echo "=== 2. Validate artifacts ==="
run_capped 60 python -m scripts.validate_data_pipeline 2>&1 | tee results/logs/validate.txt
if grep -q "skip train.pt" results/logs/validate.txt; then
  echo "[wholewindow_retrain] ABORT: train.pt missing, preprocessing failed." >&2
  exit 1
fi

SEEDS=(42 43 44)
echo "=== 3. Training: causal_mcts + flow_ablation, seeds ${SEEDS[*]} (with saturation logging) ==="
for seed in "${SEEDS[@]}"; do
  run_capped 2400 python -m scripts.run_causal_mcts --seed "$seed" \
    2>&1 | tee "results/logs/causal_mcts_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/causal_mcts_seed${seed}_train.txt"; then
    echo "[wholewindow_retrain] ABORT: causal_mcts seed=$seed used synthetic fallback." >&2
    exit 1
  fi
  run_capped 300 python -m scripts.run_flow_ablation --seed "$seed" \
    2>&1 | tee "results/logs/flow_ablation_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/flow_ablation_seed${seed}_train.txt"; then
    echo "[wholewindow_retrain] ABORT: flow_ablation seed=$seed used synthetic fallback." >&2
    exit 1
  fi
done

echo "=== 4. Eval timing probe ==="
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
" 2>&1 | tee results/logs/timing_probe.txt

EVAL_N=150
echo "=== 5. Eval: causal_mcts / flow_ablation / random_floor x seeds ${SEEDS[*]} (n=$EVAL_N) ==="
for seed in "${SEEDS[@]}"; do
  for system in causal_mcts flow_ablation random_floor; do
    run_capped 2400 python -m eval.run_eval --system "$system" --seed "$seed" --n "$EVAL_N" \
      2>&1 | tee "results/logs/eval_${system}_seed${seed}.txt"
  done
done

echo "=== 6. Significance aggregation ==="
run_capped 120 python -m eval.significance 2>&1 | tee results/logs/significance.txt

echo "=== 7. Latency baseline ==="
run_capped 400 python -m eval.latency_baseline 2>&1 | tee results/logs/latency_baseline.txt

echo "=== 8. Qualitative extraction ==="
run_capped 60 python -m eval.qualitative 2>&1 | tee results/logs/qualitative.txt

echo "=== 9. HF cache cleanup ==="
run_capped 60 python -m scripts.cleanup_hf_cache 2>&1 | tee results/logs/cleanup.txt

echo "=== 10. Offline sanity checks ==="
run_capped 120 pytest -m "not integration" --tb=short -q 2>&1 | tee results/logs/pytest.txt
run_capped 60 ruff check . 2>&1 | tee results/logs/ruff.txt

echo "=== DONE ==="
