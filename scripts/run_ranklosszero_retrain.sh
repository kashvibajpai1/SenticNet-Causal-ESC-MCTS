#!/usr/bin/env bash
# Retrain + eval under the rank_loss-weight-zero fix, 2026-09-22.
#
# Root cause (see the deep diagnostic): rank_loss = clamp(margin - mean(batch
# values), 0) had a batch-uniform gradient dragging every value output toward
# margin=1.0 regardless of instance rewards, causing the ~0.98/std-0.017
# output collapse under both tanh and ReLU, in both causal_mcts and
# flow_ablation. Fix applied: config/train_shared.yaml's rankinglossweight
# 1.0 -> 0.0. Dead flow_loss bug (MSE(x, x.detach()) == 0 always) left
# UNFIXED per explicit instruction -- flow_ablation's value head therefore
# still has no real per-instance training signal in this run; only
# causal_mcts's MSE-against-real-reward value_loss is now uncontested.
#
# NEW this run: train/trainer_causal_mcts.py's train_step() now also returns
# pearson_r/reward_mean/reward_std/value_mean/value_std (computed on each
# training batch); scripts/run_causal_mcts.py logs these every 100 steps to
# diagnostics/value_pearson_r_seed{N}.json (one file per seed, not the single
# shared path the request suggested, to avoid seeds 43/44 clobbering seed
# 42's log). flow_ablation has NO real reward in its training loop at all
# (its trainer never touches esc/reward.py), so no Pearson-r logging was
# added there -- flagged, not silently faked.
#
# Preprocessing is SKIPPED: state encoding is unaffected by this loss-weight
# change, so existing artifacts/states/*.pt are reused. Training IS re-run
# fresh (fresh init, not resumed from the backed-up pre-fix checkpoints,
# per explicit instruction "fresh is cleaner").
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[ranklosszero_retrain] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[ranklosszero_retrain] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[ranklosszero_retrain] FAILED (exit $rc): $*" >&2
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
    'git_commit': commit + ' (plus uncommitted whole-window + ReLU + rankinglossweight=0.0 changes)',
    'state_dim': ESCState.get_state_dim(),
    'value_network_activation': 'ReLU (unchanged)',
    'rankinglossweight': cfg.get('rankinglossweight', cfg.get('ranking_loss_weight')),
    'eval_n_per_seed': 500,
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
}
with open('results/configs/resolved_config_ranklosszero.json', 'w') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
"

echo "=== 1. Validate existing artifacts (preprocessing skipped -- reused) ==="
run_capped 60 python -m scripts.validate_data_pipeline 2>&1 | tee results/logs/validate_ranklosszero.txt
if grep -q "skip train.pt" results/logs/validate_ranklosszero.txt; then
  echo "[ranklosszero_retrain] ABORT: train.pt missing." >&2
  exit 1
fi

SEEDS=(42 43 44)
echo "=== 2. Training (FRESH init): causal_mcts + flow_ablation, seeds ${SEEDS[*]} (rankinglossweight=0.0) ==="
for seed in "${SEEDS[@]}"; do
  run_capped 2400 python -m scripts.run_causal_mcts --seed "$seed" \
    2>&1 | tee "results/logs/causal_mcts_seed${seed}_train_ranklosszero.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/causal_mcts_seed${seed}_train_ranklosszero.txt"; then
    echo "[ranklosszero_retrain] ABORT: causal_mcts seed=$seed used synthetic fallback." >&2
    exit 1
  fi
  run_capped 300 python -m scripts.run_flow_ablation --seed "$seed" \
    2>&1 | tee "results/logs/flow_ablation_seed${seed}_train_ranklosszero.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/flow_ablation_seed${seed}_train_ranklosszero.txt"; then
    echo "[ranklosszero_retrain] ABORT: flow_ablation seed=$seed used synthetic fallback." >&2
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
" 2>&1 | tee results/logs/timing_probe_ranklosszero.txt

EVAL_N=500
echo "=== 4. Eval: causal_mcts / flow_ablation / random_floor x seeds ${SEEDS[*]} (n=$EVAL_N) ==="
for seed in "${SEEDS[@]}"; do
  for system in causal_mcts flow_ablation random_floor; do
    run_capped 6000 python -m eval.run_eval --system "$system" --seed "$seed" --n "$EVAL_N" \
      2>&1 | tee "results/logs/eval_${system}_seed${seed}_ranklosszero.txt"
  done
done

echo "=== 5. Significance aggregation ==="
run_capped 120 python -m eval.significance 2>&1 | tee results/logs/significance_ranklosszero.txt

echo "=== 6. Latency baseline ==="
run_capped 400 python -m eval.latency_baseline 2>&1 | tee results/logs/latency_baseline_ranklosszero.txt

echo "=== 7. Qualitative extraction ==="
run_capped 60 python -m eval.qualitative 2>&1 | tee results/logs/qualitative_ranklosszero.txt

echo "=== 8. HF cache cleanup ==="
run_capped 60 python -m scripts.cleanup_hf_cache 2>&1 | tee results/logs/cleanup_ranklosszero.txt

echo "=== 9. Offline sanity checks ==="
run_capped 120 pytest -m "not integration" --tb=short -q 2>&1 | tee results/logs/pytest_ranklosszero.txt
run_capped 60 ruff check . 2>&1 | tee results/logs/ruff_ranklosszero.txt

echo "=== DONE ==="
