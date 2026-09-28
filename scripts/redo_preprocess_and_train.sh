#!/usr/bin/env bash
# Redo stages 0-3 only (config record, preprocessing, validate, training)
# after the 2026-09-12 phase-computation fix in esc/state.py -- the bug was
# baked into artifacts/states/*.pt and checkpoints/ trained against it, so
# both need rebuilding from scratch. Deliberately stops after training:
# per the plan agreed with the user, eval is run as ONE combo first and
# timed for real before committing to the full 9-combo loop again (the
# 900s cap was blown twice tonight by real per-instance tail latency +
# memory pressure -- see the conversation log).
#
# Mirrors scripts/run_full_experiment.sh's stages 0-3 verbatim so the
# scripts stay in sync -- if you change caps there, mirror it here.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[redo_preprocess_and_train] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[redo_preprocess_and_train] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[redo_preprocess_and_train] FAILED (exit $rc): $*" >&2
  fi
  return $rc
}

mkdir -p results/logs results/configs

echo "=== 0. Record resolved config + environment for reproducibility ==="
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
record = {
    'resolved_config': cfg,
    'git_commit': commit,
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
}
with open('results/configs/resolved_config.json', 'w') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
"

echo "=== 1. Preprocessing (full ESConv corpus, real Qwen encoder, phase fix applied) ==="
run_capped 7200 python -m scripts.preprocess_datasets --skip-cornell \
  2>&1 | tee results/logs/preprocess.txt

echo "=== 2. Validate artifacts ==="
run_capped 60 python -m scripts.validate_data_pipeline 2>&1 | tee results/logs/validate.txt
if grep -q "skip train.pt" results/logs/validate.txt; then
  echo "[redo_preprocess_and_train] ABORT: train.pt missing, preprocessing failed." >&2
  exit 1
fi

SEEDS=(42 43 44)
echo "=== 3. Training: causal_mcts + flow_ablation, seeds ${SEEDS[*]} ==="
for seed in "${SEEDS[@]}"; do
  run_capped 2400 python -m scripts.run_causal_mcts --seed "$seed" \
    2>&1 | tee "results/logs/causal_mcts_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/causal_mcts_seed${seed}_train.txt"; then
    echo "[redo_preprocess_and_train] ABORT: causal_mcts seed=$seed used synthetic fallback." >&2
    exit 1
  fi
  run_capped 300 python -m scripts.run_flow_ablation --seed "$seed" \
    2>&1 | tee "results/logs/flow_ablation_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/flow_ablation_seed${seed}_train.txt"; then
    echo "[redo_preprocess_and_train] ABORT: flow_ablation seed=$seed used synthetic fallback." >&2
    exit 1
  fi
done

echo "=== DONE (stages 0-3 only; eval deliberately not started -- run one combo manually and time it first) ==="
