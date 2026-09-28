#!/usr/bin/env bash
# Real, bounded end-to-end experiment run: preprocess -> train (2x3 seeds) ->
# eval (3x3 seeds) -> significance -> latency baseline -> qualitative -> cleanup.
#
# Every phase runs under a hard wall-clock cap (macOS has no `timeout(1)`,
# so this uses a perl alarm() wrapper instead) so the run cannot silently
# hang or balloon past the user's "don't run too long" instruction.
#
# 2026-09-11 recalibration (causal_esc_mcts_fix_spec.md Parts A/B):
# full ESConv corpus (A4), the missing policy loss (A1/A2), and the fixed
# eval phase-0 sampling bug (B1) are all in as specified. maxsteps,
# num_simulations, and eval --n were scaled down from the spec's literal
# values to fit an explicit 3-4 hour budget after calibration showed the
# literal spec values extrapolate to 30+ hours on this machine -- see the
# comment block in config/train_shared.yaml for the exact numbers and how
# to run the full-scope version later.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[run_full_experiment] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[run_full_experiment] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[run_full_experiment] FAILED (exit $rc): $*" >&2
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

echo "=== 1. Preprocessing (full ESConv corpus, real Qwen encoder) ==="
# No --max-esconv: uses the full ESConv corpus (~1,300 conversations across
# train/validation/test splits: 910/195/195, confirmed from the live HF
# dataset) rather than the 100-conversations-per-split pilot cap.
# Cap sized from calibration on this machine: measured ~3.0s/conversation
# warm-cache marginal rate -> ~3900s for 1300 conversations; cap set with
# ~85% headroom for cold-cache download + variance.
run_capped 7200 python -m scripts.preprocess_datasets --skip-cornell \
  2>&1 | tee results/logs/preprocess.txt

echo "=== 2. Validate artifacts ==="
run_capped 60 python -m scripts.validate_data_pipeline 2>&1 | tee results/logs/validate.txt
if grep -q "skip train.pt" results/logs/validate.txt; then
  echo "[run_full_experiment] ABORT: train.pt missing, preprocessing failed." >&2
  exit 1
fi

SEEDS=(42 43 44)
echo "=== 3. Training: causal_mcts + flow_ablation, seeds ${SEEDS[*]} ==="
# causal_mcts@num_simulations=30,maxsteps=800 measured ~2.17s/step ->
# ~1738s/seed; cap gives ~38% headroom. flow_ablation is a pure feedforward
# batch (no MCTS), measured ~5ms/step -> negligible; same generous cap
# reused as a shared safety net, not a real constraint for it.
for seed in "${SEEDS[@]}"; do
  run_capped 2400 python -m scripts.run_causal_mcts --seed "$seed" \
    2>&1 | tee "results/logs/causal_mcts_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/causal_mcts_seed${seed}_train.txt"; then
    echo "[run_full_experiment] ABORT: causal_mcts seed=$seed used synthetic fallback." >&2
    exit 1
  fi
  run_capped 300 python -m scripts.run_flow_ablation --seed "$seed" \
    2>&1 | tee "results/logs/flow_ablation_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/flow_ablation_seed${seed}_train.txt"; then
    echo "[run_full_experiment] ABORT: flow_ablation seed=$seed used synthetic fallback." >&2
    exit 1
  fi
done

echo "=== 4. Eval timing probe (5 generate() calls) ==="
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

# EVAL_N=150: raised from the pilot's 40 (B2), but well short of the ~2,700
# instances B1's fix makes eligible across the full test split ("most/all"
# per the spec) -- that would cost ~20h across 3 systems x 3 seeds at the
# ~3s/instance measured on this machine, blowing the 3-4h budget. 150 is
# fixed from calibration, not derived from the live probe below: the old
# probe->trim-to-20 heuristic assumed pilot-era timing and would have
# silently overridden this calibrated value, so it's now a warning only.
EVAL_N=150
PROBE_S=$(grep -o 'PROBE_SECONDS_PER_CALL=.*' results/logs/timing_probe.txt | cut -d= -f2)
if [ -n "${PROBE_S:-}" ]; then
  OVER=$(python3 -c "print(1 if float('$PROBE_S') > 4.0 else 0)")
  if [ "$OVER" = "1" ]; then
    echo "[run_full_experiment] WARNING: generate() is slower than calibration assumed (${PROBE_S}s/call vs ~2.3-3.4s measured) -- eval may exceed its wall-clock cap. Not auto-trimming EVAL_N=$EVAL_N; rerun calibration if this keeps happening." >&2
  fi
fi

echo "=== 5. Eval: causal_mcts / flow_ablation / random_floor x seeds ${SEEDS[*]} (n=$EVAL_N) ==="
# Cap 900->2400s, 2026-09-11: two real eval runs timed out at 900s with
# ZERO instances completed. A 10-instance instrumented probe showed
# per-instance "encode" time (ESCState.from_dialogue's real Qwen
# cause-span extraction) ranging 1.4s-6.2s, not the flat ~1.7s originally
# assumed -- it scales with how many seeker turns are in the conversation
# so far, and n=150 draws instances from much deeper into conversations
# than a small calibration sample does. 2400s gives real headroom for
# that tail variance. (A concurrent contributor on the machine this ran
# on: severe swap thrashing from other apps competing for an 8GB
# machine's RAM -- see the conversation log for that incident. Not
# something this script can control, but worth knowing if this cap still
# isn't enough: check `sysctl vm.swapusage` before assuming the code regressed.)
for seed in "${SEEDS[@]}"; do
  for system in causal_mcts flow_ablation random_floor; do
    run_capped 2400 python -m eval.run_eval --system "$system" --seed "$seed" --n "$EVAL_N" \
      2>&1 | tee "results/logs/eval_${system}_seed${seed}.txt"
  done
done

echo "=== 6. Significance aggregation ==="
run_capped 120 python -m eval.significance 2>&1 | tee results/logs/significance.txt

echo "=== 7. Latency baseline (bounded, real Qwen, LAST model use) ==="
run_capped 400 python -m eval.latency_baseline 2>&1 | tee results/logs/latency_baseline.txt

echo "=== 8. Qualitative extraction ==="
run_capped 60 python -m eval.qualitative 2>&1 | tee results/logs/qualitative.txt

echo "=== 9. HF cache cleanup ==="
run_capped 60 python -m scripts.cleanup_hf_cache 2>&1 | tee results/logs/cleanup.txt

echo "=== 10. Offline sanity checks ==="
run_capped 120 pytest -m "not integration" --tb=short -q 2>&1 | tee results/logs/pytest.txt
run_capped 60 ruff check . 2>&1 | tee results/logs/ruff.txt

echo "=== DONE ==="
