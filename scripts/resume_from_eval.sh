#!/usr/bin/env bash
# Resume run_full_experiment.sh from stage 4 (eval) onward.
#
# Used when preprocessing + training (stages 0-3) already completed and are
# safely checkpointed on disk, but the eval stage was interrupted -- e.g.
# 2026-09-11: the eval stage hit severe swap thrashing on this 8GB machine
# (other apps competing for RAM) and was killed after 2 failed combos, with
# zero partial output written (eval.run_eval only writes its JSON after all
# instances finish, so a killed run leaves no stale/partial file behind).
#
# Preconditions (not checked here -- verify before running):
# - artifacts/states/{train,valid,test}.pt exist (preprocessing done)
# - checkpoints/{causal_mcts,flow_ablation}/seed{42,43,44}/*.pt exist (training done)
#
# Stages 4-10 are copied verbatim from scripts/run_full_experiment.sh so the
# two scripts stay in sync -- if you change caps/EVAL_N there, mirror it here.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[resume_from_eval] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[resume_from_eval] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[resume_from_eval] FAILED (exit $rc): $*" >&2
  fi
  return $rc
}

mkdir -p results/logs results/configs

SEEDS=(42 43 44)

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

EVAL_N=150
PROBE_S=$(grep -o 'PROBE_SECONDS_PER_CALL=.*' results/logs/timing_probe.txt | cut -d= -f2)
if [ -n "${PROBE_S:-}" ]; then
  OVER=$(python3 -c "print(1 if float('$PROBE_S') > 4.0 else 0)")
  if [ "$OVER" = "1" ]; then
    echo "[resume_from_eval] WARNING: generate() is slower than calibration assumed (${PROBE_S}s/call vs ~2.9s/instance measured isolated) -- eval may exceed its wall-clock cap. Not auto-trimming EVAL_N=$EVAL_N." >&2
  fi
fi

echo "=== 5. Eval: causal_mcts / flow_ablation / random_floor x seeds ${SEEDS[*]} (n=$EVAL_N) ==="
# Cap raised 900->2400s after two real failures: a 10-instance instrumented
# probe showed per-instance "encode" time (ESCState.from_dialogue's real
# Qwen cause-span extraction) ranging 1.4s-6.2s, not the flat ~1.7s assumed
# earlier -- it scales with how many seeker turns are in the conversation
# so far, and n=150 draws instances from much deeper into conversations
# than a small sample does. 2400s gives real headroom for that tail
# variance instead of guessing again.
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
