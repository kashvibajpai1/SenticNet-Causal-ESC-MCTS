#!/usr/bin/env bash
# Run the full repository pipeline from a fresh clone (smoke-test defaults).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_step() {
  echo ""
  echo "==> $1"
  shift
  "$@"
}

run_step "Install dependencies" pip install -r requirements-data-full.txt
run_step "Preprocess datasets (HF ESConv; skip Cornell for lightweight smoke)" \
  python -m scripts.preprocessdatasets --skip-cornell
run_step "Validate artifacts" python -m scripts.validatedatapipeline
run_step "FlowMCTS-ablation training" python -m scripts.runflowablation
run_step "Causal MCTS training" python -m scripts.runcausalmcts
run_step "Unit tests" pytest

echo ""
echo "==> Pipeline finished successfully."
