# causal-esc-mcts

Latency-aware causal MCTS over an Emotional Support Conversation (ESC) MDP, with a FlowMCTS-ablation baseline. This repository provides offline ESC data ingestion (Hugging Face ESConv and Cornell ESC via ConvoKit), serialized state artifacts, and training entry points that consume those artifacts without coupling search code to dataset backends.

**"FlowMCTS-ablation" is an in-house flow-matching ablation** (policy/value networks trained with a GFlowNet-style flow-consistency + ranking loss, no MCTS search) — it is **not** a reproduction of the published AFlow system (Zou et al.). It borrows the flow-matching idea as a no-search baseline to isolate what MCTS search contributes. See [flow/README.md](flow/README.md) and [results/README.md](results/README.md).

**Backbone honesty note:** the paper's originally stated "Qwen-9B" backbone was, before this pass, a complete stub (`models/backbone_qwen.py`) that returned all-zero embeddings and a hardcoded reply string — no real encoding or generation ever happened. It now uses a real, small, local, open-weight `Qwen2.5-0.5B-Instruct` model (downsized from "9B" because this project has no GPU access). See [results/README.md](results/README.md) for the full disclosure of this substitution and other known limitations.

---

## Quick Start

From a fresh clone, run the pipeline in this order:

```bash
pip install -r requirements-data-full.txt
python -m scripts.preprocessdatasets --skip-cornell
python -m scripts.validatedatapipeline
python -m scripts.runflowablation
python -m scripts.runcausalmcts
pytest
```

Or use the canonical end-to-end runner:

```bash
bash scripts/run_end_to_end.sh
```

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `pip install -r requirements-data-full.txt` | Installs PyTorch, `datasets` (ESConv), and ConvoKit (Cornell ESC). |
| 2 | `python -m scripts.preprocessdatasets --skip-cornell` | Downloads ESConv, normalizes dialogs, and writes JSONL + serialized state tensors under `artifacts/`. Use `--skip-cornell` for a lightweight HF-only smoke run; omit it for the full ESConv + Cornell benchmark. |
| 3 | `python -m scripts.validatedatapipeline` | Sanity-checks `artifacts/processed/conversations.jsonl` and `artifacts/states/train.pt` (shapes, bundles, state dim). |
| 4 | `python -m scripts.runflowablation` | Trains the FlowMCTS-ablation policy/value networks on `artifacts/states/train.pt` (or synthetic batches if artifacts are missing). |
| 5 | `python -m scripts.runcausalmcts` | Trains policy, value, and transition models with MCTS on serialized root states (or env-generated fallbacks if artifacts are missing). |
| 6 | `pytest` | Runs unit tests (integration tests that download corpora are skipped by default). |

**Real experiment run:** `bash scripts/run_full_experiment.sh` runs the full, bounded, real pipeline (real Qwen encoder + generation, 3 seeds x 2 systems + a random-policy floor, real BLEU/ROUGE/BERTScore/Distinct-2/strategy-accuracy metrics, significance testing, a latency comparison, qualitative examples, then HF-cache cleanup) and writes everything to `results/`. See [results/README.md](results/README.md) for what it produced, its scope, and its limitations.

**Config defaults:** `config/train_shared.yaml` now holds "reduced-but-real" values (`maxsteps: 300`, `num_simulations: 15`, `batchsize: 16`) chosen to finish on a CPU-only machine in a bounded amount of time — distinct from both the old `maxsteps: 10` smoke test and an unverified paper-scale claim. See [config/README.md](config/README.md).

**Synthetic fallback:** If `artifacts/states/train.pt` is missing or empty, the training scripts print a warning and use random or env-generated states. That path verifies the training loop but does **not** reproduce paper experiments.

---

## Artifacts

Preprocessing writes gitignored outputs under `artifacts/`:

| Path | Description |
|------|-------------|
| `artifacts/processed/conversations.jsonl` | Normalized `ConversationRecord` entries (one JSON object per line). |
| `artifacts/states/train.pt` | Training split: state bundles + stacked tensors for dataloaders. |
| `artifacts/states/valid.pt` | Validation split. |
| `artifacts/states/test.pt` | Test split. |

Each `.pt` file contains a `"bundles"` list (for `ESCStateBundleDataset`) and optional `"state_tensors"` (for `ESCStateTensorDataset`).

---

## How this repository maps to the paper

| Paper component | Repository location |
|-----------------|---------------------|
| ESC MDP (state, causal graph, actions, reward) | `esc/` |
| MCTS planner (PUCT, tree search) | `mcts/` |
| Policy, value, transition models | `models/` |
| FlowMCTS-ablation flow objectives | `flow/`, `train/trainer_flow_ablation.py` |
| Causal MCTS training | `train/trainer_causal_mcts.py`, `scripts/runcausalmcts.py` |
| Offline data pipeline (ESConv + Cornell) | `data/`, `scripts/preprocessdatasets.py` |
| Hyperparameters | `config/env.yaml`, `config/train_shared.yaml` |

The `data/` package owns corpus I/O; trainers read only serialized states under `artifacts/states/`, not live HF or ConvoKit APIs.

---

## Project structure

```text
causal-esc-mcts/
├── data/                    # Offline dataset layer
├── esc/                     # ESCState, ESCEnv, CausalGraph, rewards
├── mcts/                    # MCTS planner, TreeNode, PUCT
├── models/                  # PolicyNetwork, ValueNetwork, TransitionModel
├── flow/                    # FlowMCTS-ablation flow objectives and losses
├── train/                   # Trainers and PyTorch datasets
├── inference/               # Interactive CLI
├── scripts/
│   ├── preprocessdatasets.py      # canonical alias → preprocess_datasets
│   ├── validatedatapipeline.py    # canonical alias → validate_data_pipeline
│   ├── runflowablation.py        # canonical alias → run_flow_ablation
│   ├── runcausalmcts.py           # canonical alias → run_causal_mcts
│   └── run_end_to_end.sh          # full pipeline runner
├── artifacts/               # Gitignored outputs (created by preprocessing)
├── config/
└── tests/
```

Legacy underscore module names (`scripts.preprocess_datasets`, etc.) remain available and delegate to the same implementations.

---

## Requirements

- **Python 3.10+** (recommended for ConvoKit / spaCy stacks).
- `requirements-data-full.txt` for ESConv + Cornell; `requirements.txt` if you intentionally skip Cornell.

---

## Subdirectory documentation

| Directory | README |
|-----------|--------|
| `esc/` | [ESC MDP](esc/README.md) |
| `mcts/` | [MCTS planner](mcts/README.md) |
| `models/` | [Networks and backbone](models/README.md) |
| `data/` | [Data pipeline](data/README.md) |
| `flow/` | [Flow objectives](flow/README.md) |
| `train/` | [Trainers and datasets](train/README.md) |
| `inference/` | [Interactive CLI](inference/README.md) |
| `scripts/` | [Entry points](scripts/README.md) |
| `config/` | [YAML configuration](config/README.md) |
| `tests/` | [Test suite](tests/README.md) |
| `utils/` | [Seed and logging](utils/README.md) |

---

## CI

GitHub Actions runs lint (`ruff`) and unit tests (`pytest -m "not integration"`) on Python 3.10 and 3.11. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
