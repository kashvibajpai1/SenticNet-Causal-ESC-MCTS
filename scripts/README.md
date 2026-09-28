# scripts -- Entry-Point Scripts

This directory contains the four runnable scripts that cover the complete pipeline from raw corpus preprocessing through training. All scripts are designed to be run as modules (`python -m scripts.<name>`) from the repository root.

---

## Execution order

For a fresh clone on any machine:

```bash
# 1. Install full dependencies (including ConvoKit for Cornell ESC)
pip install -r requirements-data-full.txt

# 2. Download and preprocess both corpora
python -m scripts.preprocess_datasets

# 3. Verify the artifacts are correct
python -m scripts.validate_data_pipeline

# 4a. Run FlowMCTS-ablation training
python -m scripts.run_flow_ablation

# 4b. Run full Causal MCTS training
python -m scripts.run_causal_mcts
```

Steps 4a and 4b are independent and can be run in any order or in parallel on separate GPUs.

---

## `preprocess_datasets.py`

**Purpose:** Download both corpora, normalize them into `ConversationRecord` format, build `ESCState` objects, and serialize everything to `artifacts/`.

**What it does:**

1. Iterates `iter_esconv_records()` for splits `train`, `validation`, and `test`.
2. Optionally iterates `iter_cornell_esc_records()` (skipped with `--skip-cornell`).
3. Runs `preprocess_record()` on each record and drops records that fail validation.
4. Writes `artifacts/processed/conversations.jsonl` with all cleaned records.
5. For each split, calls `build_esc_state_from_record()` on each record to produce an `ESCState`.
6. Serializes the states to `artifacts/states/train.pt`, `artifacts/states/valid.pt`, and `artifacts/states/test.pt`.

Each `.pt` file contains both a `"bundles"` list (for `ESCStateBundleDataset`) and a pre-stacked `"state_tensors"` tensor (for `ESCStateTensorDataset`).

**Arguments:**

- `--skip-cornell`: skip the Cornell ESC download and use only ESConv.
- `--max-esconv N`: limit ESConv to N conversations per split (useful for fast debugging).

**Expected output:**

```
artifacts/processed/conversations.jsonl   (~2300 lines)
artifacts/states/train.pt
artifacts/states/valid.pt
artifacts/states/test.pt
```

**Failure modes:**

If ConvoKit is not installed and `--skip-cornell` is not passed, the script exits with a clear error message explaining the missing dependency.

---

## `validate_data_pipeline.py`

**Purpose:** Sanity-check the serialized artifacts before launching training.

**What it checks:**

1. That `artifacts/states/train.pt` exists and can be loaded.
2. That `state_tensors` has shape `[N, 1283]` (N > 0, feature dim = 1283 = D_H + D_C + D_E + D_P).
3. That no NaN or Inf values exist in the state tensor batch.
4. That a sample of bundles can be reconstructed into valid `ESCState` objects via `esc_state_from_bundle()`.
5. That `artifacts/processed/conversations.jsonl` is readable and at least one record passes `validate_record()`.

Prints a summary of counts and shapes. Exits with a non-zero code if any check fails.

**When to run:** after every run of `preprocess_datasets.py` and before launching GPU training jobs.

---

## `run_flow_ablation.py`

**Purpose:** Train the policy and value networks using flow-matching flow consistency and ranking objectives (not MCTS) -- FlowMCTS-ablation, not a reproduction of published AFlow.

**What it does:**

1. Loads merged config from `config/env.yaml` + `config/train_shared.yaml`.
2. Instantiates `PolicyNetwork` and `ValueNetwork`.
3. Instantiates `FlowAblationTrainer`.
4. Attempts to load `artifacts/states/train.pt` as an `ESCStateTensorDataset`. If the file does not exist, falls back to synthetic random batches of shape `[batch_size, state_dim]`.
5. Runs `max_steps` training steps, printing the loss at each step.

**Fallback behaviour:** when no artifacts are present, synthetic batches are used. This allows the script to run immediately after cloning without preprocessing, which is useful for smoke-testing the training infrastructure.

**Config keys used:** `learningrate`, `batchsize`, `maxsteps`, `flowlossweight`, `rankinglossweight`, `gammamargin`.

---

## `run_causal_mcts.py`

**Purpose:** Train the policy, value, and transition model jointly using MCTS-collected rewards.

**What it does:**

1. Loads merged config.
2. Instantiates `LinearTransitionModel`, `ESCEnv`, `PolicyNetwork`, `ValueNetwork`, `MCTS`, and `CausalMCTSTrainer`.
3. Attempts to load `artifacts/states/train.pt` as an `ESCStateBundleDataset`. If absent, falls back to synthetic states from `env.reset()`.
4. At each training step, randomly samples a batch of states from the bundle dataset (or generates them), calls `trainer.train_step(states)`, and prints the loss breakdown.

**Fallback behaviour:** identical to the FlowMCTS-ablation script -- env-generated zero states are used when artifacts are missing.

**Config keys used:** all FlowMCTS-ablation keys plus `num_simulations`, `cpuct`, `max_horizon`.

---

## Artifacts directory

The `artifacts/` directory is gitignored. It is created by `preprocess_datasets.py`. On a GPU cluster, it can be placed on shared storage and shared across nodes, since the preprocessing step is CPU-bound and only needs to run once per experiment setup.

To place artifacts in a non-default location, modify the `artifacts_dir` variable at the top of `preprocess_datasets.py` and update the paths in the training scripts accordingly.
