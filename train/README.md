# train -- Training Infrastructure

This package contains trainers, dataset classes, and configuration utilities. Training code consumes only serialized artifacts under `artifacts/states/` or synthetic batches; it never imports from `data/dataset_sources.py` or calls Hugging Face/ConvoKit APIs. This separation is what allows the training scripts to run on a GPU cluster after a single offline preprocessing step.

---

## Files

### `trainer_flow_ablation.py` -- FlowAblationTrainer

`FlowAblationTrainer` trains the policy network `pi_theta` and value network `V_phi` using flow-matching flow consistency and ranking losses (FlowMCTS-ablation, not a reproduction of published AFlow). It does not use MCTS or the transition model; it is a purely supervised/self-supervised objective over flat state tensors.

**Constructor parameters:**

| Parameter | Type | Description |
|---|---|---|
| `policy` | `PolicyNetwork` | Policy to train |
| `value` | `ValueNetwork` | Value network to train |
| `config` | `dict` | Merged YAML config dict |

The optimizer is `torch.optim.Adam` over the parameters of both networks jointly. The learning rate is read from `config["learning_rate"]` or `config["learningrate"]` (the YAML key is `learningrate` with no underscore).

**`train_step(states)`**

Takes a batch of flat state tensors `[B, state_dim]` and runs one optimization step.

1. Forward pass: compute `policy_probs = pi_theta(states)` and `values = V_phi(states)`.
2. Expand values to `[B, num_actions]` for flow computation.
3. Compute state flow, edge flow, and detached reference flow.
4. Compute `L_flow` (flow consistency MSE) and `L_rank` (ranking margin loss).
5. Total loss: `w_flow * L_flow + w_rank * L_rank`.
6. Backward + optimizer step.

Returns a dict with keys `"loss"`, `"flow_loss"`, `"rank_loss"`.

FlowAblationTrainer is the ablation baseline: it shows what performance is achievable with flow-based objectives alone, without MCTS search.

---

### `trainer_causal_mcts.py` -- CausalMCTSTrainer

`CausalMCTSTrainer` is the full system trainer. It jointly optimizes `pi_theta`, `V_phi`, and the transition model `f_theta` using rewards collected by MCTS search.

**Constructor parameters:**

| Parameter | Type | Description |
|---|---|---|
| `policy` | `PolicyNetwork` | Policy to train |
| `value` | `ValueNetwork` | Value network to train |
| `transition` | `TransitionModel` | Transition model to train |
| `mcts` | `MCTS` | Planner used for reward collection |
| `config` | `dict` | Merged YAML config dict |

The optimizer is Adam over the combined parameter set of all three trainable modules.

**`train_step(states)`**

Takes a list of `ESCState` objects and runs one optimization step.

1. **Reward collection (no grad):** set policy and value to eval mode. For each state, call `mcts.search(state)` to get the best action, then `env.step(state, action)` to get the reward. This produces a list of scalar rewards, one per state in the batch.
2. **Forward pass (train mode):** set policy and value back to train mode. Stack states into a `[B, state_dim]` tensor and run forward passes.
3. **Value regression loss:** `L_value = MSE(V_phi(states), R_MCTS)`. This supervises the value network with the MCTS-collected return signal.
4. **Flow losses:** same computation as `FlowAblationTrainer`.
5. **Total loss:** `L_value + w_flow * L_flow + w_rank * L_rank`.
6. Backward + optimizer step.

Returns a dict with keys `"loss"`, `"value_loss"`, `"flow_loss"`, `"rank_loss"`.

The MCTS search and env step in step 1 are wrapped in `torch.no_grad()` to avoid accumulating gradients through the search tree, which would be both incorrect and prohibitively expensive.

---

### `train_data.py` -- PyTorch Datasets

Both dataset classes read from `.pt` files produced by `scripts/preprocess_datasets.py`. Neither class imports from `data/dataset_sources.py` or the Hugging Face / ConvoKit APIs.

**`ESCStateTensorDataset`**

Reads the `"state_tensors"` key from a `.pt` file (a pre-stacked `[N, state_dim]` tensor). If the key is absent, it falls back to reconstructing state tensors from the bundle list. This dataset is used by `FlowAblationTrainer`, which needs flat tensors and can use a standard `DataLoader` with batching.

**`ESCStateBundleDataset`**

Reads the `"bundles"` list from a `.pt` file. Each element is a dict that can be reconstructed into a full `ESCState` via `esc_state_from_bundle()`. The `get_state(index)` method performs this reconstruction on demand.

This dataset is used by `CausalMCTSTrainer`, which needs full `ESCState` objects (including the causal graph) to run MCTS search. It cannot use the flat tensor representation because the causal graph cannot be recovered from the flattened vector alone.

---

### `utils.py` -- Config Loading

**`load_env_config(path)`** reads a flat `key: value` YAML file without importing PyYAML. It splits each non-comment line on the first colon, strips whitespace, and tries to parse the value as int, then float, then string.

**`load_merged_config(root)`** loads `config/env.yaml` first, then `config/train_shared.yaml`, and merges them with `train_shared.yaml` keys taking precedence. The merged dict is what all training scripts and the interactive CLI consume.

**`set_seed(stub)`** is an unfilled stub. Global seed setting is handled by `utils/seed.py::set_global_seed()`, which is the authoritative implementation.

---

### `data_utils.py`

Lightweight utilities for the training data pipeline (currently minimal; reserved for future augmentation utilities such as on-the-fly state perturbation for data augmentation).

---

## Config keys consumed by training

All keys are read from the merged `load_merged_config()` dict.

| YAML key | Python alias | Default | Consumed by |
|---|---|---|---|
| `learningrate` | `learning_rate` | `1e-4` | Both trainers |
| `batchsize` | `batch_size` | `4` | Both scripts |
| `maxsteps` | `max_steps` | `10` | Both scripts |
| `flowlossweight` | `flow_loss_weight` | `1.0` | Both trainers |
| `rankinglossweight` | `ranking_loss_weight` | `1.0` | Both trainers |
| `gammamargin` | `gamma_margin` | `1.0` | Both trainers |
| `num_simulations` | -- | `10` | CausalMCTSTrainer (via MCTS) |
| `max_horizon` | -- | `10` | ESCEnv |
| `cpuct` | `c_puct` | `1.0` | MCTS |
| `device` | -- | `"cuda"` | Both scripts (device placement) |
| `seed` | -- | `42` | Interactive CLI |

Both trainers accept both the underscore and no-underscore variants of multi-word keys (e.g. `learning_rate` or `learningrate`) to be robust to YAML style differences.

---

## Ablation experiments

The trainer pair enables direct ablation of the MCTS component:

| Configuration | Trainer | What is being tested |
|---|---|---|
| Flow-matching only | `FlowAblationTrainer` | Flow objectives without search |
| MCTS + Flow | `CausalMCTSTrainer` | Full system |
| MCTS only (set flow weights to 0) | `CausalMCTSTrainer` | MCTS without flow regularizer |
| Random dynamics (use `RandomTransitionModel`) | Either | MDP without learned transition |

Setting `flowlossweight: 0.0` and `rankinglossweight: 0.0` in `config/train_shared.yaml` disables the flow-matching terms from either trainer without code changes.
