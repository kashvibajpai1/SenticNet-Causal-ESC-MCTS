# config -- YAML Configuration Files

This directory contains the two flat YAML configuration files that control all hyperparameters for training, search, and environment setup. They are loaded and merged by `train/utils.py::load_merged_config()`, with `train_shared.yaml` keys taking precedence over `env.yaml` keys.

---

## `env.yaml` -- Environment and Backbone Config

Controls the hardware target, reproducibility seed, and which backbone model to use.

| Key | Default | Type | Description |
|---|---|---|---|
| `device` | `"cuda"` | str | Torch device for model computation. Set to `"cpu"` on machines without a GPU. |
| `seed` | `42` | int | Global RNG seed passed to `utils.seed.set_global_seed()`. |
| `backbone_model_name` | `"Qwen/Qwen2.5-0.5B-Instruct"` | str | Hugging Face model ID for the Qwen backbone. Passed to `QwenBackbone.__init__()`. Downsized from the paper's originally stated "Qwen-9B" -- this project has no GPU access. See [results/README.md](../results/README.md). |

**When to change `env.yaml`:**

- Different GPU machines: update `device` to the appropriate CUDA device string (`"cuda:0"`, `"cuda:1"`, etc.).
- Reproducibility experiments: change `seed` to a different integer and re-run preprocessing and training.
- Backbone ablation: change `backbone_model_name` to a different HF model ID when comparing backbone sizes.

---

## `train_shared.yaml` -- Training and Search Hyperparameters

Controls all training loop, MCTS search, and loss function parameters. These are shared between the FlowMCTS-ablation trainer and the Causal MCTS trainer.

These are now "reduced-but-real" values (not the old `maxsteps: 10` smoke test, and not an unverified paper-scale claim) -- chosen to complete a genuine CPU-only run in bounded wall-clock time. See [results/README.md](../results/README.md) for the full rationale and the results this config actually produced.

| Key | Default | Type | Used by | Description |
|---|---|---|---|---|
| `learningrate` | `1e-4` | float | Both trainers | Adam learning rate. |
| `batchsize` | `16` | int | Both scripts | Number of states per training step. |
| `maxsteps` | `300` | int | Both scripts | Total number of gradient steps. |
| `flowlossweight` | `1.0` | float | Both trainers | Weight `w_flow` on the flow consistency loss term. Set to `0.0` to ablate the flow-matching term. |
| `rankinglossweight` | `1.0` | float | Both trainers | Weight `w_rank` on the ranking loss term. Set to `0.0` to ablate the ranking objective. |
| `gammamargin` | `1.0` | float | Both trainers | Margin `m` in the ranking loss `max(0, m - (v_winner - v_loser))`. |
| `num_simulations` | `15` | int | MCTS | Search budget per step. |
| `max_horizon` | `12` | int | ESCEnv | Maximum conversation turns per episode. |
| `cpuct` | `1.5` | float | MCTS | PUCT exploration constant. Higher values encourage broader search. |

---

## Config loading

Both YAML files use a flat `key: value` format with no nesting or special YAML features. They are parsed by `train/utils.py::load_env_config()`, which does not require PyYAML: it splits each line on the first colon and auto-parses the value as int, float, or string.

Comments in the YAML files (lines starting with `#` or inline `# text`) are stripped before parsing.

The merged config dict is passed directly to trainers, MCTS, and the environment. All consumer code accepts both the underscore form (`learning_rate`) and the no-underscore form (`learningrate`) for multi-word keys, since the YAML files use the no-underscore form.

---

## Scaling further

The values above already produced the results in `results/`. For a larger run (more seeds, more steps, larger eval set), raise `maxsteps`/`num_simulations`/`batchsize` further -- but note wall-clock scales accordingly, especially the eval and latency-baseline phases which call the real Qwen backbone. See `scripts/run_full_experiment.sh` for the bounded orchestration this project actually ran, including its wall-clock caps.
