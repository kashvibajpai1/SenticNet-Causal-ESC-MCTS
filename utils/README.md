# utils -- Reproducibility and Logging Utilities

This package provides two utilities that are used throughout the pipeline: a seed management module for reproducible experiments, and an episode logger for structured experiment tracking.

---

## Files

### `seed.py` -- Seed Management and Experiment Configuration

**`set_global_seed(seed)`**

Sets all sources of randomness used by the pipeline:

| Source | Call |
|---|---|
| Python `random` module | `random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch all CUDA devices | `torch.cuda.manual_seed_all(seed)` |
| cuDNN determinism | `torch.backends.cudnn.deterministic = True` |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = False` |
| `PYTHONHASHSEED` env var | `os.environ["PYTHONHASHSEED"] = str(seed)` |

The cuDNN flags make convolution operations deterministic at the cost of a small performance penalty. For MLP-heavy workloads like this one the penalty is negligible.

`PYTHONHASHSEED` affects the ordering of dict and set iteration in Python 3.3+. Setting it as an environment variable here does not retroactively affect the current interpreter session (it must be set before the interpreter starts), but it is recorded for logging purposes and can be used to reproduce the exact run by prefixing the command with `PYTHONHASHSEED=42 python ...`.

Raises `ValueError` for negative seeds.

**`ExperimentConfig`**

A dataclass that bundles all hyperparameters that affect experimental results into a single object that can be saved to JSON and reloaded:

| Field | Default | Description |
|---|---|---|
| `seed` | `42` | Global RNG seed |
| `num_simulations` | `50` | MCTS budget per step |
| `max_horizon` | `20` | Episode length |
| `c_puct` | `1.0` | PUCT exploration constant |
| `hidden_dim` | `256` | Neural network width |
| `alpha` | `1.0` | Weight for R_cause |
| `beta` | `1.0` | Weight for R_emotion |
| `gamma` | `1.0` | Weight for R_phase |
| `device` | `"cpu"` | Torch device |
| `notes` | `""` | Free-text experiment notes |

`apply()` calls `set_global_seed(self.seed)` to activate the configuration.

`save(path)` serializes the config to a JSON file. The parent directory is created if needed.

`load(path)` reconstructs an `ExperimentConfig` from a JSON file.

`log()` prints the config to stdout in a formatted table.

**Usage pattern:**

```python
from utils.seed import ExperimentConfig

cfg = ExperimentConfig(seed=42, num_simulations=50, notes="ablation: no flow-matching term")
cfg.apply()   # sets all seeds
cfg.log()     # prints to stdout
cfg.save("experiments/run_001_config.json")

# Later, to reproduce exactly:
cfg = ExperimentConfig.load("experiments/run_001_config.json")
cfg.apply()
```

---

### `logging.py` -- EpisodeLogger

`EpisodeLogger` records structured data for each turn and episode of an ESC session. It is designed for post-hoc analysis and comparison across experiment runs.

**Data model:**

`StepRecord` captures one environment step:

| Field | Type | Description |
|---|---|---|
| `turn` | int | Turn index |
| `strategy_id` | int | Strategy used (0-7) |
| `strategy_name` | str | Human-readable strategy name |
| `cause_index` | int | Cause targeted |
| `reward` | float | Scalar reward received |
| `R_cause` | float | Cause resolution component |
| `R_emotion` | float | Emotional progress component |
| `R_phase` | float | Phase advancement component |
| `phase` | int | Phase index after the step |
| `resolution` | list[float] | Per-cause resolution probabilities after the step |

`EpisodeRecord` accumulates steps for a full episode:

| Field | Type | Description |
|---|---|---|
| `episode_id` | int | Unique identifier |
| `start_time` | float | Unix timestamp at episode start |
| `end_time` | float | Unix timestamp at episode end |
| `total_reward` | float | Sum of step rewards |
| `num_turns` | int | Number of steps taken |
| `final_phase` | int | Phase index at episode end |
| `final_resolution` | list[float] | Resolution probabilities at episode end |
| `steps` | list[StepRecord] | All step records |

**Lifecycle:**

```python
logger = EpisodeLogger()
logger.start_episode(episode_id=0)

# ... inside the episode loop:
logger.log_step(action=action, reward=reward, info=info)

logger.end_episode(final_state=state)
```

Calling `start_episode()` while an episode is active raises `RuntimeError`. Calling `log_step()` or `end_episode()` without a matching `start_episode()` raises `RuntimeError`. These invariants prevent silent data corruption in training loops.

**`episode_summary(episode_idx)`** returns a dict with:

| Key | Description |
|---|---|
| `episode_id` | Episode identifier |
| `total_reward` | Sum of all step rewards |
| `num_turns` | Number of steps |
| `mean_reward_per_turn` | `total_reward / num_turns` |
| `duration_s` | Wall-clock seconds |
| `final_phase` | Phase at episode end |
| `final_resolution` | Per-cause resolution at episode end |
| `strategy_counts` | Dict mapping strategy name to number of times it was used |

**Persistence:**

`save(path)` writes all episodes to a JSON file using `dataclasses.asdict()`. The resulting JSON is a list of episode records, each containing a list of step records. All numeric fields are preserved with full float precision.

`load(path)` reconstructs a `EpisodeLogger` from a JSON file. The loaded logger is read-only in the sense that it has no active episode; it can be iterated and summarized but not continued.

`__len__` returns the number of completed episodes.

**Usage in training scripts:**

```python
logger = EpisodeLogger()
for episode_id in range(num_episodes):
    logger.start_episode(episode_id)
    state = env.reset()
    done = False
    while not done:
        action = mcts.search(state)
        state, reward, done, info = env.step(state, action)
        logger.log_step(action=action, reward=reward, info=info)
    logger.end_episode(final_state=state)

logger.save(f"experiments/run_{run_id}_log.json")

# Print summary of last episode
print(logger.episode_summary(-1))
# Print all summaries for plotting
summaries = logger.all_summaries()
```
