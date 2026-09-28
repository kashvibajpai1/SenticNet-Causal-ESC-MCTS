# tests -- Test Suite

This directory contains the unit and integration test suite for the ESC MCTS pipeline. Tests are organized into classes that map directly to the module hierarchy, with each class verifying correctness at the behavioral level rather than merely checking tensor shapes.

---

## Running tests

```bash
# Run all unit tests (no network, no HF/ConvoKit downloads)
pytest

# Run a specific test class
pytest tests/test_pipeline.py::TestMCTS -v

# Run a specific test
pytest tests/test_pipeline.py::TestReward::test_r_cause_delta_not_cumulative -v

# Run integration tests (requires HF/ConvoKit access and disk space)
pytest -m integration

# Run with coverage
pytest --cov=. --cov-report=html
```

The `pytest.ini` in the repository root marks `integration` tests so they are excluded from the default run. This means `pytest` with no arguments is safe to run in CI and on any machine without HF/ConvoKit credentials.

---

## `test_imports.py`

Verifies that all top-level packages import without error:

- `esc`, `models`, `mcts`, `flow`, `train`, `inference`, `data`
- `train.train_data` (verifies the dataset classes are importable without an artifacts file)

These eight tests are the fastest possible regression check. If any of them fails, a dependency or `__init__.py` is broken.

---

## `test_pipeline.py` -- Core behavioral tests

This file contains 11 test classes with roughly 80 individual tests.

### TestCausalGraph (11 tests)

Verifies the causal graph data structure:

- Adding causes and checking the count
- Adding causes with the wrong embedding dimension raises `ValueError`
- Adding valid and self-loop edges (self-loop raises `ValueError`)
- Out-of-range edge indices raise `IndexError`
- `update_resolution()` stores correct values and clamps out-of-range inputs to [0, 1]
- Mismatched resolution tensor length raises `ValueError`
- Initial resolution is zero
- `cause_embedding_matrix()` has the correct shape
- `CausalGraph.placeholder()` creates the correct anonymous graph
- `neighbours()` returns the correct adjacent nodes

### TestESCState (9 tests)

Verifies the state representation and construction:

- `from_dialogue()` produces tensors with the correct shapes
- `phase_embedding` from `from_dialogue()` sums to 1.0
- `to_tensor()` has shape `[1283]`
- `get_state_dim()` returns the correct value (768 + 384 + 128 + 3 = 1283)
- `to_tensor()` pooling is numerically correct (mean of history rows, correct concatenation order)
- `current_phase` returns the argmax of the phase embedding
- `clone()` produces an independent copy (mutation of clone does not affect original)
- `clone()` deep-copies the causal graph independently
- Invalid tensor shapes raise `ValueError` at construction time
- Encoder hook is called and its output is used correctly

### TestESCAction (8 tests)

Verifies action generation and embedding:

- Phase 0 only produces strategies {0,1,2,3}
- Phase 1 only produces strategies {2,3,4,7}
- Phase 2 only produces strategies {4,5,6,7}
- `filter_by_phase=False` returns all 32 actions
- Candidate list is always non-empty for any phase
- `ESCAction` hash and equality are consistent and work as dict keys
- `embed_action()` produces a tensor of the correct shape
- `embed_action()` sets the correct one-hot slots
- `ActionEmbedder` produces the correct output dimension

### TestReward (10 tests)

This is the most semantically important test class. It verifies the signs and directions of reward components, not just whether they run:

- `R_cause` is zero when resolution does not change
- `R_cause` is positive when resolution increases
- `R_cause` is negative when resolution decreases
- **Critically:** `R_cause` must be zero when resolution is already high and does not change. This tests the delta property -- the reward must not be a cumulative sum.
- `R_emotion` is positive when moving closer to target
- `R_emotion` is negative when moving further from target
- `R_emotion` is zero when emotion is stationary
- `R_phase` is 1.0 when the phase advances
- `R_phase` is 0.0 when the phase stays the same or regresses
- `reward_components()` returns all three required keys
- Doubling `alpha` doubles the cause-resolution contribution exactly
- Individual components lie in their expected ranges

### TestTransitionModels (4 tests)

- `RandomTransitionModel.forward()` returns tensors of correct shapes and non-identical emotion
- `LinearTransitionModel.forward()` returns tensors of correct shapes
- `LinearTransitionModel` emotion output is bounded in (-1, 1) (tanh)
- `LinearTransitionModel` resolution delta is bounded in (-1, 1) (tanh)

### TestESCEnv (12 tests)

Verifies environment dynamics:

- `reset()` returns a valid state with correct shapes and a unit-sum phase distribution
- `step()` produces a next state with different emotion (transition model adds noise)
- `step()` reward is a finite float
- `step()` increments `turn_index`
- History window slides correctly after `step()` (old row 0 becomes new row 0)
- `step()` does not mutate the original state
- `done` is True exactly at the horizon
- `done` is False before the horizon
- `info` dict contains all required keys
- Phase embedding after `step()` is a valid probability distribution
- Resolution is clamped to [0, 1] after `step()`
- Different reward weights produce different total rewards

### TestTreeNode (14 tests)

- Initial statistics are all zero and the node is a leaf at depth 0
- `expand()` creates the correct number of children and marks the node as non-leaf
- Double `expand()` raises `RuntimeError`
- `expand()` with mismatched priors length raises `AssertionError`
- `update()` increments N and updates W and Q correctly over multiple calls
- `backpropagate()` updates all ancestors including root
- `select_child()` prefers the highest-prior action when Q is equal for all children
- `select_child()` prefers high Q over low prior when exploration constant is small
- `select_child()` raises `AssertionError` when the node has no children
- `set_child_state()` replaces the child's state correctly
- `set_child_state()` raises `KeyError` for an unknown action
- `visit_count_policy()` returns a distribution that sums to 1.0
- `visit_count_policy()` assigns the highest probability to the most-visited child
- `most_visited_child()` returns the correct child
- `depth` property is correct at root, child, and grandchild level

### TestMCTS (9 tests)

- `search()` returns an `ESCAction`
- The returned action is in the candidate list for the root state
- The same seed yields the same action (reproducibility)
- Simulations increment child visit counts
- Q-values are non-zero after at least one simulation
- Policy network output shape is correct
- Policy network output sums to 1.0
- Value network output is in [-1, 1]
- `search_with_policy()` returns correct types and a unit-sum policy target
- More simulations produce a different visit distribution than fewer simulations

### TestReproducibility (5 tests)

- Same seed, same random tensor
- Different seeds, different tensors
- `ExperimentConfig.save()` and `ExperimentConfig.load()` round-trip without data loss
- `ExperimentConfig.apply()` sets the seed so two consecutive calls produce the same tensors
- Negative seed raises `ValueError`

### TestEpisodeLogger (9 tests)

- A full episode records the correct number of turns
- Total reward matches the sum of step rewards
- Starting two episodes without ending the first raises `RuntimeError`
- Logging a step outside an episode raises `RuntimeError`
- Ending without starting raises `RuntimeError`
- Save and load round-trip preserves total reward
- `episode_summary()` contains all required keys

### TestEndToEnd (5 tests)

These tests exercise the full pipeline as a single coherent system:

- Single step: state construction, candidate generation, env step, MCTS search, reward components
- Full episode: runs to the horizon and terminates correctly
- Full episode with logger: EpisodeLogger captures all turns
- Reward is never NaN or Inf across a full episode
- Causal graph resolution increases on average with a positive-bias transition model

---

## `test_data_pipeline.py`

Data layer tests (skipped if artifacts are absent). Tests JSONL read/write, state bundle serialization round-trips, and preprocessing output validation.

---

## Testing philosophy

The test suite is designed around three principles:

1. **Behavioral correctness over shape checking.** Checking that `R_cause` returns a positive value when resolution increases is more valuable than checking that it returns a float. Shape tests are included where shape mismatches are the most common failure mode, but they are not sufficient on their own.

2. **Seeded determinism.** Tests that involve randomness use explicit seeds so failures are reproducible. The `mcts_planner` fixture sets `seed=42` before constructing the planner.

3. **Isolation by fixture.** Each test class builds its own fixtures (`graph`, `base_state`, `env`, `mcts_planner`) rather than sharing global state. This prevents test ordering from affecting outcomes.
