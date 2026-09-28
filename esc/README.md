# esc -- Emotional Support Conversation MDP

This package defines the core Markov Decision Process for Emotional Support Conversation (ESC). It contains the state representation, causal graph, action space, transition environment, and reward function. Everything in this package is independent of dataset loading and training infrastructure: it is the mathematical and algorithmic heart of the system.

---

## Overview

An ESC episode is modelled as a finite-horizon MDP:

```
M = (S, A, P, R, T)
```

- **S**: continuous state space combining dialogue history, causal structure, emotion, and phase
- **A**: discrete action space of strategy-cause pairs, phase-filtered at each step
- **P**: transition dynamics implemented by a pluggable `TransitionModel`
- **R**: decomposed scalar reward over cause resolution, emotional progress, and phase advancement
- **T**: maximum horizon (default 20 turns)

The supporter agent selects an action `a_t = (sigma, c_i)` at each turn, where `sigma` is a support strategy and `c_i` is the causal node being targeted. The environment steps forward, producing a new state and a reward signal.

---

## Files

### `state.py` -- ESCState

`ESCState` is the unified state representation for the ESC MDP. It is a dataclass holding four components:

```
s_t = phi(H_t, G_t) = concat(H_bar_t, C_bar_t, e_t, p_t) in R^d
```

| Component | Symbol | Shape | Description |
|---|---|---|---|
| History embeddings | H_t | [K, D_H] | Last K dialogue turn encodings from the backbone |
| Causal graph | G_t | -- | CausalGraph object tracking causes and resolution |
| Emotion vector | e_t | [D_E] | Dense representation of current emotional state |
| Phase embedding | p_t | [3] | Soft probability distribution over three ESC phases |

**Dimension constants** (class-level, excluded from `__init__`):

| Constant | Value | Meaning |
|---|---|---|
| `K_HISTORY_WINDOW` | 5 | Number of recent turns retained in the window |
| `D_H` | 768 | Qwen-9B hidden dimension |
| `D_C` | 384 | Cause embedding dimension |
| `D_E` | 128 | Emotion vector dimension |
| `D_P` | 3 | Phase dimension (exactly 3 ESC phases) |
| `N_C` | 4 | Maximum number of tracked causal nodes |

The total state dimension is `D_H + D_C + D_E + D_P = 768 + 384 + 128 + 3 = 1283`.

**`to_tensor()`** flattens the state to a single 1283-dimensional vector by mean-pooling the history window and the cause embedding matrix, then concatenating with the emotion vector and phase probabilities. This vector is the input consumed by the policy and value networks.

**`from_dialogue(turns, encoder=None, target_emotion=None)`** is the primary construction interface. When `encoder` is `None` it returns a structurally valid zero-filled state (used throughout testing and scaffolding). When an `EncoderFn` is provided it must return a dict with keys `"history"` (shape [K, D_H]), `"emotion"` (shape [D_E]), and `"causes"` (list of dicts with `"label"` and `"embedding"`). The encoder hook is the only seam where the Qwen backbone connects to the MDP.

**`clone()`** performs a deep copy of the entire state including the causal graph. This is called by MCTS every time a child node is created to ensure that mutations in one search branch never propagate to another.

**`current_phase`** is a property that returns `argmax(phase_embedding)` as an integer in {0, 1, 2}. It is used by the action generator to filter strategies and by the reward function to detect phase transitions.

---

### `causal_graph.py` -- CausalGraph

The causal graph `G_t = (V, E, rho_t)` is the component that gives this system its "causal" character. It tracks:

- **V**: a list of `CauseNode` objects, each with a natural-language label and a dense embedding in R^{D_C}
- **E**: directed edges represented as an adjacency dict `{src: [dst, ...]}`. An edge `i -> j` means cause i contributes to or exacerbates cause j.
- **rho_t in [0,1]^{N_C}**: per-cause resolution probabilities that change each step as the transition model predicts how much progress has been made toward resolving each identified cause.

**`CauseNode`** is a dataclass with fields `index`, `label`, `embedding`, and `resolution`. The resolution field is a float in [0, 1] and is updated by `CausalGraph.update_resolution()` after each environment step.

**Key methods on `CausalGraph`:**

- `add_cause(label, embedding, initial_resolution)`: appends a cause node. The embedding must have shape `(d_c,)` or a `ValueError` is raised.
- `add_edge(src, dst)`: adds a directed causal edge. Self-loops raise `ValueError`; out-of-range indices raise `IndexError`.
- `update_resolution(new_probs)`: overwrites all per-cause resolution values from a tensor of shape `[n_c]`. Values are clamped to [0, 1] node-by-node.
- `resolution_tensor()`: returns current resolution probabilities as a `[n_c]` float32 tensor. This tensor feeds directly into `_r_cause` in the reward function.
- `cause_embedding_matrix(max_causes, d_c)`: builds the `C_t` matrix of shape `[max_causes, d_c]` by padding or truncating node embeddings. This is called by `ESCState.to_tensor()` to produce the `C_bar_t` mean-pooled component.
- `placeholder(n_causes, d_c)`: class method that creates an all-zero graph with anonymous cause labels. Used by `ESCState.from_dialogue()` when no encoder is provided.

The graph is intentionally simple: it does not perform any graph neural network operations itself. GNN-style message passing over the causal structure is a natural future extension but is not implemented here.

---

### `action.py` -- ESCAction and ActionEmbedder

**`ESCAction`** represents a single support action as a strategy-cause pair `a_t = (sigma, c_i)`.

The 8 support strategies, indexed 0-7, are drawn from the ESC literature:

| Index | Strategy | Phase(s) |
|---|---|---|
| 0 | Question | Exploration |
| 1 | Restatement | Exploration |
| 2 | Reflection | Exploration, Comforting |
| 3 | Self-disclosure | Exploration, Comforting |
| 4 | Affirmation | Comforting, Action |
| 5 | Providing Suggestions | Action |
| 6 | Information | Action |
| 7 | Others | Comforting, Action |

**Phase filtering** via `PHASE_STRATEGY_MAP` is the primary mechanism for keeping the action space tractable. At phase 0 (Exploration) only strategies {0,1,2,3} are valid; at phase 1 (Comforting) only {2,3,4,7}; at phase 2 (Action Planning) only {4,5,6,7}. This reduces the maximum branching factor from 32 (8 strategies x 4 causes) to 16.

`ESCAction` implements `__hash__` and `__eq__` so it can be used as a dict key in the MCTS tree node's `children` dict.

**`generate_candidate_actions(state, filter_by_phase=True)`** produces the full candidate list for a given state. With phase filtering enabled it yields 4 strategies x N_C causes = 16 actions. With filtering disabled it yields 32.

**`ActionEmbedder`** is an `nn.Module` that produces dense action embeddings via learned `nn.Embedding` tables for strategy and cause separately. Strategy and cause embeddings are concatenated to form `psi(sigma, c_i) in R^{d_a}`. The default embedding dimensions are 32 each, for a 64-dimensional output.

**`embed_action(action, strategy_dim, cause_dim)`** is a lightweight one-hot alternative suitable for testing and logging without requiring an `nn.Module`.

---

### `reward.py` -- Decomposed Reward

The reward is decomposed into three semantically distinct components:

```
R(s_t, a_t) = alpha * R_cause + beta * R_emotion + gamma * R_phase
```

Each component is normalized to approximately [-1, 1] so that the weights `alpha`, `beta`, `gamma` carry comparable influence.

**`R_cause` -- Cause resolution progress**

```
R_cause = mean(rho_{t+1} - rho_t)
```

This is the mean per-cause resolution delta. It is positive when the agent helps resolve causes and negative when it causes regression. Crucially, it uses the delta, not the cumulative sum: an agent that has already resolved a cause to 0.9 gains zero reward for leaving it at 0.9. This prevents the reward signal from being dominated by early, easy gains.

**`R_emotion` -- Emotional progress toward target**

```
dist_t    = ||e_t    - e*||_2
dist_next = ||e_{t+1} - e*||_2
R_emotion = (dist_t - dist_next) / sqrt(D_E)
```

This measures movement toward the target emotion `e*`. It is positive when the agent moves the user's emotional state closer to the desired endpoint (calm, neutral). Dividing by `sqrt(D_E)` normalizes for the dimensionality of the emotion space.

**`R_phase` -- Phase advancement**

```
R_phase = 1.0 if current_phase(s_{t+1}) > current_phase(s_t) else 0.0
```

A binary reward for advancing the conversation from Exploration to Comforting, or from Comforting to Action Planning. Phase regression is not penalized here; it is implicitly penalized by the opportunity cost of not earning +1.

**`reward_components(state, next_state)`** returns all three components as a labeled dict. This is used by `ESCEnv.step()` to populate the `info` dict for logging.

---

### `env.py` -- ESCEnv

`ESCEnv` wraps the MDP transition dynamics into a standard `reset() / step()` interface.

**`reset(initial_turns, target_emotion)`** calls `ESCState.from_dialogue()` to produce an initial state `s_0`. Until the Qwen backbone is wired in, embeddings are zeros.

**`step(state, action)`** executes one MDP transition:

1. Calls `self._transition.forward(state, action)` to get a `TransitionOutput` containing `next_emotion`, `next_phase_logits`, and `delta_resolution`.
2. Clones the current state.
3. Slides the history window: drops the oldest row and appends a zero placeholder for the new turn.
4. Updates `emotion_vector` from `TransitionOutput.next_emotion`.
5. Updates `phase_embedding` via `softmax(next_phase_logits)`, ensuring it remains a valid probability distribution.
6. Applies `delta_resolution` to the causal graph, clamping each cause to [0, 1].
7. Computes the scalar reward via `compute_reward()`.
8. Checks the done condition: `turn_index >= max_horizon`.
9. Returns `(next_state, reward, done, info)`.

The `info` dict contains: `turn`, `strategy_used`, `cause_targeted`, `horizon_reached`, `phase`, `R_cause`, `R_emotion`, `R_phase`, `resolution`.

The transition model is pluggable. `RandomTransitionModel` is the default and is used for pipeline testing. `LinearTransitionModel` is the learned model used during training. Swapping them requires only passing a different instance to `ESCEnv.__init__()`.

---

## Three-phase ESC structure

The three conversation phases map to the natural progression of an emotional support session:

| Phase | Index | Goal | Valid strategies |
|---|---|---|---|
| Exploration | 0 | Understand the user's situation and feelings | Question, Restatement, Reflection, Self-disclosure |
| Comforting | 1 | Validate the user's emotions and reduce distress | Reflection, Self-disclosure, Affirmation, Others |
| Action Planning | 2 | Offer practical steps toward resolution | Affirmation, Providing Suggestions, Information, Others |

The soft phase embedding allows the model to represent uncertainty about which phase the conversation is in, rather than committing to a hard discrete label. This is important because real conversations frequently move between phases fluidly.

---

## Design principles

- **No dataset code**: nothing in this package imports `datasets`, `convokit`, or any corpus-specific code. The encoder hook in `ESCState.from_dialogue()` is the only connection point.
- **No training code**: this package does not depend on `train/`, `flow/`, or the script layer.
- **Deep copying**: `clone()` on both `ESCState` and `CausalGraph` ensures MCTS tree branches are fully independent.
- **Validation at construction**: `ESCState.__post_init__()` checks tensor shapes immediately, so shape mismatches are caught at the point of construction rather than during a forward pass.
