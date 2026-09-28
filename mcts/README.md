# mcts -- Monte Carlo Tree Search Planner

This package implements the MCTS planning loop for action selection in the ESC MDP. It provides the `MCTS` class that runs the full four-phase search algorithm, the `TreeNode` class that represents a node in the search tree with PUCT statistics, and inline `PolicyNetwork` and `ValueNetwork` definitions used for prior computation and leaf evaluation.

---

## Overview

At each conversation turn, the planner calls `MCTS.search(root_state)` and returns the action `a_t` with the highest visit count after all simulations complete. The key design choices relative to a naive implementation are:

- **Value network leaf evaluation** replaces random rollout, reducing variance dramatically.
- **Phase-filtered candidate actions** keep the branching factor at 16 (4 strategies x 4 causes) rather than 32.
- **Visit-count selection** rather than argmax of raw priors gives a final action that reflects accumulated search evidence.
- **Single-agent backpropagation** does not negate values between tree levels, because ESC is a cooperative single-agent problem, not a two-player zero-sum game.

---

## Files

### `node.py` -- TreeNode

`TreeNode` represents one node in the MCTS search tree. It stores the ESC state at that node and maintains the four standard PUCT statistics:

| Field | Symbol | Description |
|---|---|---|
| `N` | N(s, a) | Visit count -- number of times this node was selected |
| `W` | W(s, a) | Total accumulated value across all visits |
| `Q` | Q(s, a) | Mean value W / N |
| `P` | P(a|s) | Prior probability from the policy network |

Child nodes are stored in a dict `{ESCAction: TreeNode}`. This allows O(1) lookup by action, which is important because `ESCAction` implements `__hash__` and `__eq__` based on `(strategy_id, cause_index)`.

**PUCT selection formula** (Rosin 2011, AlphaGo Zero variant):

```
PUCT(s, a) = Q(s, a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s, a))
```

The exploration term U is large when:
- The prior `P(a|s)` is high (policy believes this action is good)
- The parent has been visited many times (`sqrt(N(s))` is large)
- The child has been visited few times (`1 + N(s, a)` is small)

`select_child(c_puct)` iterates over all children and returns the one with the highest PUCT score. It uses `max(self.N, 1)` in the sqrt to avoid a zero exploration term at the root.

**`expand(actions, priors)`** creates one child `TreeNode` for each candidate action, assigning the corresponding prior probability. Child states are initially clones of the parent state; the full MCTS loop replaces them with real next states from `env.step()` immediately after expansion.

**`set_child_state(action, state)`** overwrites a child's state after environment simulation. This is called by `MCTS._expand_node()` to replace the placeholder clone with the actual transition output.

**`backpropagate(value)`** walks the parent chain from the current node to the root, calling `update(value)` at each node. Because ESC is single-agent, the value is not negated as it propagates upward. This is the correct behaviour: a good outcome at depth k is genuinely good from the root's perspective.

**`visit_count_policy(temperature)`** derives a policy distribution from child visit counts:

```
pi_target(a) proportional to N(s, a)^{1/temperature}
```

At temperature 1.0 the distribution is proportional to raw visit counts. At temperature approaching 0 it becomes a one-hot on the most-visited action. This is used during training to extract policy targets for imitation learning from the MCTS-improved policy.

**`most_visited_child()`** is used at the end of `search()` to return the final recommended action. Visit count is more robust than Q for final selection because Q can be volatile when N is small.

---

### `mcts.py` -- MCTS

The `MCTS` class orchestrates the full four-phase search loop and houses the inline `PolicyNetwork` and `ValueNetwork`.

#### Architecture: PolicyNetwork and ValueNetwork

Both networks share an identical two-layer trunk:

```
SharedTrunk:
  Linear(state_dim, hidden_dim) -> LayerNorm -> ReLU
  Linear(hidden_dim, hidden_dim) -> LayerNorm -> ReLU
```

The policy head adds `Linear(hidden_dim, num_actions) -> Softmax` to produce a probability distribution over the candidate action set.

The value head adds `Linear(hidden_dim, 1) -> Tanh` to produce a scalar estimate in [-1, 1]. The tanh bound matches the normalized reward range of the ESC MDP, where individual components are designed to lie in approximately [-1, 1].

These inline definitions in `mcts.py` are used during search (eval mode). The standalone `models/policy.py` and `models/value.py` are used during training (train mode). The two are architecturally identical; the separation allows training code to import from `models/` without pulling in the MCTS planner.

#### Four-phase simulation loop

Each call to `_simulate(root)` runs one complete simulation:

**Phase 1 -- Selection**

Starting from the root, repeatedly call `select_child(c_puct)` to descend through expanded nodes until a leaf (unexpanded node) is reached. Because the root has `N = 1` set before the loop, the exploration term is nonzero from the first simulation.

**Phase 2 -- Expansion**

At the leaf node, generate candidate actions with `generate_candidate_actions(state)`, get priors from `_get_priors()`, and call `_expand_node()`. Expansion calls `env.step(node.state, action)` for each candidate action to get the real next state. This is where the transition model is exercised: child states are not guesses but actual model outputs.

**Phase 3 -- Evaluation**

Rather than rolling out to the horizon, the value network evaluates the leaf state: `V_theta(s_{leaf})`. This single forward pass replaces potentially 15+ random steps of rollout and reduces variance by an order of magnitude.

**Phase 4 -- Backpropagation**

Call `node.backpropagate(value)` to propagate the leaf evaluation back to the root, incrementing N and updating W and Q at every ancestor.

#### Prior computation and size mismatch handling

`_get_priors(state, candidates)` runs the policy network and returns a probability distribution over the candidate list. Because the policy network is constructed with a fixed output size at init time, but the candidate list size varies by phase (16 or 32), a size mismatch can occur. The handler:

1. If sizes match: return directly.
2. If the network is larger: truncate to the first `num_actions` logits.
3. If the network is smaller: pad with the mean logit to assign a uniform prior to unseen actions.

All three paths re-normalize via softmax so the output is always a valid probability distribution. The correct long-term fix is to train the policy head with the same filtered action count; the adapter is a safe fallback during the scaffolding phase.

#### `search(root_state)` and `search_with_policy(root_state)`

`search()` returns the single best `ESCAction` based on visit count. It is the primary interface used by the trainer and interactive CLI.

`search_with_policy()` additionally returns the full action list and the visit-count probability distribution over it. This is used by `CausalMCTSTrainer` to extract policy imitation targets during training.

---

## MCTS hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `num_simulations` | 50 | Search budget per step. Higher values improve action quality at the cost of wall time. |
| `c_puct` | 1.0 | Exploration-exploitation tradeoff. Higher values encourage more exploration of unvisited branches. |
| `hidden_dim` | 256 | Width of the shared trunk in policy and value networks. |

These are configurable via `config/train_shared.yaml` (keys `num_simulations`, `cpuct`) and override-able through the `config` dict passed to `MCTS.__init__()`.

---

## Reproducibility

MCTS relies on the `RandomTransitionModel`'s internal `torch.Generator` for stochastic transitions during the scaffold phase. Passing a fixed `seed` to `RandomTransitionModel` and calling `utils.seed.set_global_seed()` before constructing the planner guarantees that the same root state will produce the same action across runs. This is verified in `tests/test_pipeline.py::TestMCTS::test_search_reproducible_with_same_seed`.

---

## Relationship to other components

```
MCTS
  reads: ESCState (from esc/)
  generates: ESCAction candidates (from esc/action.py)
  uses: ESCEnv.step() for expansion (from esc/env.py)
  uses: PolicyNetwork for priors
  uses: ValueNetwork for leaf evaluation
  produces: ESCAction recommendation
  produces: visit-count policy target (for CausalMCTSTrainer)
```
