# flow -- Flow-Matching Objectives (FlowMCTS-ablation)

**Not a reproduction of the published AFlow system (Zou et al.).** This package implements an in-house flow-matching regularizer *inspired by* AFlow/GFlowNet ideas -- state flow and edge flow computations, plus a flow-consistency loss and a ranking loss on value estimates. These are used by both trainers (`FlowAblationTrainer` and `CausalMCTSTrainer`); `FlowAblationTrainer` alone (no MCTS, no transition model) is the "FlowMCTS-ablation" baseline referenced elsewhere in this repo. See [results/README.md](../results/README.md) for why this rename happened.

---

## Conceptual background

AFlow adapts the GFlowNet flow-matching objective to the sequential decision-making setting. The core idea is that the value function `V(s)` and the policy `pi(a|s)` should be consistent with each other in a flow sense: the "amount of flow" assigned to a state by the value function should match the flow distributed across its outgoing edges by the policy.

In the ESC setting this is applied as a regularizer rather than as the primary training signal. The flow loss encourages the policy and value to remain mutually consistent, which improves training stability especially in the early stages when MCTS-collected rewards are noisy.

---

## Files

### `flow.py` -- Flow Computations

**`compute_state_flow(Q_b, V)`**

Computes the state flow tensor as the element-wise difference between the behavior policy Q-values and the value estimates:

```
F(s) = Q_b(s) - V_phi(s)
```

Both `Q_b` and `V` must have the same shape. The function raises `ValueError` if they differ.

In practice, `Q_b` is the policy probability matrix `[B, num_actions]` and `V` is the value estimate `[B, num_actions]` (expanded from `[B, 1]` to match). The state flow represents the discrepancy between what the policy assigns to each action and what the value network predicts for the resulting state.

**`compute_edge_flow(F_s, policy_probs)`**

Computes the edge flow tensor, which distributes the state flow across outgoing actions according to the policy:

```
F(s, a) = F(s) * pi_theta(a | s)
```

The function handles both the case where `F_s` and `policy_probs` have the same number of dimensions (element-wise product) and the case where `F_s` has one fewer dimension (broadcasting via `unsqueeze(-1)`).

---

### `losses.py` -- Loss Functions

**`flow_consistency_loss(flow_edges, flow_from_policy)`**

Mean squared error between the computed edge flow and a detached reference flow:

```
L_flow = MSE(F(s, a), F_detached(s, a))
```

The reference `flow_from_policy` is computed from detached policy probabilities and a detached state flow, so the gradient flows only through `flow_edges`. This asymmetric treatment encourages the policy to match the value-consistent flow target rather than both networks chasing each other simultaneously.

Both tensors must have the same shape; a `ValueError` is raised if they differ.

**`ranking_loss(v_winner, v_loser, margin)`**

A margin-based ranking loss that encourages `v_winner` to score higher than `v_loser` by at least `margin`:

```
L_rank = max(0, margin - (v_winner - v_loser))
```

In the training loops, `v_winner` is the mean value of the current batch and `v_loser` is a zero tensor, so the loss penalizes batches where the mean predicted value is below the margin. This acts as a weak value regularizer that prevents collapse to near-zero predictions.

The `margin` is configurable via `config["gamma_margin"]` (default 1.0).

---

## How the losses connect to training

In `FlowAblationTrainer.train_step()`:

```python
F_s     = compute_state_flow(Q_b=policy_probs, V=values.expand(-1, num_actions))
F_edges = compute_edge_flow(F_s, policy_probs.detach())
F_ref   = compute_edge_flow(F_s.detach(), policy_probs.detach()).detach()

L_flow = flow_consistency_loss(F_edges, F_ref)
L_rank = ranking_loss(v_winner=values.mean(), v_loser=zeros, margin=margin)
loss   = w_flow * L_flow + w_rank * L_rank
```

In `CausalMCTSTrainer.train_step()` the same flow terms are included, plus a value regression loss against MCTS-collected rewards:

```python
L_value = MSE(V_theta(s), R_MCTS)
loss    = L_value + w_flow * L_flow + w_rank * L_rank
```

The weights `w_flow` and `w_rank` are loaded from `config["flow_loss_weight"]` and `config["ranking_loss_weight"]` (both default 1.0 in `config/train_shared.yaml`).

---

## `flow/__init__.py` exports

The package `__init__.py` re-exports all four symbols so that trainers can import from a single location:

```python
from flow import compute_edge_flow, compute_state_flow, flow_consistency_loss, ranking_loss
```

---

## Ablation

To ablate the flow-matching objectives and train with only the MCTS value regression loss, set both weights to zero in `config/train_shared.yaml`:

```yaml
flowlossweight: 0.0
rankinglossweight: 0.0
```

This isolates the contribution of the flow consistency regularizer to training stability and final policy quality.
