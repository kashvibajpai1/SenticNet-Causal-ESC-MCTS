"""Deep diagnostic Step 4: gradient-flow + per-loss-term breakdown for the
value head during causal_mcts training.

Re-runs training from a FRESH seed=42 init (not the trained checkpoint) for
100 steps, replicating train/trainer_causal_mcts.py's exact math (imported,
not reimplemented, for the loss functions) so no production file is touched.
At each step, logs: each loss component (value_loss, rank_loss, flow_loss,
policy_ce_loss, entropy, total), the value head's gradient norm (mean/std
across all value-network parameters), and the batch's value-output mean/std.
Dumps to diagnostics/value_gradient_flow.json. Read-only w.r.t. checkpoints
(does not save anything back to checkpoints/).
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from flow import (
    compute_edge_flow,
    compute_state_flow,
    flow_consistency_loss,
    ranking_loss,
)
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.train_data import ESCStateBundleDataset
from train.trainer_causal_mcts import _action_index
from train.utils import load_merged_config
from utils.seed import set_global_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_STEPS = 100


def main() -> None:
    config = load_merged_config(ROOT)
    seed = 42
    set_global_seed(seed)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C
    n_causes = ESCState.N_C

    transition_model = LinearTransitionModel(
        state_dim=state_dim, n_causes=n_causes, d_e=ESCState.D_E,
        n_strategies=ESCAction.NUM_STRATEGIES,
    )
    env = ESCEnv(transition_model=transition_model, config=config)
    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    planner = MCTS(env=env, policy_network=policy, value_network=value, config=config)

    lr = float(config.get("learning_rate", config.get("learningrate", 1e-4)))
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()) + list(transition_model.parameters()),
        lr=lr,
    )

    batch_size = int(config.get("batch_size", config.get("batchsize", 16)))
    bundles_path = os.path.join(ROOT, "artifacts", "states", "train.pt")
    bundle_ds = ESCStateBundleDataset(bundles_path)

    margin = float(config.get("gamma_margin", config.get("gammamargin", 1.0)))
    w_flow = float(config.get("flow_loss_weight", config.get("flowlossweight", 1.0)))
    w_rank = float(config.get("ranking_loss_weight", config.get("rankinglossweight", 1.0)))
    w_policy = float(config.get("policy_loss_weight", config.get("policylossweight", 1.0)))
    entropy_bonus = float(config.get("entropy_bonus", config.get("entropybonus", 0.01)))
    eps = 1e-8

    log: list[dict] = []

    for step in range(N_STEPS):
        idx = torch.randint(0, len(bundle_ds), (batch_size,))
        states = [bundle_ds.get_state(int(i)) for i in idx]

        planner.policy_network.eval()
        planner.value_network.eval()
        rewards: list[float] = []
        candidate_actions = []
        policy_targets = []
        with torch.no_grad():
            for state in states:
                action, actions, policy_target = planner.search_with_policy(state)
                _next_state, reward, _done, _info = planner.env.step(state, action)
                rewards.append(float(reward))
                candidate_actions.append(actions)
                policy_targets.append(policy_target)

        planner.policy_network.train()
        planner.value_network.train()

        S = torch.stack([s.to_tensor() for s in states])
        R = torch.tensor(rewards, dtype=S.dtype, device=S.device)
        policy_probs = planner.policy_network(S)
        values_2d = planner.value_network(S)
        values = values_2d.squeeze(-1)

        policy_ce_terms = []
        entropy_terms = []
        for b, (actions, target) in enumerate(zip(candidate_actions, policy_targets)):
            aidx = torch.tensor([_action_index(a, n_causes) for a in actions], dtype=torch.long)
            restricted = policy_probs[b].gather(0, aidx)
            restricted = restricted / (restricted.sum() + eps)
            target = target.to(restricted.dtype)
            policy_ce_terms.append(-(target * torch.log(restricted + eps)).sum())
            entropy_terms.append(-(restricted * torch.log(restricted + eps)).sum())
        policy_ce_loss = torch.stack(policy_ce_terms).mean()
        entropy = torch.stack(entropy_terms).mean()

        value_loss = F.mse_loss(values, R)

        num_actions_ = policy_probs.shape[-1]
        V = values_2d.expand(-1, num_actions_)
        Q_b = policy_probs
        F_s = compute_state_flow(Q_b, V)
        F_edges = compute_edge_flow(F_s, policy_probs.detach())
        flow_from_policy = compute_edge_flow(F_s.detach(), policy_probs.detach()).detach()
        flow_loss = flow_consistency_loss(F_edges, flow_from_policy)

        v_winner = values_2d.mean()
        rank_loss = ranking_loss(v_winner, torch.zeros_like(v_winner), margin=margin)

        loss = (
            value_loss + w_flow * flow_loss + w_rank * rank_loss
            + w_policy * policy_ce_loss - entropy_bonus * entropy
        )

        optimizer.zero_grad()
        loss.backward()

        value_grad_norms = [
            p.grad.norm().item() for p in value.parameters() if p.grad is not None
        ]
        grad_mean = sum(value_grad_norms) / len(value_grad_norms) if value_grad_norms else 0.0
        grad_std = (
            (sum((g - grad_mean) ** 2 for g in value_grad_norms) / len(value_grad_norms)) ** 0.5
            if value_grad_norms else 0.0
        )

        # Gradient of value_loss alone vs rank_loss alone on the value head's
        # output layer bias (a scalar -- isolates each term's raw pull direction
        # without needing a second backward pass through the whole graph).
        head_bias_grad_total = float(value._head.bias.grad.item()) if value._head.bias.grad is not None else None

        optimizer.step()

        log.append({
            "step": step,
            "loss_total": float(loss.item()),
            "value_loss": float(value_loss.item()),
            "rank_loss": float(rank_loss.item()),
            "flow_loss": float(flow_loss.item()),
            "policy_ce_loss": float(policy_ce_loss.item()),
            "entropy": float(entropy.item()),
            "reward_batch_mean": float(R.mean().item()),
            "reward_batch_std": float(R.std().item()),
            "value_output_mean": float(values.mean().item()),
            "value_output_std": float(values.std().item()),
            "value_head_grad_norm_mean": grad_mean,
            "value_head_grad_norm_std": grad_std,
            "value_head_bias_grad": head_bias_grad_total,
        })
        if step % 10 == 0 or step == N_STEPS - 1:
            print(
                f"[grad_flow] step={step} loss={loss.item():.4f} value_loss={value_loss.item():.4f} "
                f"rank_loss={rank_loss.item():.4f} flow_loss={flow_loss.item():.6f} "
                f"R_mean={R.mean().item():.4f} R_std={R.std().item():.4f} "
                f"V_mean={values.mean().item():.4f} V_std={values.std().item():.4f} "
                f"grad_norm_mean={grad_mean:.6f}"
            )

    out_dir = os.path.join(ROOT, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "value_gradient_flow.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[grad_flow] wrote {out_path}")


if __name__ == "__main__":
    main()
