"""Causal MCTS trainer: MCTS action selection plus value + flow-matching regularizers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from esc.action import ESCAction
from esc.state import ESCState
from flow import (
    compute_edge_flow,
    compute_state_flow,
    flow_consistency_loss,
    ranking_loss,
)
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import TransitionModel
from models.value import ValueNetwork


def _action_index(action: ESCAction, n_causes: int) -> int:
    """Flat policy-network index for an action.

    Matches the ordering generate_candidate_actions(filter_by_phase=False)
    produces (strategy_id outer, cause_index inner) and the indexing
    eval/run_eval.py already assumes when decoding argmax back to an
    action (idx // n_c, idx % n_c).
    """
    return action.strategy_id * n_causes + action.cause_index


class CausalMCTSTrainer:
    """Joint optimizer over π, V, and f_θ with MCTS-collected rewards."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        transition: TransitionModel,
        mcts: MCTS,
        config: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.value = value
        self.transition = transition
        self.mcts = mcts
        self.config = config

        lr = float(config.get("learning_rate", config.get("learningrate", 1e-4)))
        self.optimizer = torch.optim.Adam(
            list(policy.parameters())
            + list(value.parameters())
            + list(transition.parameters()),
            lr=lr,
        )

    def train_step(self, states: list[ESCState]) -> dict[str, float]:
        self.mcts.policy_network.eval()
        self.mcts.value_network.eval()

        rewards: list[float] = []
        candidate_actions: list[list[ESCAction]] = []
        policy_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for state in states:
                action, actions, policy_target = self.mcts.search_with_policy(state)
                _next_state, reward, _done, _info = self.mcts.env.step(state, action)
                rewards.append(float(reward))
                candidate_actions.append(actions)
                policy_targets.append(policy_target)

        self.mcts.policy_network.train()
        self.mcts.value_network.train()

        S = torch.stack([s.to_tensor() for s in states])
        R = torch.tensor(rewards, dtype=S.dtype, device=S.device)
        policy_probs = self.mcts.policy_network(S)
        values_2d = self.mcts.value_network(S)
        values = values_2d.squeeze(-1)

        # --- A1: policy loss against the MCTS visit-count distribution ---
        # policy_probs is the network's distribution over the full
        # (unfiltered) action space; restrict/gather it to each state's
        # phase-filtered candidate list so it lines up with policy_target,
        # then renormalize into a valid distribution over just those
        # candidates before scoring against the MCTS target.
        n_causes = ESCState.N_C
        eps = 1e-8
        policy_ce_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        for b, (actions, target) in enumerate(zip(candidate_actions, policy_targets)):
            idx = torch.tensor(
                [_action_index(a, n_causes) for a in actions],
                dtype=torch.long,
                device=policy_probs.device,
            )
            restricted = policy_probs[b].gather(0, idx)
            restricted = restricted / (restricted.sum() + eps)
            target = target.to(restricted.dtype)
            policy_ce_terms.append(-(target * torch.log(restricted + eps)).sum())
            entropy_terms.append(-(restricted * torch.log(restricted + eps)).sum())

        policy_ce_loss = torch.stack(policy_ce_terms).mean()
        entropy = torch.stack(entropy_terms).mean()

        value_loss = F.mse_loss(values, R)

        num_actions = policy_probs.shape[-1]
        V = values_2d.expand(-1, num_actions)
        Q_b = policy_probs
        F_s = compute_state_flow(Q_b, V)
        F_edges = compute_edge_flow(F_s, policy_probs.detach())
        flow_from_policy = compute_edge_flow(
            F_s.detach(), policy_probs.detach()
        ).detach()
        flow_loss = flow_consistency_loss(F_edges, flow_from_policy)

        margin = float(
            self.config.get("gamma_margin", self.config.get("gammamargin", 1.0))
        )
        v_winner = values_2d.mean()
        rank_loss = ranking_loss(
            v_winner,
            torch.zeros_like(v_winner),
            margin=margin,
        )

        w_flow = float(
            self.config.get("flow_loss_weight", self.config.get("flowlossweight", 1.0))
        )
        w_rank = float(
            self.config.get(
                "ranking_loss_weight", self.config.get("rankinglossweight", 1.0)
            )
        )
        w_policy = float(
            self.config.get(
                "policy_loss_weight", self.config.get("policylossweight", 1.0)
            )
        )
        entropy_bonus = float(
            self.config.get("entropy_bonus", self.config.get("entropybonus", 0.01))
        )
        loss = (
            value_loss
            + w_flow * flow_loss
            + w_rank * rank_loss
            + w_policy * policy_ce_loss
            - entropy_bonus * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            v_batch = values.detach()
            r_batch = R.detach()
            v_std = v_batch.std(unbiased=False)
            r_std = r_batch.std(unbiased=False)
            if v_std > 1e-8 and r_std > 1e-8:
                cov = ((v_batch - v_batch.mean()) * (r_batch - r_batch.mean())).mean()
                pearson_r = float((cov / (v_std * r_std)).item())
            else:
                pearson_r = 0.0

        return {
            "loss": float(loss.item()),
            "value_loss": float(value_loss.item()),
            "flow_loss": float(flow_loss.item()),
            "rank_loss": float(rank_loss.item()),
            "policy_ce_loss": float(policy_ce_loss.item()),
            "entropy": float(entropy.item()),
            "pearson_r": pearson_r,
            "reward_mean": float(R.mean().item()),
            "reward_std": float(R.std(unbiased=False).item()),
            "value_mean": float(values.mean().item()),
            "value_std": float(values.std(unbiased=False).item()),
        }
