"""
Comprehensive test suite for the ESC MDP + MCTS pipeline.

Tests are organised into classes matching the module hierarchy.
Each test verifies behaviour and algorithmic correctness, not just shapes.

Guiding principles
------------------
- Every reward component is tested for correct sign and direction.
- MCTS tests verify that search produces different outputs than a random
  baseline and that Q-values change between simulations.
- CausalGraph tests verify edge validation and resolution clamping.
- Transition tests verify that step() produces non-identical next states.
- Reproducibility tests verify that the same seed yields the same output.
"""

import json
import math
import os
import tempfile
import pytest
import torch

from esc.causal_graph import CausalGraph, CauseNode
from esc.state import ESCState
from esc.action import ESCAction, generate_candidate_actions, embed_action, ActionEmbedder
from esc.reward import compute_reward, reward_components, _r_cause, _r_emotion, _r_phase
from models.transition import LinearTransitionModel, RandomTransitionModel, TransitionOutput
from esc.env import ESCEnv
from mcts.node import TreeNode
from mcts.mcts import MCTS, PolicyNetwork, ValueNetwork
from utils.seed import set_global_seed, ExperimentConfig
from utils.logging import EpisodeLogger


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture()
def graph() -> CausalGraph:
    """CausalGraph with 4 causes and one edge."""
    g = CausalGraph(d_c=ESCState.D_C)
    for i in range(ESCState.N_C):
        g.add_cause(label=f"cause_{i}", embedding=torch.zeros(ESCState.D_C))
    g.add_edge(0, 1)
    return g


@pytest.fixture()
def base_state(graph) -> ESCState:
    """Structurally valid zero-filled ESCState at phase 0."""
    phase = torch.zeros(3)
    phase[0] = 1.0   # argmax = 0 → Exploration
    return ESCState(
        history_embeddings=torch.zeros(ESCState.K_HISTORY_WINDOW, ESCState.D_H),
        causal_graph=graph,
        emotion_vector=torch.zeros(ESCState.D_E),
        phase_embedding=phase,
        turn_index=0,
        target_emotion=torch.zeros(ESCState.D_E),
    )


@pytest.fixture()
def env() -> ESCEnv:
    return ESCEnv(
        transition_model=RandomTransitionModel(
            d_e=ESCState.D_E,
            n_causes=ESCState.N_C,
            seed=0,
        ),
        max_horizon=20,
    )


@pytest.fixture()
def mcts_planner(env) -> MCTS:
    set_global_seed(42)
    return MCTS(env=env, num_simulations=10)


# ======================================================================
# 1. CausalGraph
# ======================================================================

class TestCausalGraph:

    def test_add_cause_and_count(self):
        g = CausalGraph(d_c=ESCState.D_C)
        g.add_cause("work stress", embedding=torch.zeros(ESCState.D_C))
        assert g.num_causes == 1

    def test_add_cause_wrong_dim_raises(self):
        g = CausalGraph(d_c=ESCState.D_C)
        with pytest.raises(ValueError, match="shape"):
            g.add_cause("bad", embedding=torch.zeros(ESCState.D_C + 1))

    def test_add_edge_valid(self, graph):
        # Edge 0→1 was added in the fixture; check it appears
        assert 1 in graph.get_edges(0)

    def test_add_edge_self_loop_raises(self, graph):
        with pytest.raises(ValueError, match="Self-loop"):
            graph.add_edge(0, 0)

    def test_add_edge_out_of_range_raises(self, graph):
        with pytest.raises(IndexError):
            graph.add_edge(0, 99)

    def test_update_resolution_correct_values(self, graph):
        new_res = torch.tensor([0.1, 0.5, 0.9, 0.3])
        graph.update_resolution(new_res)
        stored = graph.resolution_tensor()
        assert torch.allclose(stored, new_res, atol=1e-5)

    def test_update_resolution_clamps_to_unit_interval(self, graph):
        # Values outside [0,1] should be clamped
        graph.update_resolution(torch.tensor([1.5, -0.2, 0.5, 0.0]))
        res = graph.resolution_tensor()
        assert res[0].item() == pytest.approx(1.0)
        assert res[1].item() == pytest.approx(0.0)

    def test_update_resolution_wrong_length_raises(self, graph):
        with pytest.raises(ValueError, match="shape"):
            graph.update_resolution(torch.tensor([0.5, 0.5]))

    def test_resolution_tensor_all_zero_initially(self, graph):
        res = graph.resolution_tensor()
        assert torch.all(res == 0.0)

    def test_cause_embedding_matrix_shape(self, graph):
        mat = graph.cause_embedding_matrix(ESCState.N_C, ESCState.D_C)
        assert mat.shape == (ESCState.N_C, ESCState.D_C)

    def test_placeholder_creates_correct_graph(self):
        g = CausalGraph.placeholder(n_causes=3, d_c=ESCState.D_C)
        assert g.num_causes == 3
        assert torch.all(g.resolution_tensor() == 0.0)

    def test_neighbours(self, graph):
        neighbours = graph.neighbours(0)
        assert len(neighbours) == 1
        assert neighbours[0].index == 1


# ======================================================================
# 2. ESCState
# ======================================================================

class TestESCState:

    def test_from_dialogue_produces_valid_state(self):
        state = ESCState.from_dialogue(["I feel anxious", "I can't sleep"])
        assert state.history_embeddings.shape == (ESCState.K_HISTORY_WINDOW, ESCState.D_H)
        assert state.emotion_vector.shape == (ESCState.D_E,)
        assert state.phase_embedding.shape == (3,)
        assert state.causal_graph.num_causes == ESCState.N_C

    def test_phase_embedding_sums_to_one(self):
        state = ESCState.from_dialogue([])
        assert state.phase_embedding.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_to_tensor_shape(self, base_state):
        vec = base_state.to_tensor()
        assert vec.shape == (ESCState.get_state_dim(),)

    def test_get_state_dim_correct(self):
        # D_P is now 3 (not 64), so total = 768+384+128+3 = 1283
        expected = ESCState.D_H + ESCState.D_C + ESCState.D_E + ESCState.D_P
        assert ESCState.get_state_dim() == expected
        assert ESCState.D_P == 3

    def test_to_tensor_pooling_correct(self, graph):
        """Mean-pooling of history and cause embeddings is numerically correct."""
        hist = torch.randn(ESCState.K_HISTORY_WINDOW, ESCState.D_H)
        emo = torch.randn(ESCState.D_E)
        phase = torch.softmax(torch.randn(3), dim=0)
        state = ESCState(
            history_embeddings=hist,
            causal_graph=graph,
            emotion_vector=emo,
            phase_embedding=phase,
            target_emotion=torch.zeros(ESCState.D_E),
        )
        vec = state.to_tensor()
        DH, DC, DE = ESCState.D_H, ESCState.D_C, ESCState.D_E
        assert torch.allclose(vec[:DH], hist.mean(dim=0), atol=1e-5)
        assert torch.allclose(vec[DH + DC: DH + DC + DE], emo, atol=1e-5)
        assert torch.allclose(vec[DH + DC + DE:], phase, atol=1e-5)

    def test_current_phase_returns_argmax(self, base_state):
        assert base_state.current_phase == 0

    def test_clone_is_independent(self, base_state):
        clone = base_state.clone()
        clone.emotion_vector[0] = 999.0
        assert base_state.emotion_vector[0].item() != 999.0

    def test_clone_causal_graph_is_independent(self, base_state):
        clone = base_state.clone()
        clone.causal_graph.update_resolution(torch.ones(ESCState.N_C))
        original_res = base_state.causal_graph.resolution_tensor()
        assert torch.all(original_res == 0.0), (
            "Mutation of clone's causal graph must not affect original"
        )

    def test_shape_validation_on_construction(self, graph):
        with pytest.raises(ValueError, match="history_embeddings"):
            ESCState(
                history_embeddings=torch.zeros(3, ESCState.D_H),  # wrong K
                causal_graph=graph,
                emotion_vector=torch.zeros(ESCState.D_E),
                phase_embedding=torch.softmax(torch.zeros(3), dim=0),
            )

    def test_from_dialogue_with_encoder_hook(self):
        """Encoder hook is called and its output is used."""
        def fake_encoder(turns):
            return {
                "history": torch.ones(ESCState.K_HISTORY_WINDOW, ESCState.D_H),
                "emotion": torch.ones(ESCState.D_E) * 0.5,
                "causes": [
                    {"label": "stress", "embedding": torch.ones(ESCState.D_C) * 0.1}
                ],
            }
        state = ESCState.from_dialogue(["test"], encoder=fake_encoder)
        assert torch.all(state.history_embeddings == 1.0)
        assert torch.all(state.emotion_vector == 0.5)
        # First cause node should have embedding ~0.1
        node = state.causal_graph.get_node(0)
        assert node.embedding[0].item() == pytest.approx(0.1)


# ======================================================================
# 3. ESCAction and candidate generation
# ======================================================================

class TestESCAction:

    def test_phase_0_strategies(self, base_state):
        """Phase 0 (Exploration) should only produce strategies {0,1,2,3}."""
        assert base_state.current_phase == 0
        candidates = generate_candidate_actions(base_state, filter_by_phase=True)
        strategy_ids = {a.strategy_id for a in candidates}
        assert strategy_ids == set(ESCAction.PHASE_STRATEGY_MAP[0])

    def test_phase_1_strategies(self, base_state):
        """Phase 1 (Comforting) should only produce strategies {2,3,4,7}."""
        base_state.phase_embedding = torch.tensor([0.0, 1.0, 0.0])  # phase 1
        candidates = generate_candidate_actions(base_state, filter_by_phase=True)
        strategy_ids = {a.strategy_id for a in candidates}
        assert strategy_ids == set(ESCAction.PHASE_STRATEGY_MAP[1])

    def test_phase_2_strategies(self, base_state):
        """Phase 2 (Action) should only produce strategies {4,5,6,7}."""
        base_state.phase_embedding = torch.tensor([0.0, 0.0, 1.0])  # phase 2
        candidates = generate_candidate_actions(base_state, filter_by_phase=True)
        strategy_ids = {a.strategy_id for a in candidates}
        assert strategy_ids == set(ESCAction.PHASE_STRATEGY_MAP[2])

    def test_no_phase_filter_returns_all(self, base_state):
        candidates = generate_candidate_actions(base_state, filter_by_phase=False)
        assert len(candidates) == ESCAction.NUM_STRATEGIES * ESCState.N_C

    def test_candidates_non_empty_always(self, base_state):
        for phase_idx in range(3):
            base_state.phase_embedding = torch.zeros(3)
            base_state.phase_embedding[phase_idx] = 1.0
            candidates = generate_candidate_actions(base_state, filter_by_phase=True)
            assert len(candidates) > 0

    def test_action_hash_equality(self):
        a1 = ESCAction(strategy_id=2, cause_index=1)
        a2 = ESCAction(strategy_id=2, cause_index=1)
        a3 = ESCAction(strategy_id=3, cause_index=1)
        assert a1 == a2
        assert a1 != a3
        assert hash(a1) == hash(a2)
        d = {a1: "ok"}
        assert d[a2] == "ok"

    def test_embed_action_shape(self):
        action = ESCAction(strategy_id=0, cause_index=0)
        emb = embed_action(action, strategy_dim=64, cause_dim=64)
        assert emb.shape == (128,)

    def test_embed_action_one_hot_correct_slots(self):
        action = ESCAction(strategy_id=5, cause_index=2)
        emb = embed_action(action, strategy_dim=64, cause_dim=64)
        assert emb[5] == 1.0
        assert emb[64 + 2] == 1.0
        assert emb.sum().item() == pytest.approx(2.0)

    def test_action_embedder_output_shape(self):
        embedder = ActionEmbedder(
            num_strategies=ESCAction.NUM_STRATEGIES,
            num_causes=ESCState.N_C,
            strategy_emb_dim=32,
            cause_emb_dim=32,
        )
        action = ESCAction(strategy_id=3, cause_index=1)
        emb = embedder(action)
        assert emb.shape == (64,)


# ======================================================================
# 4. Reward components — correctness of signs and semantics
# ======================================================================

class TestReward:

    def test_r_cause_zero_when_no_change(self, base_state):
        clone = base_state.clone()
        assert _r_cause(base_state, clone) == pytest.approx(0.0)

    def test_r_cause_positive_when_resolution_increases(self, base_state):
        next_s = base_state.clone()
        next_s.causal_graph.update_resolution(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        assert _r_cause(base_state, next_s) > 0.0

    def test_r_cause_negative_when_resolution_decreases(self, base_state):
        # Set initial resolution to 0.8
        base_state.causal_graph.update_resolution(torch.tensor([0.8, 0.8, 0.8, 0.8]))
        next_s = base_state.clone()
        next_s.causal_graph.update_resolution(torch.tensor([0.3, 0.3, 0.3, 0.3]))
        assert _r_cause(base_state, next_s) < 0.0

    def test_r_cause_delta_not_cumulative(self, base_state):
        """R_cause must reflect *delta*, not the absolute sum of resolution probs."""
        # Already 90% resolved: the agent should get ~0 credit if probs don't change
        base_state.causal_graph.update_resolution(torch.tensor([0.9, 0.9, 0.9, 0.9]))
        next_s = base_state.clone()
        # No change in resolution
        next_s.causal_graph.update_resolution(torch.tensor([0.9, 0.9, 0.9, 0.9]))
        assert _r_cause(base_state, next_s) == pytest.approx(0.0), (
            "R_cause must be 0 when resolution does not change, "
            "regardless of the absolute resolution level"
        )

    def test_r_emotion_positive_when_approaching_target(self, base_state):
        """Moving emotion closer to target → positive R_emotion."""
        target = torch.zeros(ESCState.D_E)
        base_state.target_emotion = target
        # Start far from target
        base_state.emotion_vector = torch.ones(ESCState.D_E)
        next_s = base_state.clone()
        # Move closer
        next_s.emotion_vector = torch.ones(ESCState.D_E) * 0.5
        next_s.target_emotion = target
        assert _r_emotion(base_state, next_s) > 0.0

    def test_r_emotion_negative_when_moving_away(self, base_state):
        """Moving emotion further from target → negative R_emotion."""
        target = torch.zeros(ESCState.D_E)
        base_state.target_emotion = target
        base_state.emotion_vector = torch.zeros(ESCState.D_E)  # at target
        next_s = base_state.clone()
        next_s.emotion_vector = torch.ones(ESCState.D_E)       # moved away
        next_s.target_emotion = target
        assert _r_emotion(base_state, next_s) < 0.0

    def test_r_emotion_zero_when_stationary(self, base_state):
        clone = base_state.clone()
        assert _r_emotion(base_state, clone) == pytest.approx(0.0, abs=1e-6)

    def test_r_phase_one_when_phase_advances(self, base_state):
        next_s = base_state.clone()
        next_s.phase_embedding = torch.tensor([0.0, 1.0, 0.0])  # phase 1
        assert _r_phase(base_state, next_s) == pytest.approx(1.0)

    def test_r_phase_zero_when_phase_same(self, base_state):
        clone = base_state.clone()
        assert _r_phase(base_state, clone) == pytest.approx(0.0)

    def test_r_phase_zero_when_phase_regresses(self, base_state):
        base_state.phase_embedding = torch.tensor([0.0, 0.0, 1.0])  # phase 2
        next_s = base_state.clone()
        next_s.phase_embedding = torch.tensor([0.0, 1.0, 0.0])     # phase 1
        assert _r_phase(base_state, next_s) == pytest.approx(0.0)

    def test_reward_components_dict_has_all_keys(self, base_state):
        comp = reward_components(base_state, base_state.clone())
        assert "R_cause" in comp
        assert "R_emotion" in comp
        assert "R_phase" in comp

    def test_reward_weights_scale_correctly(self, base_state):
        """Doubling a weight should double that component's contribution."""
        next_s = base_state.clone()
        next_s.causal_graph.update_resolution(torch.tensor([0.5, 0.5, 0.5, 0.5]))

        r1 = compute_reward(base_state, next_s, alpha=1.0, beta=0.0, gamma=0.0)
        r2 = compute_reward(base_state, next_s, alpha=2.0, beta=0.0, gamma=0.0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-5)

    def test_reward_normalised_in_reasonable_range(self, base_state):
        """Each component individually should be in roughly [-2, 2] for typical inputs."""
        next_s = base_state.clone()
        next_s.emotion_vector = torch.randn(ESCState.D_E)
        next_s.causal_graph.update_resolution(torch.rand(ESCState.N_C))
        next_s.phase_embedding = torch.tensor([0.0, 1.0, 0.0])

        comp = reward_components(base_state, next_s)
        assert abs(comp["R_cause"]) <= 1.0 + 1e-5, "R_cause must be in [-1, 1]"
        assert abs(comp["R_phase"]) <= 1.0 + 1e-5, "R_phase must be in {0, 1}"
        # R_emotion can exceed 1 for extreme emotion vectors but should be finite
        assert math.isfinite(comp["R_emotion"])


# ======================================================================
# 5. Transition models
# ======================================================================

class TestTransitionModels:

    def test_random_transition_produces_different_states(self, base_state):
        """env.step() must produce a next state that differs from the current."""
        tm = RandomTransitionModel(d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=1)
        out = tm.forward(base_state, ESCAction(strategy_id=0, cause_index=0))
        assert out.next_emotion.shape == (ESCState.D_E,)
        assert out.next_phase_logits.shape == (3,)
        assert out.delta_resolution.shape == (ESCState.N_C,)
        # With noise_scale=0.05 and 128 dims, emotion should have changed
        assert not torch.all(out.next_emotion == base_state.emotion_vector)

    def test_linear_transition_model_shapes(self, base_state):
        tm = LinearTransitionModel(
            state_dim=ESCState.get_state_dim(),
            n_causes=ESCState.N_C,
            d_e=ESCState.D_E,
            n_strategies=ESCAction.NUM_STRATEGIES,
        )
        out = tm.forward(base_state, ESCAction(strategy_id=2, cause_index=1))
        assert out.next_emotion.shape == (ESCState.D_E,)
        assert out.next_phase_logits.shape == (3,)
        assert out.delta_resolution.shape == (ESCState.N_C,)

    def test_linear_transition_emotion_bounded(self, base_state):
        """tanh output must be strictly in (-1, 1)."""
        tm = LinearTransitionModel(
            state_dim=ESCState.get_state_dim(),
            n_causes=ESCState.N_C,
            d_e=ESCState.D_E,
            n_strategies=ESCAction.NUM_STRATEGIES,
        )
        out = tm.forward(base_state, ESCAction(strategy_id=0, cause_index=0))
        assert torch.all(out.next_emotion > -1.0)
        assert torch.all(out.next_emotion < 1.0)

    def test_linear_transition_resolution_bounded(self, base_state):
        """tanh output must be in (-1, 1)."""
        tm = LinearTransitionModel(
            state_dim=ESCState.get_state_dim(),
            n_causes=ESCState.N_C,
            d_e=ESCState.D_E,
            n_strategies=ESCAction.NUM_STRATEGIES,
        )
        out = tm.forward(base_state, ESCAction(strategy_id=0, cause_index=0))
        assert torch.all(out.delta_resolution > -1.0)
        assert torch.all(out.delta_resolution < 1.0)


# ======================================================================
# 6. ESCEnv — step() produces real dynamics
# ======================================================================

class TestESCEnv:

    def test_reset_returns_valid_state(self, env):
        state = env.reset()
        assert state.history_embeddings.shape == (env.k_hist, env.d_h)
        assert state.emotion_vector.shape == (env.d_e,)
        assert state.phase_embedding.shape == (3,)
        assert state.phase_embedding.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_step_next_state_differs_from_current(self, env):
        """The transition model must produce a non-identical next state."""
        state = env.reset()
        action = ESCAction(strategy_id=0, cause_index=0)
        next_state, _, _, _ = env.step(state, action)
        # Emotion must have changed (RandomTransitionModel adds noise)
        assert not torch.allclose(
            state.emotion_vector, next_state.emotion_vector
        ), "next_state.emotion_vector must differ from state.emotion_vector"

    def test_step_reward_is_finite_float(self, env):
        state = env.reset()
        _, reward, _, _ = env.step(state, ESCAction(strategy_id=0, cause_index=0))
        assert isinstance(reward, float)
        assert math.isfinite(reward)

    def test_step_increments_turn_index(self, env):
        state = env.reset()
        assert state.turn_index == 0
        next_state, _, _, _ = env.step(state, ESCAction(strategy_id=0, cause_index=0))
        assert next_state.turn_index == 1

    def test_step_history_window_slides(self, env):
        """After step(), history row 0 of parent should be dropped."""
        state = env.reset()
        old_row_1 = state.history_embeddings[1].clone()
        next_state, _, _, _ = env.step(state, ESCAction(strategy_id=0, cause_index=0))
        # Row 0 of next_state should equal old row 1
        assert torch.allclose(next_state.history_embeddings[0], old_row_1, atol=1e-6)

    def test_step_does_not_mutate_original_state(self, env):
        state = env.reset()
        original_turn = state.turn_index
        original_emo = state.emotion_vector.clone()
        env.step(state, ESCAction(strategy_id=0, cause_index=0))
        assert state.turn_index == original_turn
        assert torch.allclose(state.emotion_vector, original_emo)

    def test_done_at_horizon(self, env):
        state = env.reset()
        state.turn_index = env.max_horizon - 1
        _, _, done, _ = env.step(state, ESCAction(strategy_id=0, cause_index=0))
        assert done

    def test_not_done_before_horizon(self, env):
        state = env.reset()
        _, _, done, _ = env.step(state, ESCAction(strategy_id=0, cause_index=0))
        assert not done

    def test_info_dict_has_required_keys(self, env):
        state = env.reset()
        _, _, _, info = env.step(state, ESCAction(strategy_id=2, cause_index=1))
        for key in ("turn", "strategy_used", "cause_targeted",
                    "horizon_reached", "phase", "R_cause", "R_emotion",
                    "R_phase", "resolution"):
            assert key in info, f"Missing info key: {key}"

    def test_phase_embedding_is_valid_prob_distribution_after_step(self, env):
        state = env.reset()
        next_state, _, _, _ = env.step(state, ESCAction(strategy_id=0, cause_index=0))
        prob_sum = next_state.phase_embedding.sum().item()
        assert prob_sum == pytest.approx(1.0, abs=1e-5)
        assert torch.all(next_state.phase_embedding >= 0.0)

    def test_resolution_clamped_to_unit_interval_after_step(self, env):
        state = env.reset()
        next_state, _, _, _ = env.step(state, ESCAction(strategy_id=0, cause_index=0))
        res = next_state.causal_graph.resolution_tensor()
        assert torch.all(res >= 0.0) and torch.all(res <= 1.0)

    def test_reward_weights_affect_output(self, env):
        """Different reward weights must produce different total rewards."""
        env_a = ESCEnv(
            transition_model=RandomTransitionModel(
                d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=7
            ),
            reward_weights=(1.0, 0.0, 0.0),
        )
        env_b = ESCEnv(
            transition_model=RandomTransitionModel(
                d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=7
            ),
            reward_weights=(0.0, 1.0, 0.0),
        )
        state = ESCState.from_dialogue([])
        action = ESCAction(strategy_id=0, cause_index=0)
        _, r_a, _, _ = env_a.step(state, action)
        state2 = ESCState.from_dialogue([])
        _, r_b, _, _ = env_b.step(state2, action)
        # They should differ unless both components happen to be identical
        # (astronomically unlikely with random dynamics)
        assert r_a != pytest.approx(r_b, abs=1e-8), (
            "Different reward weights should produce different total rewards"
        )

    def test_render_returns_string(self, env):
        state = env.reset()
        s = env.render(state)
        assert isinstance(s, str)
        assert "Phase" in s
        assert "Turn" in s


# ======================================================================
# 7. TreeNode
# ======================================================================

class TestTreeNode:

    def test_initial_stats_all_zero(self, base_state):
        node = TreeNode(state=base_state)
        assert node.N == 0
        assert node.W == pytest.approx(0.0)
        assert node.Q == pytest.approx(0.0)
        assert node.is_leaf
        assert node.depth == 0

    def test_expand_creates_correct_number_of_children(self, base_state):
        node = TreeNode(state=base_state)
        actions = generate_candidate_actions(base_state)
        priors = torch.ones(len(actions)) / len(actions)
        node.expand(actions, priors)
        assert len(node.children) == len(actions)
        assert not node.is_leaf

    def test_expand_twice_raises(self, base_state):
        node = TreeNode(state=base_state)
        actions = [ESCAction(strategy_id=0, cause_index=0)]
        priors = torch.tensor([1.0])
        node.expand(actions, priors)
        with pytest.raises(RuntimeError, match="already-expanded"):
            node.expand(actions, priors)

    def test_expand_mismatch_raises(self, base_state):
        node = TreeNode(state=base_state)
        actions = [ESCAction(strategy_id=0, cause_index=0)]
        priors = torch.tensor([0.5, 0.5])  # wrong length
        with pytest.raises(AssertionError):
            node.expand(actions, priors)

    def test_update_statistics_correct(self, base_state):
        node = TreeNode(state=base_state)
        node.update(2.0)
        assert node.N == 1
        assert node.W == pytest.approx(2.0)
        assert node.Q == pytest.approx(2.0)
        node.update(4.0)
        assert node.N == 2
        assert node.W == pytest.approx(6.0)
        assert node.Q == pytest.approx(3.0)

    def test_backpropagate_updates_all_ancestors(self, base_state):
        root = TreeNode(state=base_state)
        child = TreeNode(state=base_state, parent=root)
        grandchild = TreeNode(state=base_state, parent=child)

        grandchild.backpropagate(5.0)
        assert grandchild.N == 1
        assert child.N == 1
        assert root.N == 1
        assert grandchild.Q == pytest.approx(5.0)
        assert child.Q == pytest.approx(5.0)
        assert root.Q == pytest.approx(5.0)

    def test_select_child_prefers_highest_prior_at_equal_Q(self, base_state):
        node = TreeNode(state=base_state)
        node.N = 10  # give parent visits so exploration term is non-zero
        actions = [ESCAction(strategy_id=i, cause_index=0) for i in range(3)]
        priors = torch.tensor([0.1, 0.7, 0.2])
        node.expand(actions, priors)
        best_action, _ = node.select_child(c_puct=1.0)
        assert best_action == actions[1], "Should prefer highest prior when Q=0 for all"

    def test_select_child_prefers_high_Q_over_low_prior(self, base_state):
        """When one child has a very high Q, it should win despite lower prior."""
        node = TreeNode(state=base_state)
        node.N = 100
        actions = [ESCAction(strategy_id=i, cause_index=0) for i in range(2)]
        priors = torch.tensor([0.1, 0.9])  # action[1] has high prior
        node.expand(actions, priors)
        # Give action[0] a very high Q value
        node.children[actions[0]].N = 50
        node.children[actions[0]].W = 500.0
        node.children[actions[0]].Q = 10.0
        best_action, _ = node.select_child(c_puct=0.1)  # low exploration
        assert best_action == actions[0], "High Q should win with low c_puct"

    def test_select_child_no_children_raises(self, base_state):
        node = TreeNode(state=base_state)
        with pytest.raises(AssertionError):
            node.select_child()

    def test_set_child_state(self, base_state):
        node = TreeNode(state=base_state)
        actions = [ESCAction(strategy_id=0, cause_index=0)]
        node.expand(actions, torch.tensor([1.0]))
        new_state = base_state.clone()
        new_state.turn_index = 99
        node.set_child_state(actions[0], new_state)
        assert node.children[actions[0]].state.turn_index == 99

    def test_set_child_state_wrong_action_raises(self, base_state):
        node = TreeNode(state=base_state)
        node.expand(
            [ESCAction(strategy_id=0, cause_index=0)],
            torch.tensor([1.0]),
        )
        with pytest.raises(KeyError):
            node.set_child_state(ESCAction(strategy_id=5, cause_index=3), base_state)

    def test_visit_count_policy_sums_to_one(self, base_state):
        node = TreeNode(state=base_state)
        actions = [ESCAction(strategy_id=i, cause_index=0) for i in range(3)]
        node.expand(actions, torch.ones(3) / 3)
        # Simulate some visits
        node.children[actions[0]].N = 10
        node.children[actions[1]].N = 5
        node.children[actions[2]].N = 3
        _, probs = node.visit_count_policy(temperature=1.0)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_visit_count_policy_most_visited_has_highest_prob(self, base_state):
        node = TreeNode(state=base_state)
        actions = [ESCAction(strategy_id=i, cause_index=0) for i in range(3)]
        node.expand(actions, torch.ones(3) / 3)
        node.children[actions[0]].N = 100
        node.children[actions[1]].N = 10
        node.children[actions[2]].N = 5
        returned_actions, probs = node.visit_count_policy(temperature=1.0)
        idx_a0 = returned_actions.index(actions[0])
        assert probs[idx_a0].item() == probs.max().item()

    def test_most_visited_child(self, base_state):
        node = TreeNode(state=base_state)
        actions = [ESCAction(strategy_id=i, cause_index=0) for i in range(3)]
        node.expand(actions, torch.ones(3) / 3)
        node.children[actions[2]].N = 42
        best_action, _ = node.most_visited_child()
        assert best_action == actions[2]

    def test_depth_correct(self, base_state):
        root = TreeNode(state=base_state)
        child = TreeNode(state=base_state, parent=root)
        grandchild = TreeNode(state=base_state, parent=child)
        assert root.depth == 0
        assert child.depth == 1
        assert grandchild.depth == 2


# ======================================================================
# 8. MCTS — algorithmic correctness
# ======================================================================

class TestMCTS:

    def test_search_returns_esc_action(self, base_state, mcts_planner):
        action = mcts_planner.search(base_state)
        assert isinstance(action, ESCAction)

    def test_search_action_in_candidates(self, base_state, mcts_planner):
        candidates = generate_candidate_actions(base_state)
        action = mcts_planner.search(base_state)
        assert action in candidates

    def test_search_reproducible_with_same_seed(self, base_state):
        """
        Two MCTS planners constructed with the same seed and fresh envs must
        select the same action.  Each run gets its own ESCEnv so that the
        RandomTransitionModel's internal Generator starts from identical state.
        """
        def make_seeded_planner() -> MCTS:
            set_global_seed(42)
            fresh_env = ESCEnv(
                transition_model=RandomTransitionModel(
                    d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=42
                ),
                max_horizon=20,
            )
            return MCTS(env=fresh_env, num_simulations=10)

        a1 = make_seeded_planner().search(base_state)
        a2 = make_seeded_planner().search(base_state)
        assert a1 == a2, "Same seed must yield same action"

    def test_search_visit_counts_nonzero_after_simulations(self, base_state, env):
        """After search, root children should have been visited."""
        set_global_seed(0)
        planner = MCTS(env=env, num_simulations=20)
        candidates = generate_candidate_actions(base_state)
        priors = torch.ones(len(candidates)) / len(candidates)

        root = TreeNode(state=base_state, prior=1.0)
        root.N = 1
        root.expand(candidates, priors)

        # Run one simulation manually to confirm visit counts update
        planner._simulate(root)
        total_child_visits = sum(c.N for c in root.children.values())
        assert total_child_visits > 0, "Simulations must increment visit counts"

    def test_q_values_updated_after_simulation(self, base_state, env):
        """Q-values should be non-zero after at least one simulation."""
        set_global_seed(1)
        planner = MCTS(env=env, num_simulations=5)
        candidates = generate_candidate_actions(base_state)
        priors = torch.ones(len(candidates)) / len(candidates)

        root = TreeNode(state=base_state)
        root.N = 1
        root.expand(candidates, priors)
        planner._expand_node(root, candidates, priors)

        for _ in range(5):
            planner._simulate(root)

        q_values = [c.Q for c in root.children.values()]
        assert any(q != 0.0 for q in q_values), (
            "After simulations, at least one child Q must be non-zero"
        )

    def test_policy_network_output_shape(self, base_state):
        state_dim = ESCState.get_state_dim()
        num_actions = len(generate_candidate_actions(base_state))
        net = PolicyNetwork(state_dim=state_dim, num_actions=num_actions)
        out = net(base_state.to_tensor())
        assert out.shape == (num_actions,)

    def test_policy_network_sums_to_one(self, base_state):
        state_dim = ESCState.get_state_dim()
        num_actions = len(generate_candidate_actions(base_state))
        net = PolicyNetwork(state_dim=state_dim, num_actions=num_actions)
        out = net(base_state.to_tensor())
        assert out.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_value_network_range(self, base_state):
        net = ValueNetwork(state_dim=ESCState.get_state_dim())
        v = net.value(base_state.to_tensor())
        assert -1.0 <= v <= 1.0

    def test_search_with_policy_returns_correct_types(self, base_state, env):
        set_global_seed(5)
        planner = MCTS(env=env, num_simulations=5)
        action, actions, policy_target = planner.search_with_policy(base_state)
        assert isinstance(action, ESCAction)
        assert len(actions) > 0
        assert policy_target.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_more_simulations_changes_visit_distribution(self, base_state, env):
        """More simulations should generally produce different visit counts."""
        set_global_seed(10)
        p_few = MCTS(env=env, num_simulations=2)
        _, _, policy_few = p_few.search_with_policy(base_state)

        set_global_seed(10)
        p_many = MCTS(env=env, num_simulations=30)
        _, _, policy_many = p_many.search_with_policy(base_state)

        # The policies from 2 vs 30 simulations should differ
        assert not torch.allclose(policy_few, policy_many, atol=1e-3), (
            "Visit-count policy from 2 simulations must differ from 30 simulations"
        )


# ======================================================================
# 9. Reproducibility (utils.seed)
# ======================================================================

class TestReproducibility:

    def test_same_seed_same_random_tensor(self):
        set_global_seed(99)
        t1 = torch.randn(10)
        set_global_seed(99)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seeds_different_tensors(self):
        set_global_seed(1)
        t1 = torch.randn(100)
        set_global_seed(2)
        t2 = torch.randn(100)
        assert not torch.allclose(t1, t2)

    def test_experiment_config_save_load_roundtrip(self):
        cfg = ExperimentConfig(seed=7, num_simulations=25, c_puct=2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cfg.json")
            cfg.save(path)
            loaded = ExperimentConfig.load(path)
        assert loaded.seed == 7
        assert loaded.num_simulations == 25
        assert loaded.c_puct == pytest.approx(2.0)

    def test_experiment_config_apply_sets_seed(self):
        cfg = ExperimentConfig(seed=42)
        cfg.apply()
        t1 = torch.randn(5)
        cfg.apply()
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            set_global_seed(-1)


# ======================================================================
# 10. EpisodeLogger
# ======================================================================

class TestEpisodeLogger:

    def _run_episode(self, env: ESCEnv, planner: MCTS, episode_id: int = 0) -> EpisodeLogger:
        logger = EpisodeLogger()
        logger.start_episode(episode_id)
        state = env.reset()
        done = False
        while not done:
            action = planner.search(state)
            state, reward, done, info = env.step(state, action)
            logger.log_step(action=action, reward=reward, info=info)
        logger.end_episode(final_state=state)
        return logger

    def test_episode_records_correct_num_turns(self, env, mcts_planner):
        logger = self._run_episode(env, mcts_planner)
        rec = logger._episodes[0]
        assert rec.num_turns == env.max_horizon

    def test_total_reward_matches_sum_of_step_rewards(self, env, mcts_planner):
        logger = self._run_episode(env, mcts_planner)
        rec = logger._episodes[0]
        step_total = sum(s.reward for s in rec.steps)
        assert rec.total_reward == pytest.approx(step_total, abs=1e-5)

    def test_start_without_end_then_new_start_raises(self, env, mcts_planner):
        logger = EpisodeLogger()
        logger.start_episode(0)
        with pytest.raises(RuntimeError, match="already active"):
            logger.start_episode(1)

    def test_log_step_without_start_raises(self, env, mcts_planner):
        logger = EpisodeLogger()
        with pytest.raises(RuntimeError, match="outside an episode"):
            logger.log_step(
                action=ESCAction(strategy_id=0, cause_index=0),
                reward=0.0,
                info={"turn": 1, "R_cause": 0, "R_emotion": 0,
                      "R_phase": 0, "phase": 0, "resolution": []},
            )

    def test_end_without_start_raises(self):
        logger = EpisodeLogger()
        with pytest.raises(RuntimeError, match="without a matching"):
            logger.end_episode(final_state=ESCState.from_dialogue([]))

    def test_save_and_load_roundtrip(self, env, mcts_planner):
        logger = self._run_episode(env, mcts_planner)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            logger.save(path)
            loaded = EpisodeLogger.load(path)
        assert len(loaded) == 1
        original_reward = logger._episodes[0].total_reward
        loaded_reward = loaded._episodes[0].total_reward
        assert original_reward == pytest.approx(loaded_reward, abs=1e-5)

    def test_episode_summary_contains_required_keys(self, env, mcts_planner):
        logger = self._run_episode(env, mcts_planner)
        summary = logger.episode_summary(-1)
        for key in ("episode_id", "total_reward", "num_turns",
                    "mean_reward_per_turn", "duration_s",
                    "final_phase", "final_resolution", "strategy_counts"):
            assert key in summary, f"Missing summary key: {key}"


# ======================================================================
# 11. Full end-to-end integration
# ======================================================================

class TestEndToEnd:

    def test_full_pipeline_single_step(self):
        """All five pipeline steps execute without error on real inputs."""
        set_global_seed(0)

        # Step 1: State from dialogue
        turns = ["I've been so anxious about work.", "I can't stop worrying."]
        state = ESCState.from_dialogue(turns)
        assert state.to_tensor().shape == (ESCState.get_state_dim(),)

        # Step 2: Candidate actions (phase-filtered)
        candidates = generate_candidate_actions(state, filter_by_phase=True)
        assert len(candidates) > 0

        # Step 3: Environment step with real transition
        env = ESCEnv(
            transition_model=RandomTransitionModel(
                d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=0
            )
        )
        action = candidates[0]
        next_state, reward, done, info = env.step(state, action)
        assert math.isfinite(reward)
        assert next_state.turn_index == 1
        assert not done

        # Step 4: MCTS selects action
        planner = MCTS(env=env, num_simulations=5)
        chosen = planner.search(state)
        assert chosen in candidates

        # Step 5: Reward is non-trivially structured
        comp = reward_components(state, next_state)
        assert all(math.isfinite(v) for v in comp.values())

    def test_full_episode_terminates(self):
        """A full episode runs to the horizon and terminates correctly."""
        set_global_seed(0)
        env = ESCEnv(
            transition_model=RandomTransitionModel(
                d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=1
            ),
            max_horizon=5,
        )
        planner = MCTS(env=env, num_simulations=3)
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = planner.search(state)
            state, _, done, _ = env.step(state, action)
            steps += 1
            assert steps <= env.max_horizon + 1
        assert state.turn_index == env.max_horizon

    def test_full_episode_with_logger(self):
        """EpisodeLogger correctly captures a full episode."""
        set_global_seed(0)
        env = ESCEnv(
            transition_model=RandomTransitionModel(
                d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=2
            ),
            max_horizon=5,
        )
        planner = MCTS(env=env, num_simulations=3)
        logger = EpisodeLogger()
        logger.start_episode(episode_id=0)
        state = env.reset()
        done = False
        while not done:
            action = planner.search(state)
            state, reward, done, info = env.step(state, action)
            logger.log_step(action=action, reward=reward, info=info)
        logger.end_episode(final_state=state)

        assert len(logger) == 1
        summary = logger.episode_summary(-1)
        assert summary["num_turns"] == env.max_horizon
        assert math.isfinite(summary["total_reward"])

    def test_reward_never_nan_across_episode(self):
        """No NaN or Inf rewards should occur during a normal episode."""
        set_global_seed(3)
        env = ESCEnv(
            transition_model=RandomTransitionModel(
                d_e=ESCState.D_E, n_causes=ESCState.N_C, seed=3
            ),
            max_horizon=10,
        )
        planner = MCTS(env=env, num_simulations=5)
        state = env.reset()
        done = False
        while not done:
            action = planner.search(state)
            state, reward, done, _ = env.step(state, action)
            assert math.isfinite(reward), f"Non-finite reward: {reward}"

    def test_causal_graph_resolution_increases_on_average(self):
        """With RandomTransitionModel's positive bias, resolution should trend up."""
        set_global_seed(4)
        env = ESCEnv(
            transition_model=RandomTransitionModel(
                d_e=ESCState.D_E,
                n_causes=ESCState.N_C,
                resolution_delta_scale=0.15,  # strong positive bias
                seed=4,
            ),
            max_horizon=10,
        )
        planner = MCTS(env=env, num_simulations=3)
        state = env.reset()
        initial_res = state.causal_graph.resolution_tensor().mean().item()
        done = False
        while not done:
            action = planner.search(state)
            state, _, done, _ = env.step(state, action)
        final_res = state.causal_graph.resolution_tensor().mean().item()
        assert final_res > initial_res, (
            "With positive resolution bias, mean resolution should increase over an episode"
        )
