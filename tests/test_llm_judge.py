"""Tests for eval/llm_judge.py's pure logic (no network calls / no API key needed).

Covers prompt construction, response parsing, the bias-flip winner mapping,
and run_pairing() with an injected fake judge -- exercises Part C's design
requirements (randomized A/B position, ties allowed, judge never told which
system is which) without calling a real LLM.
"""

from __future__ import annotations

from eval.llm_judge import (
    build_user_prompt,
    parse_judge_response,
    resolve_winner_system,
    run_pairing,
)


def test_build_user_prompt_labels_speakers_and_replies():
    prompt = build_user_prompt(["I feel awful", "That sounds hard"], "reply A text", "reply B text")
    assert "Seeker: I feel awful" in prompt
    assert "Supporter: That sounds hard" in prompt
    assert "Candidate reply A: reply A text" in prompt
    assert "Candidate reply B: reply B text" in prompt


def test_parse_judge_response_valid_json():
    text = '{"winner": "A", "reasoning": "More empathetic."}'
    parsed = parse_judge_response(text)
    assert parsed == {"winner": "A", "reasoning": "More empathetic."}


def test_parse_judge_response_tie():
    text = '{"winner": "tie", "reasoning": "Equally good."}'
    assert parse_judge_response(text)["winner"] == "tie"


def test_parse_judge_response_malformed_falls_back_to_tie():
    parsed = parse_judge_response("not json at all")
    assert parsed["winner"] == "tie"
    assert "unparsed judge output" in parsed["reasoning"]


def test_parse_judge_response_invalid_winner_value_falls_back_to_tie():
    text = '{"winner": "C", "reasoning": "??"}'
    assert parse_judge_response(text)["winner"] == "tie"


def test_resolve_winner_system_not_flipped():
    # Not flipped: judge's "A" really is system_a.
    assert resolve_winner_system("A", "causal_mcts", "random_floor", flipped=False) == "causal_mcts"
    assert resolve_winner_system("B", "causal_mcts", "random_floor", flipped=False) == "random_floor"


def test_resolve_winner_system_flipped():
    # Flipped: judge's "A" is really system_b, and vice versa.
    assert resolve_winner_system("A", "causal_mcts", "random_floor", flipped=True) == "random_floor"
    assert resolve_winner_system("B", "causal_mcts", "random_floor", flipped=True) == "causal_mcts"


def test_resolve_winner_system_tie_is_tie_regardless_of_flip():
    assert resolve_winner_system("tie", "a", "b", flipped=False) == "tie"
    assert resolve_winner_system("tie", "a", "b", flipped=True) == "tie"


def test_run_pairing_never_reveals_system_identity_to_judge():
    """The injected judge function must only ever see reply text, never
    which system produced it -- this is the bias-control requirement."""
    seen_replies = []

    def fake_judge(context, reply_a, reply_b):
        seen_replies.append((reply_a, reply_b))
        return {"winner": "A", "reasoning": "ok"}

    instances = [{"conversation_id": f"c{i}", "context": ["hi"]} for i in range(6)]
    preds_a = [f"causal_reply_{i}" for i in range(6)]
    preds_b = [f"random_reply_{i}" for i in range(6)]

    run_pairing(
        fake_judge, "causal_mcts", "random_floor", instances, preds_a, preds_b, seed=0
    )

    # fake_judge only ever receives (reply_a, reply_b) text -- no system name.
    for shown_a, shown_b in seen_replies:
        assert shown_a in preds_a + preds_b
        assert shown_b in preds_a + preds_b


def test_run_pairing_position_is_randomized_across_instances():
    """With a fixed seed, not every instance should show the same system
    in slot A -- i.e. position bias control is actually exercised."""

    def fake_judge(context, reply_a, reply_b):
        return {"winner": "A", "reasoning": "ok"}

    instances = [{"conversation_id": f"c{i}", "context": ["hi"]} for i in range(30)]
    preds_a = [f"causal_reply_{i}" for i in range(30)]
    preds_b = [f"random_reply_{i}" for i in range(30)]

    result = run_pairing(
        fake_judge, "causal_mcts", "random_floor", instances, preds_a, preds_b, seed=0
    )

    first_slots = [t["reply_a_system"] for t in result["transcripts"]]
    assert len(set(first_slots)) > 1, "position should vary across instances, not be fixed"


def test_run_pairing_counts_wins_ties_and_writes_transcripts():
    call_count = {"n": 0}

    def fake_judge(context, reply_a, reply_b):
        call_count["n"] += 1
        # Alternate between A, B, and tie.
        return {"winner": ["A", "B", "tie"][call_count["n"] % 3], "reasoning": "r"}

    instances = [{"conversation_id": f"c{i}", "context": ["hi"]} for i in range(9)]
    preds_a = [f"a{i}" for i in range(9)]
    preds_b = [f"b{i}" for i in range(9)]

    result = run_pairing(
        fake_judge, "causal_mcts", "flow_ablation", instances, preds_a, preds_b,
        seed=1, n_transcripts=3,
    )

    assert result["pairing"] == "causal_mcts_vs_flow_ablation"
    assert result["n_instances"] == 9
    assert sum(result["wins"].values()) == 9
    assert set(result["wins"].keys()) == {"causal_mcts", "flow_ablation", "tie"}
    assert len(result["transcripts"]) == 3
    for t in result["transcripts"]:
        assert "judge_reasoning" in t
        assert t["judge_winner_system"] in {"causal_mcts", "flow_ablation", "tie"}


def test_run_pairing_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        run_pairing(
            lambda *a: {"winner": "tie", "reasoning": ""},
            "a",
            "b",
            [{"conversation_id": "c0", "context": []}],
            ["x", "y"],
            ["z"],
            seed=0,
        )
