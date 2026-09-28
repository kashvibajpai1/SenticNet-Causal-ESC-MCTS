"""Tests for the real Qwen backbone (models/backbone_qwen.py)."""

from __future__ import annotations

import pytest

from esc.state import ESCState
from models.backbone_qwen import QwenBackbone


def test_lazy_init_no_network():
    """Constructing a QwenBackbone must not touch the network/HF hub."""
    backbone = QwenBackbone("Qwen/Qwen2.5-0.5B-Instruct")
    assert backbone._model is None
    assert backbone._tokenizer is None


@pytest.mark.integration
def test_encode_dialogue_shapes():
    backbone = QwenBackbone("Qwen/Qwen2.5-0.5B-Instruct")
    backbone.load()
    result = backbone.encode_dialogue(["I've been feeling really anxious lately.", "That sounds hard. What's been going on?"])
    assert result["history"].shape == (ESCState.K_HISTORY_WINDOW, ESCState.D_H)
    assert result["emotion"].shape == (ESCState.D_E,)
    for cause in result["causes"]:
        assert cause["embedding"].shape == (ESCState.D_C,)
    # real embeddings must not be all-zero (that was the old stub behaviour)
    assert result["history"].abs().sum().item() > 0.0


@pytest.mark.integration
def test_generate_response_nonempty():
    backbone = QwenBackbone("Qwen/Qwen2.5-0.5B-Instruct")
    backbone.load()
    reply = backbone.generate_response("As the Supporter, respond supportively.")
    assert isinstance(reply, str)
    assert reply != "[Qwen-9B stub reply]"
