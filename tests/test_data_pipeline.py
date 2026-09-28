"""Smoke tests for ESC dataset layer (no Hugging Face / ConvoKit downloads)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from esc.state import ESCState

from data import (
    build_esc_state_from_record,
    esc_state_from_bundle,
    esc_state_to_bundle,
    esconv_row_to_record,
    preprocess_record,
    save_state_split,
    validate_record,
    write_jsonl,
)
from data.conversation_schema import ConversationRecord
from train.train_data import ESCStateBundleDataset, ESCStateTensorDataset


def test_esconv_row_normalization() -> None:
    row = {
        "dialog": [
            {"text": " hi ", "speaker": "usr"},
            {"text": "hello", "speaker": "sys", "strategy": "Question"},
        ],
        "emotion_type": "sadness",
        "problem_type": "job crisis",
    }
    rec = esconv_row_to_record(row, split="train", index=0)
    pr = preprocess_record(rec)
    assert pr is not None
    assert validate_record(pr)
    assert pr.turns[0] == "hi"
    assert pr.metadata["split"] == "train"


def test_preprocess_drops_short_dialogues() -> None:
    rec = ConversationRecord(
        conversation_id="x",
        source="esconv",
        turns=["only"],
        speaker_roles=["seeker"],
        annotations={},
        metadata={},
    )
    assert preprocess_record(rec, min_turns=2) is None


def test_build_esc_state_and_resolution_length() -> None:
    rec = ConversationRecord(
        conversation_id="c1",
        source="esconv",
        turns=["I feel lost.", "Tell me more.", "Work is hard.", "I understand."],
        speaker_roles=["seeker", "supporter", "seeker", "supporter"],
        annotations={"turn_strategies": [None, "Question", None, "Reflection"]},
        metadata={"emotion_type": "anxiety", "split": "train"},
    )
    pr = preprocess_record(rec)
    assert pr is not None
    state = build_esc_state_from_record(pr, encoder=None)
    vec = state.to_tensor()
    assert vec.shape == (ESCState.get_state_dim(),)
    assert state.causal_graph.resolution_tensor().numel() == ESCState.N_C


def test_state_bundle_roundtrip() -> None:
    rec = ConversationRecord(
        conversation_id="c2",
        source="esconv",
        turns=["a", "b", "c", "d"],
        speaker_roles=["seeker", "supporter", "seeker", "supporter"],
        annotations={},
        metadata={"emotion_type": None},
    )
    pr = preprocess_record(rec)
    assert pr is not None
    state = build_esc_state_from_record(pr, encoder=None)
    bundle = esc_state_to_bundle(state)
    restored = esc_state_from_bundle(bundle)
    assert torch.allclose(state.to_tensor(), restored.to_tensor())


def test_jsonl_write_read() -> None:
    rec = ConversationRecord(
        conversation_id="j1",
        source="esconv",
        turns=["x", "y"],
        speaker_roles=["seeker", "supporter"],
        annotations={},
        metadata={"split": "train"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.jsonl")
        write_jsonl(path, iter([rec]))
        line = open(path, encoding="utf-8").read().strip()
        obj = json.loads(line)
        back = ConversationRecord.from_json_obj(obj)
        assert back.conversation_id == "j1"


def test_pytorch_datasets_from_split_file() -> None:
    rec = ConversationRecord(
        conversation_id="p1",
        source="esconv",
        turns=["u1", "u2", "u3", "u4"],
        speaker_roles=["seeker", "supporter", "seeker", "supporter"],
        annotations={},
        metadata={},
    )
    pr = preprocess_record(rec)
    assert pr is not None
    state = build_esc_state_from_record(pr, encoder=None)
    bundle = esc_state_to_bundle(state)
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = os.path.join(tmp, "train.pt")
        save_state_split(
            pt_path,
            bundles=[bundle],
            state_tensors=state.to_tensor().unsqueeze(0),
        )
        tds = ESCStateTensorDataset(pt_path)
        assert len(tds) == 1
        assert tds[0].shape == (ESCState.get_state_dim(),)
        bds = ESCStateBundleDataset(pt_path)
        assert len(bds) == 1
        assert bds.get_state(0).turn_index == state.turn_index


@pytest.mark.integration
def test_download_esconv_single_sample() -> None:
    datasets = pytest.importorskip("datasets")
    _ = datasets
    from data.dataset_sources import iter_esconv_records

    rec = next(iter_esconv_records("train", max_samples=1))
    assert rec.source == "esconv"
    assert len(rec.turns) >= 2
    pr = preprocess_record(rec)
    assert pr is not None
    assert validate_record(pr)
