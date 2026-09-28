"""Build :class:`esc.state.ESCState` from canonical conversation records."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import torch

from esc.state import ESCState, EncoderFn

from data.conversation_schema import ConversationRecord


def deterministic_target_emotion(label: str | None, *, dim: int) -> torch.Tensor:
    """
    Map a stable string (e.g. emotion_type) to a bounded vector for R_emotion.

    Uses SHA256 so the mapping is stable across processes and Python versions.
    """
    if not label:
        return torch.zeros(dim)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        b = digest[i % len(digest)]
        vals.append((float(b) / 255.0) * 2.0 - 1.0)
    return torch.tensor(vals, dtype=torch.float32)


def encoder_adapter(backbone: Any) -> EncoderFn:
    """Wrap a ``QwenBackbone``-like module implementing ``encode_dialogue``."""

    def _enc(turns: list[str]) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
        return backbone.encode_dialogue(turns)

    return _enc


def extract_cause_labels(record: ConversationRecord) -> list[str]:
    """Derive simple textual causes from metadata for graph initialization."""
    causes: list[str] = []
    prob = record.metadata.get("problem_type")
    sit = record.metadata.get("situation")
    if isinstance(prob, str) and prob.strip():
        causes.append(prob.strip())
    if isinstance(sit, str) and sit.strip():
        causes.append(sit.strip())
    if not causes:
        causes.append("unspecified_distress")
    return causes[: ESCState.N_C]


def build_esc_state_from_record(
    record: ConversationRecord,
    *,
    encoder: EncoderFn | None = None,
    target_emotion: torch.Tensor | None = None,
) -> ESCState:
    """
    Construct ``ESCState`` from normalized turns using the ESC encoder hook.

    When ``encoder`` is ``None``, builds the structural placeholder state used
    elsewhere in the codebase. Real offline pipelines should pass an encoder
    (typically ``encoder_adapter(QwenBackbone(...))``).
    """
    turns = list(record.turns)
    if target_emotion is None:
        et = record.metadata.get("emotion_type")
        key = str(et) if et not in (None, "") else ""
        target_emotion = deterministic_target_emotion(key, dim=ESCState.D_E)

    state = ESCState.from_dialogue(
        turns,
        encoder=encoder,
        target_emotion=target_emotion,
    )

    # Attach lightweight provenance for debugging (not part of MDP tensor contract)
    setattr(state, "_conversation_id", record.conversation_id)
    setattr(state, "_source", record.source)
    return state
