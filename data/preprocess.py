"""Deterministic cleaning and truncation for :class:`ConversationRecord`."""

from __future__ import annotations

import random
import re
from typing import Any

from esc.action import ESCAction

from data.conversation_schema import ConversationRecord


_WS_RE = re.compile(r"\s+")


# Map raw corpus strings (normalized lower) → ESCAction strategy names
_CANONICAL_STRATEGIES: tuple[str, ...] = tuple(ESCAction.STRATEGIES)

_RAW_TO_INTERNAL: dict[str, str] = {
    "question": "Question",
    "questions": "Question",
    "restatement or paraphrasing": "Restatement",
    "restatement": "Restatement",
    "reflection of feelings": "Reflection",
    "reflection": "Reflection",
    "emotional support": "Reflection",
    "self-disclosure": "Self-disclosure",
    "self disclosure": "Self-disclosure",
    "affirmation and reassurance": "Affirmation",
    "affirmation": "Affirmation",
    "providing suggestions": "Providing Suggestions",
    "suggestion": "Providing Suggestions",
    "information": "Information",
    "others": "Others",
    "other": "Others",
}


def normalize_strategy_label(raw: str | None) -> str | None:
    """Map a dataset-specific strategy string into ``ESCAction.STRATEGIES``."""
    if raw is None:
        return None
    key = raw.strip().lower()
    if not key:
        return None
    mapped = _RAW_TO_INTERNAL.get(key)
    if mapped is not None:
        return mapped
    # Loose contains-match for variants like "Affirmation / Reassurance"
    for raw_k, internal in _RAW_TO_INTERNAL.items():
        if raw_k in key or key in raw_k:
            return internal
    return None


def _strip_turn(text: str) -> str:
    t = text.strip()
    t = _WS_RE.sub(" ", t)
    return t


def preprocess_record(
    record: ConversationRecord,
    *,
    min_turns: int = 2,
    max_turns: int = 60,
    rng: random.Random | None = None,
) -> ConversationRecord | None:
    """
    Clean utterances, optionally enforce alternating roles, truncate length.

    Returns ``None`` if the conversation should be dropped.
    """
    turns_in = record.turns
    roles_in = record.speaker_roles
    strat_list = list(record.annotations.get("turn_strategies") or [])
    # Pad strategies to length
    while len(strat_list) < len(turns_in):
        strat_list.append(None)

    turns: list[str] = []
    roles: list[str] = []
    new_strats: list[str | None] = []

    for t, r, s in zip(turns_in, roles_in, strat_list):
        cleaned = _strip_turn(t)
        if not cleaned:
            continue
        turns.append(cleaned)
        roles.append(r if r in ("seeker", "supporter") else "seeker")
        new_strats.append(s)

    if len(turns) < min_turns:
        return None

    if roles and roles[0] != "seeker":
        # Rotate until first seeker turn when possible
        idx = next((i for i, rr in enumerate(roles) if rr == "seeker"), None)
        if idx is not None and idx > 0:
            turns = turns[idx:]
            roles = roles[idx:]
            new_strats = new_strats[idx:]

    if len(turns) < min_turns:
        return None

    if len(turns) > max_turns:
        turns = turns[:max_turns]
        roles = roles[:max_turns]
        new_strats = new_strats[:max_turns]

    standardized: list[str | None] = [normalize_strategy_label(s) for s in new_strats]

    meta = dict(record.metadata)
    annotations: dict[str, Any] = {
        **record.annotations,
        "turn_strategies_raw": new_strats,
        "turn_strategies": standardized,
    }

    # Optional shuffle-safe noop hook for future augmentations (rng reserved)
    _ = rng

    return ConversationRecord(
        conversation_id=record.conversation_id,
        source=record.source,
        turns=turns,
        speaker_roles=roles,
        annotations=annotations,
        metadata=meta,
    )


def validate_record(record: ConversationRecord, *, min_turns: int = 2) -> bool:
    """Lightweight checks before persisting or encoding."""
    if len(record.turns) < min_turns:
        return False
    if record.source not in ("esconv", "cornell_esc"):
        return False
    if not record.conversation_id:
        return False
    if record.speaker_roles and record.speaker_roles[0] != "seeker":
        return False
    return True
