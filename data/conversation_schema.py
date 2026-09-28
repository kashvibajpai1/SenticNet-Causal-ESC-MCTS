"""Canonical conversation record shared by ESConv and Cornell ESC corpora."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ConversationRecord:
    """Unified dialogue record after source loaders normalize raw corpora."""

    conversation_id: str
    source: str
    turns: list[str]
    speaker_roles: list[str]
    annotations: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.turns) != len(self.speaker_roles):
            raise ValueError(
                "turns and speaker_roles must have the same length "
                f"({len(self.turns)} vs {len(self.speaker_roles)})"
            )

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-serializable dict for JSONL artifacts."""
        return asdict(self)

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> ConversationRecord:
        return cls(
            conversation_id=str(obj["conversation_id"]),
            source=str(obj["source"]),
            turns=list(obj["turns"]),
            speaker_roles=list(obj["speaker_roles"]),
            annotations=dict(obj.get("annotations") or {}),
            metadata=dict(obj.get("metadata") or {}),
        )
