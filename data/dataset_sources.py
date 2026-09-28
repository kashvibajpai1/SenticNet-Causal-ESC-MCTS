"""Raw dataset loaders: Hugging Face ESConv and ConvoKit emotional-support.

Heavy dependencies are imported **only inside** the functions that need them
(``datasets``, ``convokit``). Importing this module never pulls in Hugging Face
or ConvoKit at interpreter startup.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

from data.conversation_schema import ConversationRecord


SeekerSupporter = Literal["seeker", "supporter"]


def _normalize_role_esconv(speaker: str | None) -> SeekerSupporter | None:
    if speaker is None:
        return None
    s = speaker.strip().lower()
    if s in ("usr", "user", "seeker"):
        return "seeker"
    if s in ("sys", "system", "supporter"):
        return "supporter"
    return None


def _infer_roles_alternating(
    roles: list[SeekerSupporter | None],
) -> list[SeekerSupporter]:
    """Fill None roles by alternating, seeding from first known role."""
    out: list[SeekerSupporter] = []
    cur: SeekerSupporter | None = None
    for r in roles:
        if r is not None:
            cur = r
            out.append(r)
            continue
        if cur is None:
            cur = "seeker"
        else:
            cur = "supporter" if cur == "seeker" else "seeker"
        out.append(cur)
    return out


def esconv_row_to_record(row: dict[str, Any], *, split: str, index: int) -> ConversationRecord:
    """Map one HF ``thu-coai/esconv`` row to :class:`ConversationRecord`.

    The ``thu-coai/esconv`` dataset stores each row as a single JSON-encoded
    string under a ``"text"`` key (not a flat dict with a top-level
    ``"dialog"`` key) -- this decodes that wrapper when present.
    """
    if "text" in row and isinstance(row.get("text"), str):
        import json

        try:
            row = {**row, **json.loads(row["text"])}
        except (json.JSONDecodeError, TypeError):
            pass
    dialog = row.get("dialog") or []
    turns: list[str] = []
    raw_roles: list[SeekerSupporter | None] = []
    strategies: list[str | None] = []

    for turn in dialog:
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("text") or "").strip()
        role = _normalize_role_esconv(turn.get("speaker"))
        strat = turn.get("strategy")
        strat_s = str(strat).strip() if strat not in (None, "") else None

        turns.append(text)
        raw_roles.append(role)
        strategies.append(strat_s)

    speaker_roles_s = _infer_roles_alternating(raw_roles)

    annotations: dict[str, Any] = {
        "turn_strategies": strategies,
        "survey_score": row.get("survey_score"),
    }
    metadata: dict[str, Any] = {
        "split": split,
        "experience_type": row.get("experience_type"),
        "emotion_type": row.get("emotion_type"),
        "problem_type": row.get("problem_type"),
        "situation": row.get("situation"),
    }

    cid = str(row.get("dialog_id") or row.get("id") or f"esconv-{split}-{index}")

    return ConversationRecord(
        conversation_id=cid,
        source="esconv",
        turns=turns,
        speaker_roles=speaker_roles_s,
        annotations=annotations,
        metadata=metadata,
    )


def load_cornell_esc_corpus() -> Any:
    """
    Download (if needed) and return the ConvoKit ``emotional-support`` corpus.

    Performs lazy ``convokit`` imports. Call this to fail fast before long ESConv
    iteration when ConvoKit is misconfigured or missing.
    """
    from convokit import Corpus, download

    return Corpus(filename=download("emotional-support"))


def iter_esconv_records(
    split: str = "train",
    *,
    max_samples: int | None = None,
    streaming: bool = False,
) -> Iterator[ConversationRecord]:
    """
    Yield ESConv conversations from Hugging Face.

    Parameters
    ----------
    split:
        One of ``train``, ``validation``, ``test`` (HF naming).
    max_samples:
        Stop after this many conversations (for debugging).
    streaming:
        Passed through to ``load_dataset`` when supported.
    """
    from datasets import load_dataset  # lazy import

    ds = load_dataset("thu-coai/esconv", split=split, streaming=streaming)
    for i, row in enumerate(ds):
        rec = esconv_row_to_record(dict(row), split=split, index=i)
        yield rec
        if max_samples is not None and i + 1 >= max_samples:
            break


def _utterance_sort_key(utt_id: str) -> tuple[int, str]:
    parts = utt_id.split("_")
    try:
        return int(parts[-1]), utt_id
    except (ValueError, IndexError):
        return 0, utt_id


def convokit_corpus_to_records(
    corpus: Any,
    *,
    max_conversations: int | None = None,
) -> Iterator[ConversationRecord]:
    """Iterate Cornell ESC conversations from an in-memory ConvoKit corpus."""
    n = 0
    for conv in corpus.iter_conversations():
        utts = sorted(conv.iter_utterances(), key=lambda u: _utterance_sort_key(u.id))
        turns: list[str] = []
        speaker_roles: list[str] = []
        strategies: list[str | None] = []

        for utt in utts:
            text = (utt.text or "").strip()
            spk_ref = utt.speaker
            spk_id = spk_ref.id if hasattr(spk_ref, "id") else str(spk_ref)
            speaker = corpus.get_speaker(spk_id)
            role = (speaker.meta or {}).get("role") if speaker is not None else None
            if isinstance(role, str):
                r = role.strip().lower()
                if r not in ("seeker", "supporter"):
                    r = "seeker"
            else:
                r = "seeker"

            ann = utt.meta.get("annotation") if hasattr(utt, "meta") else None
            strat: str | None = None
            if isinstance(ann, dict):
                raw = ann.get("strategy") or ann.get("strategies")
                if isinstance(raw, list) and raw:
                    strat = str(raw[0])
                elif isinstance(raw, str):
                    strat = raw
            elif isinstance(ann, str):
                strat = ann

            turns.append(text)
            speaker_roles.append(r)
            strategies.append(strat)

        meta_conv = conv.meta or {}
        annotations: dict[str, Any] = {"turn_strategies": strategies}
        metadata: dict[str, Any] = {
            "split": None,
            "experience_type": meta_conv.get("experience_type"),
            "emotion_type": meta_conv.get("emotion_type"),
            "problem_type": meta_conv.get("problem_type"),
            "situation": meta_conv.get("situation"),
            "survey_score": meta_conv.get("survey_score"),
        }

        yield ConversationRecord(
            conversation_id=str(conv.id),
            source="cornell_esc",
            turns=turns,
            speaker_roles=speaker_roles,
            annotations=annotations,
            metadata=metadata,
        )

        n += 1
        if max_conversations is not None and n >= max_conversations:
            break


def iter_cornell_esc_records(
    *,
    max_samples: int | None = None,
) -> Iterator[ConversationRecord]:
    """Iterate the ConvoKit ``emotional-support`` corpus (lazy ``convokit`` import)."""
    corpus = load_cornell_esc_corpus()
    yield from convokit_corpus_to_records(corpus, max_conversations=max_samples)
