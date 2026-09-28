"""Serialize ESC states and JSONL conversation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch

from esc.causal_graph import CausalGraph
from esc.state import ESCState

from data.conversation_schema import ConversationRecord


def write_jsonl(path: str | Path, records: Iterator[ConversationRecord]) -> int:
    """Append one JSON object per line; returns number of records written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json_obj(), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[ConversationRecord]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield ConversationRecord.from_json_obj(obj)


def causal_graph_to_bundle(graph: CausalGraph) -> dict[str, Any]:
    n = graph.num_causes
    labels = [graph.get_node(i).label for i in range(n)]
    embeddings = torch.stack([graph.get_node(i).embedding.clone() for i in range(n)])
    resolutions = graph.resolution_tensor().clone()
    edges: list[list[int]] = []
    for i in range(n):
        for dst in graph.get_edges(i):
            edges.append([i, dst])
    return {
        "labels": labels,
        "embeddings": embeddings,
        "resolutions": resolutions,
        "edges": edges,
        "d_c": graph.d_c,
    }


def causal_graph_from_bundle(bundle: dict[str, Any]) -> CausalGraph:
    d_c = int(bundle["d_c"])
    g = CausalGraph(d_c=d_c)
    labels: list[str] = list(bundle["labels"])
    embeddings: torch.Tensor = bundle["embeddings"]
    for lab, emb in zip(labels, embeddings):
        g.add_cause(label=str(lab), embedding=emb.clone())
    res = bundle["resolutions"]
    if res.shape[0] == g.num_causes:
        g.update_resolution(res)
    for pair in bundle.get("edges") or []:
        src, dst = int(pair[0]), int(pair[1])
        try:
            g.add_edge(src, dst)
        except (IndexError, ValueError):
            pass
    return g


def esc_state_to_bundle(state: ESCState) -> dict[str, Any]:
    return {
        "history_embeddings": state.history_embeddings.detach().cpu().clone(),
        "emotion_vector": state.emotion_vector.detach().cpu().clone(),
        "phase_embedding": state.phase_embedding.detach().cpu().clone(),
        "turn_index": int(state.turn_index),
        "target_emotion": state.target_emotion.detach().cpu().clone(),
        "causal_graph": causal_graph_to_bundle(state.causal_graph),
        "conversation_id": getattr(state, "_conversation_id", ""),
        "source": getattr(state, "_source", ""),
    }


def esc_state_from_bundle(bundle: dict[str, Any]) -> ESCState:
    cg = causal_graph_from_bundle(bundle["causal_graph"])
    state = ESCState(
        history_embeddings=bundle["history_embeddings"].clone(),
        causal_graph=cg,
        emotion_vector=bundle["emotion_vector"].clone(),
        phase_embedding=bundle["phase_embedding"].clone(),
        turn_index=int(bundle["turn_index"]),
        target_emotion=bundle["target_emotion"].clone(),
    )
    cid = bundle.get("conversation_id")
    src = bundle.get("source")
    if cid is not None:
        setattr(state, "_conversation_id", cid)
    if src is not None:
        setattr(state, "_source", src)
    return state


def save_state_split(
    path: str | Path,
    *,
    bundles: list[dict[str, Any]],
    state_tensors: torch.Tensor | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"bundles": bundles}
    if state_tensors is not None:
        payload["state_tensors"] = state_tensors
    torch.save(payload, p)


def load_state_split(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
