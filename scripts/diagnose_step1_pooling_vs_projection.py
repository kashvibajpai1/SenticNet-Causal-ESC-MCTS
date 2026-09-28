"""Debug spec Step 1: is the ~0.97-0.998 state-vector similarity already
present in Qwen's raw pooled output, or introduced/worsened by the frozen
random projection?

Takes ~25 real ESConv turns chosen for genuinely different topics/emotions
(one per conversation, selected to maximize distinct emotion_type /
problem_type combinations from the real preprocessed corpus -- not just
different phases), computes each turn's raw pooled embedding
(QwenBackbone._pooled_embedding) and its projected embedding
(_proj_history applied to that raw embedding), and reports the full
pairwise cosine similarity matrix's mean/min/max for both.

Usage: python -m scripts.diagnose_step1_pooling_vs_projection
"""

from __future__ import annotations

import json
import os

import torch

from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config

N_SAMPLES = 25


def _select_diverse_turns(jsonl_path: str, n: int) -> list[tuple[str, str]]:
    """Pick n (label, text) pairs from conversations with maximally
    distinct (emotion_type, problem_type) combinations, one seeker turn
    (the conversation's opening turn) per conversation."""
    seen_combos: set[tuple[str, str]] = set()
    picked: list[tuple[str, str]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            combo = (meta.get("emotion_type", ""), meta.get("problem_type", ""))
            if combo in seen_combos:
                continue
            turns = rec.get("turns", [])
            if not turns or not turns[0].strip():
                continue
            seen_combos.add(combo)
            label = f"{combo[0]}/{combo[1]}"
            picked.append((label, turns[0]))
            if len(picked) >= n:
                break
    return picked


def _pairwise_stats(vectors: torch.Tensor) -> dict[str, float]:
    normed = torch.nn.functional.normalize(vectors, dim=1)
    sim = normed @ normed.T
    n = sim.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool)
    off_diag = sim[off_diag_mask]
    return {
        "mean": float(off_diag.mean().item()),
        "min": float(off_diag.min().item()),
        "max": float(off_diag.max().item()),
    }


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsonl_path = os.path.join(root, "artifacts", "processed", "conversations.jsonl")

    picked = _select_diverse_turns(jsonl_path, N_SAMPLES)
    print(f"[step1] selected {len(picked)} turns spanning distinct emotion/problem combos")
    for label, text in picked:
        print(f"  - {label}: {text[:70]!r}")

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()

    raw_vecs = []
    proj_vecs = []
    for _label, text in picked:
        raw = backbone._pooled_embedding(text)
        proj = backbone._proj_history(raw)
        raw_vecs.append(raw)
        proj_vecs.append(proj)

    raw_stack = torch.stack(raw_vecs)
    proj_stack = torch.stack(proj_vecs)

    print(f"\n[step1] raw pooled embedding dim: {raw_stack.shape[1]}")
    print(f"[step1] projected embedding dim: {proj_stack.shape[1]}")

    raw_stats = _pairwise_stats(raw_stack)
    proj_stats = _pairwise_stats(proj_stack)

    print("\n[step1] RAW pooled embedding pairwise cosine similarity:")
    print(f"  mean={raw_stats['mean']:.4f}  min={raw_stats['min']:.4f}  max={raw_stats['max']:.4f}")
    print("[step1] PROJECTED embedding pairwise cosine similarity:")
    print(f"  mean={proj_stats['mean']:.4f}  min={proj_stats['min']:.4f}  max={proj_stats['max']:.4f}")

    out = {
        "n_samples": len(picked),
        "labels": [label for label, _ in picked],
        "raw_hidden_dim": raw_stack.shape[1],
        "proj_dim": proj_stack.shape[1],
        "raw_pairwise_cosine": raw_stats,
        "proj_pairwise_cosine": proj_stats,
    }
    out_path = os.path.join(root, "results", "diagnostics", "step1_pooling_vs_projection.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[step1] wrote {out_path}")


if __name__ == "__main__":
    main()
