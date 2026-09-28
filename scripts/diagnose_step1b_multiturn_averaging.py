"""Follow-up to Step 1: does averaging over multiple turns/cause-spans in
encode_dialogue() (not raw pooling itself) wash out the per-conversation
signal?

Step 1 found single-turn raw/projected embeddings are NOT badly collapsed
(mean cosine ~0.6, min near 0) on genuinely diverse turns -- contradicting
the ~0.97-0.998 similarity seen on actual pipeline state vectors. This
script builds the REAL encode_dialogue()-produced state vectors (same
multi-turn averaging the production pipeline does) for the same diverse
conversations, to see whether averaging over up to 5 history turns and up
to 4 cause spans is what compounds into the observed collapse.

Usage: python -m scripts.diagnose_step1b_multiturn_averaging
"""

from __future__ import annotations

import json
import os

import torch

from data import encoder_adapter
from esc.state import ESCState
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config

N_SAMPLES = 25


def _select_diverse_conversations(jsonl_path: str, n: int) -> list[tuple[str, list[str]]]:
    seen_combos: set[tuple[str, str]] = set()
    picked: list[tuple[str, list[str]]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            combo = (meta.get("emotion_type", ""), meta.get("problem_type", ""))
            if combo in seen_combos:
                continue
            turns = rec.get("turns", [])
            if len(turns) < 4:
                continue
            seen_combos.add(combo)
            label = f"{combo[0]}/{combo[1]}"
            # Use a real multi-turn context, same as training/eval would see
            # for a mid-conversation instance (not just the opening turn).
            context = turns[: min(len(turns), 8)]
            picked.append((label, context))
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
    picked = _select_diverse_conversations(jsonl_path, N_SAMPLES)
    print(f"[step1b] selected {len(picked)} multi-turn contexts (up to 8 turns each)")

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    history_pooled = []
    cause_pooled = []
    full_state = []
    for label, context in picked:
        state = ESCState.from_dialogue(context, encoder=encoder, target_emotion=None)
        history_pooled.append(state.history_embeddings.mean(dim=0))
        c_matrix = state.causal_graph.cause_embedding_matrix(ESCState.N_C, ESCState.D_C)
        cause_pooled.append(c_matrix.mean(dim=0))
        full_state.append(state.to_tensor())

    history_stack = torch.stack(history_pooled)
    cause_stack = torch.stack(cause_pooled)
    full_stack = torch.stack(full_state)

    history_stats = _pairwise_stats(history_stack)
    cause_stats = _pairwise_stats(cause_stack)
    full_stats = _pairwise_stats(full_stack)

    print("\n[step1b] mean-pooled HISTORY component (H_bar, dominant chunk) pairwise cosine:")
    print(f"  mean={history_stats['mean']:.4f} min={history_stats['min']:.4f} max={history_stats['max']:.4f}")
    print("[step1b] mean-pooled CAUSE component (C_bar) pairwise cosine:")
    print(f"  mean={cause_stats['mean']:.4f} min={cause_stats['min']:.4f} max={cause_stats['max']:.4f}")
    print("[step1b] FULL state vector (to_tensor()) pairwise cosine:")
    print(f"  mean={full_stats['mean']:.4f} min={full_stats['min']:.4f} max={full_stats['max']:.4f}")

    out = {
        "n_samples": len(picked),
        "history_pairwise_cosine": history_stats,
        "cause_pairwise_cosine": cause_stats,
        "full_state_pairwise_cosine": full_stats,
    }
    out_path = os.path.join(root, "results", "diagnostics", "step1b_multiturn_averaging.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[step1b] wrote {out_path}")


if __name__ == "__main__":
    main()
