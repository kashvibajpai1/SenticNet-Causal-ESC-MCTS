"""Five-candidate state-aggregation comparison (cheap proxy, not a retrain).

Compares five ways of reducing ESCState's per-turn history matrix [K,D_H]
and per-cause matrix [N_C,D_C] into the state vector the policy/value
networks see:

  1. select       -- H[-1], C[0]                       (prior result: 21.7% @ n=240)
  2. concatenate  -- flatten(H), flatten(C)             (prior result: 16.7% @ n=240)
  3. max-pool     -- max(H, dim=0), max(C, dim=0)       (new)
  4. whole-window -- join K turns into one text, pool
                     once via Qwen (same pattern as
                     `emotion`), instead of encoding
                     turns separately then combining     (new; separate encoding pass)
  5. attn-pool    -- learned query vector per group,
                     softmax-weighted sum; trained
                     JOINTLY with the probe classifier    (new; has real parameters)

Candidates 1/2/3/5 share ONE encoding pass (they all read the same raw
per-turn/per-cause embeddings, only differing in aggregation). Candidate 4
needs a second, genuinely different encoding pass. Calibrated via
scripts/diagnose_step0_timing_calibration.py before running this at scale.

Usage: python -m scripts.diagnose_step2c_five_candidate_comparison
"""

from __future__ import annotations

import collections
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

from data import encoder_adapter, iter_jsonl
from esc.action import ESCAction
from esc.state import ESCState
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config

N_INSTANCES = 1000
MIN_CONTEXT = 2
TRAIN_FRAC = 0.8
PROBE_EPOCHS = 300
PROBE_LR = 0.05
SEED = 0


def _build_instances(root: str, n: int) -> list[dict]:
    jsonl_path = os.path.join(root, "artifacts", "processed", "conversations.jsonl")
    instances: list[dict] = []
    for rec in iter_jsonl(jsonl_path):
        if rec.metadata.get("split") != "train":
            continue
        strategies = rec.annotations.get("turn_strategies") or []
        for t in range(MIN_CONTEXT, len(rec.turns)):
            if t >= len(rec.speaker_roles) or rec.speaker_roles[t] != "supporter":
                continue
            gold_strategy = strategies[t] if t < len(strategies) else None
            if gold_strategy is None:
                continue
            instances.append(
                {"context": list(rec.turns[:t]), "gold_strategy": gold_strategy}
            )
            if len(instances) >= n:
                return instances
    return instances


def _train_linear_probe(x_train, y_train, x_test, y_test, num_classes, epochs=PROBE_EPOCHS, lr=PROBE_LR):
    torch.manual_seed(SEED)
    dim = x_train.shape[1]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train_n = (x_train - mean) / std
    x_test_n = (x_test - mean) / std
    clf = nn.Linear(dim, num_classes)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(clf(x_train_n), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = float((clf(x_test_n).argmax(dim=-1) == y_test).float().mean().item())
    n_params = sum(p.numel() for p in clf.parameters())
    return acc, n_params


class AttentionPool(nn.Module):
    """One learned query vector per group; softmax-weighted sum over items."""

    def __init__(self, d_h: int, d_c: int) -> None:
        super().__init__()
        self.history_query = nn.Parameter(torch.randn(d_h) * 0.01)
        self.cause_query = nn.Parameter(torch.randn(d_c) * 0.01)

    def forward(self, history_batch: torch.Tensor, cause_batch: torch.Tensor) -> torch.Tensor:
        # history_batch: [B, K, D_H], cause_batch: [B, N_C, D_C]
        h_scores = history_batch @ self.history_query  # [B, K]
        h_weights = F.softmax(h_scores, dim=-1).unsqueeze(-1)  # [B, K, 1]
        h_pooled = (history_batch * h_weights).sum(dim=1)  # [B, D_H]

        c_scores = cause_batch @ self.cause_query  # [B, N_C]
        c_weights = F.softmax(c_scores, dim=-1).unsqueeze(-1)
        c_pooled = (cause_batch * c_weights).sum(dim=1)  # [B, D_C]
        return torch.cat([h_pooled, c_pooled], dim=-1)  # [B, D_H+D_C]


def _train_attention_probe(
    h_train, c_train, emo_train, phase_train, y_train,
    h_test, c_test, emo_test, phase_test, y_test,
    num_classes, d_h, d_c, epochs=PROBE_EPOCHS, lr=PROBE_LR,
):
    torch.manual_seed(SEED)
    pool = AttentionPool(d_h, d_c)
    clf = nn.Linear(d_h + d_c + emo_train.shape[1] + phase_train.shape[1], num_classes)
    params = list(pool.parameters()) + list(clf.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    # Standardize the non-attention components using train stats (emotion/phase);
    # attention output isn't standardized since it's jointly learned.
    for step in range(epochs):
        optimizer.zero_grad()
        pooled_train = pool(h_train, c_train)
        x_train = torch.cat([pooled_train, emo_train, phase_train], dim=-1)
        loss = F.cross_entropy(clf(x_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pooled_test = pool(h_test, c_test)
        x_test = torch.cat([pooled_test, emo_test, phase_test], dim=-1)
        acc = float((clf(x_test).argmax(dim=-1) == y_test).float().mean().item())

    n_params = sum(p.numel() for p in params)
    n_extra_params = sum(p.numel() for p in pool.parameters())
    return acc, n_params, n_extra_params, epochs


def _pairwise_stats(vectors: torch.Tensor) -> dict[str, float]:
    normed = F.normalize(vectors, dim=1)
    sim = normed @ normed.T
    n = sim.shape[0]
    off_diag = sim[~torch.eye(n, dtype=torch.bool)]
    return {"mean": float(off_diag.mean()), "min": float(off_diag.min()), "max": float(off_diag.max())}


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_start = time.perf_counter()

    instances = _build_instances(root, N_INSTANCES)
    print(f"[step2c] built {len(instances)} instances from train split")

    strategy_to_idx = {s: i for i, s in enumerate(ESCAction.STRATEGIES)}
    labels = [strategy_to_idx[i["gold_strategy"]] for i in instances]
    label_counts = collections.Counter(labels)
    print("[step2c] label distribution:", {ESCAction.STRATEGIES[k]: v for k, v in label_counts.items()})

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    D_H, D_C, D_E = ESCState.D_H, ESCState.D_C, ESCState.D_E
    K, N_C = ESCState.K_HISTORY_WINDOW, ESCState.N_C

    history_mats, cause_mats, emotions, phases = [], [], [], []
    whole_window_vecs = []

    t0 = time.perf_counter()
    for i, inst in enumerate(instances):
        state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
        history_mats.append(state.history_embeddings)                                  # [K, D_H]
        c_matrix = state.causal_graph.cause_embedding_matrix(N_C, D_C)
        cause_mats.append(c_matrix)                                                    # [N_C, D_C]
        emotions.append(state.emotion_vector)
        phases.append(state.phase_embedding)

        joined_history = " ".join(inst["context"][-K:])
        wh = backbone._proj_history(backbone._pooled_embedding(joined_history))
        seeker_turns = [t for j, t in enumerate(inst["context"]) if j % 2 == 0 and t.strip()]
        candidate_spans = list(reversed(seeker_turns))[:N_C]
        joined_causes = " ".join(candidate_spans) if candidate_spans else ""
        wc = backbone._proj_cause(backbone._pooled_embedding(joined_causes))
        whole_window_vecs.append(torch.cat([wh, wc, state.emotion_vector, state.phase_embedding], dim=0))

        if (i + 1) % 100 == 0:
            print(f"[step2c] encoded {i + 1}/{len(instances)}  "
                  f"({(time.perf_counter() - t0) / (i + 1):.3f}s/instance so far)")

    encode_elapsed = time.perf_counter() - t0
    print(f"[step2c] encoding done in {encode_elapsed:.1f}s ({encode_elapsed/60:.1f} min)")

    history_batch = torch.stack(history_mats)   # [N, K, D_H]
    cause_batch = torch.stack(cause_mats)       # [N, N_C, D_C]
    emo_batch = torch.stack(emotions)           # [N, D_E]
    phase_batch = torch.stack(phases)           # [N, 3]
    whole_window_x = torch.stack(whole_window_vecs)  # [N, D_H+D_C+D_E+3]
    y = torch.tensor(labels, dtype=torch.long)

    n = len(instances)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * TRAIN_FRAC)
    train_idx = torch.tensor(idx[:n_train])
    test_idx = torch.tensor(idx[n_train:])
    num_classes = len(ESCAction.STRATEGIES)
    random_floor = 1.0 / num_classes
    majority_floor = max(collections.Counter(labels[i] for i in idx[n_train:]).values()) / len(test_idx)

    results = {}

    # --- Candidate 1: select ---
    select_x = torch.cat([history_batch[:, -1, :], cause_batch[:, 0, :], emo_batch, phase_batch], dim=-1)
    acc, n_params = _train_linear_probe(
        select_x[train_idx], y[train_idx], select_x[test_idx], y[test_idx], num_classes
    )
    sim = _pairwise_stats(select_x[:200])
    results["select"] = {"probe_accuracy": acc, "dim": select_x.shape[1], "cosine": sim,
                          "classifier_params": n_params, "needs_ongoing_training": False}

    # --- Candidate 2: concatenate ---
    concat_x = torch.cat(
        [history_batch.reshape(n, -1), cause_batch.reshape(n, -1), emo_batch, phase_batch], dim=-1
    )
    acc, n_params = _train_linear_probe(
        concat_x[train_idx], y[train_idx], concat_x[test_idx], y[test_idx], num_classes
    )
    sim = _pairwise_stats(concat_x[:200])
    results["concatenate"] = {"probe_accuracy": acc, "dim": concat_x.shape[1], "cosine": sim,
                               "classifier_params": n_params, "needs_ongoing_training": False}

    # --- Candidate 3: max-pool ---
    maxpool_x = torch.cat(
        [history_batch.max(dim=1).values, cause_batch.max(dim=1).values, emo_batch, phase_batch], dim=-1
    )
    acc, n_params = _train_linear_probe(
        maxpool_x[train_idx], y[train_idx], maxpool_x[test_idx], y[test_idx], num_classes
    )
    sim = _pairwise_stats(maxpool_x[:200])
    results["max_pool"] = {"probe_accuracy": acc, "dim": maxpool_x.shape[1], "cosine": sim,
                            "classifier_params": n_params, "needs_ongoing_training": False}

    # --- Candidate 4: whole-window-through-Qwen ---
    acc, n_params = _train_linear_probe(
        whole_window_x[train_idx], y[train_idx], whole_window_x[test_idx], y[test_idx], num_classes
    )
    sim = _pairwise_stats(whole_window_x[:200])
    results["whole_window"] = {"probe_accuracy": acc, "dim": whole_window_x.shape[1], "cosine": sim,
                                "classifier_params": n_params, "needs_ongoing_training": False}

    # --- Candidate 5: attention-pool (trained jointly with classifier) ---
    acc, n_params, n_extra, epochs_used = _train_attention_probe(
        history_batch[train_idx], cause_batch[train_idx], emo_batch[train_idx], phase_batch[train_idx], y[train_idx],
        history_batch[test_idx], cause_batch[test_idx], emo_batch[test_idx], phase_batch[test_idx], y[test_idx],
        num_classes, D_H, D_C,
    )
    with torch.no_grad():
        torch.manual_seed(SEED)
        probe_pool = AttentionPool(D_H, D_C)
        attn_x_sample = torch.cat(
            [probe_pool(history_batch[:200], cause_batch[:200]), emo_batch[:200], phase_batch[:200]], dim=-1
        )
    sim = _pairwise_stats(attn_x_sample)
    results["attention_pool"] = {
        "probe_accuracy": acc,
        "dim": D_H + D_C + D_E + 3,
        "cosine": sim,
        "total_params": n_params,
        "extra_params_vs_others": n_extra,
        "training_epochs": epochs_used,
        "needs_ongoing_training": True,
    }

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[step2c] TOTAL wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"[step2c] n_train={len(train_idx)} n_test={len(test_idx)} num_classes={num_classes}")
    print(f"[step2c] random floor: {random_floor:.4f}  majority floor: {majority_floor:.4f}\n")

    for name, r in results.items():
        print(f"[step2c] {name}: acc={r['probe_accuracy']:.4f} dim={r['dim']} "
              f"cosine(mean/min)={r['cosine']['mean']:.4f}/{r['cosine']['min']:.4f}")

    out = {
        "n_instances": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "num_classes": num_classes,
        "random_floor": random_floor,
        "majority_floor": majority_floor,
        "results": results,
        "total_wall_seconds": total_elapsed,
    }
    out_path = os.path.join(root, "results", "diagnostics", "step2c_five_candidate_comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[step2c] wrote {out_path}")


if __name__ == "__main__":
    main()
