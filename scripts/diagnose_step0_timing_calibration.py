"""Timing calibration for the 5-candidate aggregation comparison, before
committing to the full run.

Two distinct encoding paths are needed (not five):
  - "standard": ESCState.from_dialogue's normal per-turn/per-cause encoding
    (candidates 1 select, 2 concatenate, 3 max-pool, 5 attention-pool all
    read the SAME per-turn history matrix / per-cause matrix this produces
    -- they only differ in how they aggregate it afterward, which is free
    once the matrix is encoded).
  - "whole-window": candidate 4's alternative -- join the K turns into one
    text block and pool once (same pattern as `emotion`), instead of
    encoding turns separately. A genuinely different set of Qwen calls.

Usage: python -m scripts.diagnose_step0_timing_calibration
"""

from __future__ import annotations

import os
import time

from data import encoder_adapter, iter_jsonl
from esc.state import ESCState
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config

N_CALIB = 50
TARGET_N = 1000
MIN_CONTEXT = 2


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
            instances.append({"context": list(rec.turns[:t]), "gold_strategy": gold_strategy})
            if len(instances) >= n:
                return instances
    return instances


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instances = _build_instances(root, N_CALIB)
    print(f"[calib] built {len(instances)} calibration instances")

    config = load_merged_config(root)
    backbone = QwenBackbone(config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    backbone.load()
    encoder = encoder_adapter(backbone)

    # --- Standard path (feeds candidates 1, 2, 3, 5) ---
    t0 = time.perf_counter()
    for inst in instances:
        ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
    standard_elapsed = time.perf_counter() - t0
    standard_per_instance = standard_elapsed / len(instances)
    print(f"[calib] standard path: {standard_elapsed:.2f}s for {len(instances)} "
          f"-> {standard_per_instance:.4f}s/instance")

    # --- Whole-window path (feeds candidate 4) ---
    t0 = time.perf_counter()
    for inst in instances:
        joined_history = " ".join(inst["context"][-ESCState.K_HISTORY_WINDOW:])
        _ = backbone._proj_history(backbone._pooled_embedding(joined_history))
        seeker_turns = [t for i, t in enumerate(inst["context"]) if i % 2 == 0 and t.strip()]
        candidate_spans = list(reversed(seeker_turns))[: ESCState.N_C]
        joined_causes = " ".join(candidate_spans) if candidate_spans else ""
        _ = backbone._proj_cause(backbone._pooled_embedding(joined_causes))
    whole_window_elapsed = time.perf_counter() - t0
    whole_window_per_instance = whole_window_elapsed / len(instances)
    print(f"[calib] whole-window path: {whole_window_elapsed:.2f}s for {len(instances)} "
          f"-> {whole_window_per_instance:.4f}s/instance")

    total_per_instance = standard_per_instance + whole_window_per_instance
    extrapolated_total_s = total_per_instance * TARGET_N
    print(f"\n[calib] combined (both paths) per instance: {total_per_instance:.4f}s")
    print(f"[calib] extrapolated total for n={TARGET_N}: {extrapolated_total_s:.0f}s "
          f"({extrapolated_total_s / 60:.1f} min, {extrapolated_total_s / 3600:.2f} h)")


if __name__ == "__main__":
    main()
