"""Step 4 diagnostic: histogram the trained value network's post-ReLU outputs
across the real 150-instance eval set (seed=42 checkpoint), to check for a
spike at exactly 0.0 (evidence of the reward-sign-mismatch risk) vs a healthy
spread. Read-only: loads a checkpoint, no training, no writes to checkpoints/.
"""

from __future__ import annotations

import json
import os

import torch

from data import encoder_adapter
from esc.state import ESCState
from eval.data import build_eval_instances
from models.backbone_qwen import QwenBackbone
from models.value import ValueNetwork
from train.utils import load_merged_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    config = load_merged_config(ROOT)
    state_dim = ESCState.get_state_dim()

    ckpt_dir = os.path.join(ROOT, "checkpoints", "causal_mcts", "seed42")
    value = ValueNetwork(state_dim=state_dim)
    value.load_state_dict(torch.load(os.path.join(ckpt_dir, "value.pt"), map_location="cpu"))
    value.eval()

    backbone_model_name = config.get("backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    backbone = QwenBackbone(backbone_model_name, seed=42)
    backbone.load()
    encoder = encoder_adapter(backbone)

    instances = build_eval_instances(ROOT, n=150)
    print(f"[diagnose_step4] n_instances={len(instances)}")

    outputs: list[float] = []
    with torch.no_grad():
        for inst in instances:
            state = ESCState.from_dialogue(inst["context"], encoder=encoder, target_emotion=None)
            v = float(value(state.to_tensor()).item())
            outputs.append(v)

    t = torch.tensor(outputs)
    n_exact_zero = int((t == 0.0).sum().item())
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, float("inf"))]
    hist = []
    for lo, hi in bins:
        count = int(((t >= lo) & (t < hi)).sum().item())
        hist.append({"range": f"[{lo}, {hi if hi != float('inf') else 'inf'})", "count": count})

    summary = {
        "checkpoint": ckpt_dir,
        "n_instances": len(outputs),
        "min": float(t.min()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "median": float(t.median()),
        "max": float(t.max()),
        "n_exact_zero": n_exact_zero,
        "pct_exact_zero": n_exact_zero / len(outputs),
        "histogram": hist,
        "raw_outputs": outputs,
    }

    out_dir = os.path.join(ROOT, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "relu_output_histogram.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[diagnose_step4] wrote {out_path}")
    print(
        f"[diagnose_step4] min={summary['min']:.4f} mean={summary['mean']:.4f} "
        f"std={summary['std']:.4f} median={summary['median']:.4f} max={summary['max']:.4f}"
    )
    print(f"[diagnose_step4] exact-0.0 count={n_exact_zero} ({summary['pct_exact_zero']*100:.1f}%)")
    for b in hist:
        bar = "#" * b["count"]
        print(f"[diagnose_step4] {b['range']:>14}: {b['count']:4d} {bar}")


if __name__ == "__main__":
    main()
