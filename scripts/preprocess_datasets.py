"""Offline preprocessing: ESConv + Cornell → JSONL conversations + state tensors."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from collections import defaultdict

import torch

from data import (
    build_esc_state_from_record,
    convokit_corpus_to_records,
    encoder_adapter,
    esc_state_to_bundle,
    iter_esconv_records,
    load_cornell_esc_corpus,
    preprocess_record,
    save_state_split,
    validate_record,
    write_jsonl,
)
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config
from utils.seed import set_global_seed


def _assign_cornell_split(conversation_id: str, seed: int) -> str:
    h = int(
        hashlib.sha256(f"{seed}:{conversation_id}".encode()).hexdigest()[:12],
        16,
    )
    r = h % 100
    if r < 80:
        return "train"
    if r < 90:
        return "valid"
    return "test"


def _map_esconv_split(split: str) -> str:
    return "valid" if split == "validation" else split


def _convokit_failure_banner(exc: BaseException) -> str:
    return (
        "\n"
        "======================================================================\n"
        "ERROR: ConvoKit (Cornell emotional-support corpus) failed to load.\n"
        "Cornell ESC inclusion is the recommended default for EMNLP-style runs.\n"
        "----------------------------------------------------------------------\n"
        "Fix:\n"
        "  1. Install full data deps:  pip install -r requirements-data-full.txt\n"
        "  2. Use Python 3.10+ when possible (ConvoKit pulls spaCy-compatible stacks).\n"
        "  3. Ensure network access on first run (downloads corpus cache).\n"
        "----------------------------------------------------------------------\n"
        "To run HF ESConv only (no Cornell):  --skip-cornell\n"
        "======================================================================\n"
        f"Underlying exception ({type(exc).__name__}): {exc}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Recommended pipeline for benchmarks: install requirements-data-full.txt, "
            "then run this script without --skip-cornell so both ESConv (HF) and Cornell ESC "
            "(ConvoKit) contribute (~2.3k normalized dialogs combined—sources overlap). "
            "Use --skip-cornell only for lightweight HF-only smoke tests."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-esconv", type=int, default=None)
    parser.add_argument("--max-cornell", type=int, default=None)
    parser.add_argument("--skip-esconv", action="store_true", help="Skip Hugging Face ESConv.")
    parser.add_argument(
        "--skip-cornell",
        action="store_true",
        help=(
            "HF-only: skip Cornell ESC (ConvoKit). Use only for lightweight smoke runs; "
            "full benchmarks should install requirements-data-full.txt and omit this flag."
        ),
    )
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=60)
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    processed_dir = os.path.join(root, "artifacts", "processed")
    states_dir = os.path.join(root, "artifacts", "states")
    jsonl_path = os.path.join(processed_dir, "conversations.jsonl")

    set_global_seed(args.seed)
    rng = random.Random(args.seed)
    records_out: list = []

    cornell_corpus = None
    if not args.skip_cornell:
        try:
            cornell_corpus = load_cornell_esc_corpus()
        except Exception as e:
            print(_convokit_failure_banner(e), file=sys.stderr)
            return 1

    esconv_cap = args.max_esconv

    if not args.skip_esconv:
        # Cap is applied PER HF split (not globally): a global running total
        # would exhaust entirely within "train" (the largest split) and never
        # touch "validation"/"test", leaving downstream eval with no test
        # instances. Each split independently gets up to `esconv_cap` records.
        for split_name in ("train", "validation", "test"):
            split_seen = 0
            mapped = _map_esconv_split(split_name)
            try:
                gen = iter_esconv_records(split_name, max_samples=None)
            except Exception as e:
                print(
                    f"[preprocess] ESConv unavailable ({e}); install `datasets` or pass --skip-esconv.",
                    file=sys.stderr,
                )
                break
            for rec in gen:
                if esconv_cap is not None and split_seen >= esconv_cap:
                    break
                pr = preprocess_record(
                    rec,
                    min_turns=args.min_turns,
                    max_turns=args.max_turns,
                    rng=rng,
                )
                if pr is None or not validate_record(pr):
                    continue
                pr.metadata["split"] = mapped
                records_out.append(pr)
                split_seen += 1

    if cornell_corpus is not None:
        for rec in convokit_corpus_to_records(
            cornell_corpus,
            max_conversations=args.max_cornell,
        ):
            pr = preprocess_record(
                rec,
                min_turns=args.min_turns,
                max_turns=args.max_turns,
                rng=rng,
            )
            if pr is None or not validate_record(pr):
                continue
            pr.metadata["split"] = _assign_cornell_split(pr.conversation_id, args.seed)
            records_out.append(pr)

    write_jsonl(jsonl_path, iter(records_out))
    print(f"[preprocess] wrote {len(records_out)} records → {jsonl_path}")

    backbone_model_name = load_merged_config(root).get(
        "backbone_model_name", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    backbone = QwenBackbone(backbone_model_name, seed=args.seed)
    backbone.load()
    encoder = encoder_adapter(backbone)

    by_split: dict[str, list] = defaultdict(list)
    for rec in records_out:
        sp = rec.metadata.get("split") or "train"
        by_split[sp].append(rec)

    for split_key, filename in (
        ("train", "train.pt"),
        ("valid", "valid.pt"),
        ("test", "test.pt"),
    ):
        chunk = by_split.get(split_key, [])
        bundles = []
        tensors = []
        for rec in chunk:
            state = build_esc_state_from_record(rec, encoder=encoder)
            bundles.append(esc_state_to_bundle(state))
            tensors.append(state.to_tensor().detach().cpu())
        st = torch.stack(tensors, dim=0) if tensors else None
        out_path = os.path.join(states_dir, filename)
        save_state_split(out_path, bundles=bundles, state_tensors=st)
        print(f"[preprocess] split={split_key} n={len(bundles)} → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
