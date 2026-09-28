"""Validate processed JSONL and serialized state splits."""

from __future__ import annotations

import argparse
import os
import sys

from esc.state import ESCState

from data import esc_state_from_bundle, iter_jsonl, load_state_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to conversations.jsonl (default: artifacts/processed/conversations.jsonl).",
    )
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    jsonl_path = args.jsonl or os.path.join(
        root, "artifacts", "processed", "conversations.jsonl"
    )
    states_train = os.path.join(root, "artifacts", "states", "train.pt")

    if os.path.isfile(jsonl_path):
        n = 0
        for rec in iter_jsonl(jsonl_path):
            n += 1
            assert len(rec.turns) == len(rec.speaker_roles)
            if rec.speaker_roles and rec.speaker_roles[0] != "seeker":
                print(f"[validate] warn: first role not seeker: {rec.conversation_id}", file=sys.stderr)
        print(f"[validate] JSONL ok: {n} records at {jsonl_path}")
    else:
        print(f"[validate] skip JSONL (missing): {jsonl_path}")

    if os.path.isfile(states_train):
        payload = load_state_split(states_train)
        bundles = payload.get("bundles") or []
        st = payload.get("state_tensors")
        if st is not None:
            assert st.dim() == 2
            assert st.shape[1] == ESCState.get_state_dim(), (
                f"state dim mismatch {st.shape[1]} vs {ESCState.get_state_dim()}"
            )
        for i, b in enumerate(bundles[:3]):
            state = esc_state_from_bundle(b)
            vec = state.to_tensor()
            assert vec.shape == (ESCState.get_state_dim(),)
            res = state.causal_graph.resolution_tensor()
            assert res.numel() == ESCState.N_C, (
                f"resolution length {res.numel()} expected {ESCState.N_C}"
            )
        print(f"[validate] train.pt ok: n={len(bundles)} tensors={st is not None}")
    else:
        print(f"[validate] skip train.pt (missing): {states_train}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
