"""PyTorch datasets over serialized ESC artifacts (no Hugging Face / ConvoKit)."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset

from esc.state import ESCState

from data.serialization import esc_state_from_bundle, load_state_split

__all__ = ["ESCStateBundleDataset", "ESCStateTensorDataset"]


class ESCStateTensorDataset(Dataset[torch.Tensor]):
    """Flat state tensors, computed fresh from ``save_state_split``'s bundles.

    2026-09-15: this used to prefer a precomputed ``state_tensors`` array
    from the payload when present, falling back to live ``to_tensor()``
    calls only if absent. That shortcut went silently stale the moment
    to_tensor()'s aggregation changed (found while debugging the state
    representation-collapse investigation -- `run_flow_ablation.py` uses
    this class, and `artifacts/states/*.pt` already had a `state_tensors`
    key baked in under the old mean-pooling code, so flow_ablation training
    would have kept reading pre-fix tensors indefinitely with no error).
    Removed entirely rather than invalidating/rebuilding the cached key --
    to_tensor()'s aggregation is still being actively compared across
    several candidate methods, so any one-time cache rebuild would just go
    stale again at the next change. Bundles store the pre-aggregation
    components (history matrix, causal graph), which don't change when
    to_tensor()'s aggregation does, so recomputing here is cheap (pure
    tensor arithmetic on already-encoded embeddings, no Qwen calls) and
    can never go stale again.
    """

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        payload = load_state_split(path)
        bundles = payload.get("bundles") or []
        if not bundles:
            self._tensors = torch.zeros(0, ESCState.get_state_dim())
        else:
            rows = [esc_state_from_bundle(b).to_tensor() for b in bundles]
            self._tensors = torch.stack(rows, dim=0)

    def __len__(self) -> int:
        return int(self._tensors.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._tensors[index]


class ESCStateBundleDataset(Dataset[dict[str, Any]]):
    """Expose raw bundles for environments that need full ``ESCState`` objects."""

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        payload = load_state_split(path)
        self._bundles: list[dict[str, Any]] = list(payload.get("bundles") or [])

    def __len__(self) -> int:
        return len(self._bundles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._bundles[index]

    def get_state(self, index: int) -> ESCState:
        return esc_state_from_bundle(self._bundles[index])
