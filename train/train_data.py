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
    """Reads flat ``state_tensors`` from ``save_state_split`` output."""

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        payload = load_state_split(path)
        st = payload.get("state_tensors")
        if st is None:
            bundles = payload.get("bundles") or []
            if not bundles:
                self._tensors = torch.zeros(0, ESCState.get_state_dim())
            else:
                rows = [esc_state_from_bundle(b).to_tensor() for b in bundles]
                self._tensors = torch.stack(rows, dim=0)
        else:
            self._tensors = st

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
