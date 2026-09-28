"""Dataset and encoder hooks for future ESC dialogue corpora (no I/O yet)."""

from __future__ import annotations

from typing import Any

import torch

from models.backbone_qwen import QwenBackbone


class ESCDialogueEncoder:
    """Thin wrapper around ``QwenBackbone.encode_dialogue`` for training / data pipes."""

    def __init__(self, backbone: QwenBackbone) -> None:
        self._backbone = backbone

    def encode(self, turns: list[str]) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
        return self._backbone.encode_dialogue(turns)


class ESCConversationDataset:
    """
    Placeholder for real ESC trajectory loading (ESConv, SenticNet internal, etc.).

    Parameters
    ----------
    data_path : Root path or manifest for the corpus (unused in the stub).
    encoder   : Dialogue encoder used when real samples are loaded.
    """

    def __init__(self, data_path: str, encoder: ESCDialogueEncoder) -> None:
        _ = data_path
        self._encoder = encoder

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Any:
        _ = index
        raise NotImplementedError(
            "This class will be used to load real ESC trajectories "
            "(e.g., ESConv or Sentic internal ESC data). For now, it is a stub."
        )
