"""Collate functions for PyTorch dataloaders."""

from __future__ import annotations

import torch


def collate_state_tensors(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack flat ESC state vectors shaped ``[state_dim]`` → ``[B, state_dim]``."""
    return torch.stack(batch, dim=0)


def collate_bundles(batch: list[dict]) -> list[dict]:
    """Passthrough collate for reconstruction outside default tensor stacking."""
    return batch
