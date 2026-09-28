"""Trainer entry points for FlowMCTS-ablation and causal MCTS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from train.trainer_causal_mcts import CausalMCTSTrainer
    from train.trainer_flow_ablation import FlowAblationTrainer

__all__ = ["CausalMCTSTrainer", "FlowAblationTrainer"]


def __getattr__(name: str) -> Any:
    if name == "FlowAblationTrainer":
        from train.trainer_flow_ablation import FlowAblationTrainer

        return FlowAblationTrainer
    if name == "CausalMCTSTrainer":
        from train.trainer_causal_mcts import CausalMCTSTrainer

        return CausalMCTSTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
