"""
Reproducibility utilities: seed management and experiment configuration.

Usage
-----
At the top of any script or test that must be reproducible:

    from utils.seed import set_global_seed
    set_global_seed(42)

For per-experiment reproducibility with full config logging:

    from utils.seed import ExperimentConfig
    cfg = ExperimentConfig(seed=42, num_simulations=50, max_horizon=20)
    cfg.apply()          # sets all seeds
    cfg.log()            # prints config to stdout
    cfg.save("run.json") # persists config for later reproduction
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional
import torch


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python, PyTorch (CPU and CUDA), and the OS.

    This covers all sources of randomness in the pipeline:
    - torch.manual_seed: CPU tensor ops
    - torch.cuda.manual_seed_all: all CUDA devices
    - random.seed: Python random module
    - PYTHONHASHSEED: hash randomisation (must be set before interpreter start,
      but setting here ensures the value is at least logged)

    Parameters
    ----------
    seed : Non-negative integer seed value
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slight performance cost; worth it for research)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]

    # Log environment variable (cannot set it retroactively, but record it)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class ExperimentConfig:
    """
    Full experiment configuration for reproducible research runs.

    Captures every hyperparameter that affects experimental results so
    that any run can be exactly reproduced from the saved JSON.

    Parameters
    ----------
    seed            : Global RNG seed
    num_simulations : MCTS simulation budget per step
    max_horizon     : Maximum conversation turns per episode
    c_puct          : MCTS exploration constant
    hidden_dim      : Neural network hidden layer width
    alpha           : Reward weight for R_cause
    beta            : Reward weight for R_emotion
    gamma           : Reward weight for R_phase
    device          : Torch device ("cpu" or "cuda")
    notes           : Free-text notes about the experiment
    """
    seed: int = 42
    num_simulations: int = 50
    max_horizon: int = 20
    c_puct: float = 1.0
    hidden_dim: int = 256
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    device: str = "cpu"
    notes: str = ""

    def apply(self) -> None:
        """
        Apply this config: set all global seeds.

        Call this before constructing any models or environments.
        """
        set_global_seed(self.seed)

    def save(self, path: str) -> None:
        """
        Persist config to a JSON file.

        Parameters
        ----------
        path : File path (e.g. "experiments/run_001.json")
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """
        Load config from a JSON file.

        Parameters
        ----------
        path : Path to the config JSON file

        Returns
        -------
        ExperimentConfig instance
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def log(self) -> None:
        """Print config to stdout in a readable format."""
        print("=" * 60)
        print("Experiment Configuration")
        print("=" * 60)
        for k, v in asdict(self).items():
            print(f"  {k:<20}: {v}")
        print("=" * 60)
