"""
Episode logger for ESC experiments.

Records per-step and per-episode data in a structured format that can be
exported to JSON or converted to a pandas DataFrame for analysis.

Usage
-----
    from utils.logging import EpisodeLogger

    logger = EpisodeLogger()
    logger.start_episode(episode_id=0)

    for turn in range(max_horizon):
        action = mcts.search(state)
        next_state, reward, done, info = env.step(state, action)
        logger.log_step(action=action, reward=reward, info=info)
        if done:
            break

    logger.end_episode(final_state=next_state)

    summary = logger.episode_summary(-1)   # last episode
    logger.save("experiments/run_001_log.json")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import torch

from esc.state import ESCState
from esc.action import ESCAction


@dataclass
class StepRecord:
    """Data recorded at each environment step."""
    turn: int
    strategy_id: int
    strategy_name: str
    cause_index: int
    reward: float
    R_cause: float
    R_emotion: float
    R_phase: float
    phase: int
    resolution: list[float]


@dataclass
class EpisodeRecord:
    """Data recorded across a full episode."""
    episode_id: int
    start_time: float
    end_time: float = 0.0
    total_reward: float = 0.0
    num_turns: int = 0
    final_phase: int = 0
    final_resolution: list[float] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def mean_reward_per_turn(self) -> float:
        return self.total_reward / max(self.num_turns, 1)


class EpisodeLogger:
    """
    Structured logger for ESC MDP episodes.

    Thread-safety: not thread-safe; use one logger per process/experiment.
    """

    def __init__(self) -> None:
        self._episodes: list[EpisodeRecord] = []
        self._current: Optional[EpisodeRecord] = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, episode_id: int) -> None:
        """
        Begin recording a new episode.

        Parameters
        ----------
        episode_id : Unique identifier for this episode
        """
        if self._current is not None:
            raise RuntimeError(
                "start_episode() called while an episode is already active.  "
                "Call end_episode() first."
            )
        self._current = EpisodeRecord(
            episode_id=episode_id,
            start_time=time.time(),
        )

    def log_step(
        self,
        action: ESCAction,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """
        Record one environment step.

        Parameters
        ----------
        action : ESCAction taken this step
        reward : Scalar reward received
        info   : Info dict from env.step()
        """
        if self._current is None:
            raise RuntimeError(
                "log_step() called outside an episode.  "
                "Call start_episode() first."
            )
        step = StepRecord(
            turn=info.get("turn", -1),
            strategy_id=action.strategy_id,
            strategy_name=ESCAction.STRATEGIES[action.strategy_id],
            cause_index=action.cause_index,
            reward=reward,
            R_cause=info.get("R_cause", 0.0),
            R_emotion=info.get("R_emotion", 0.0),
            R_phase=info.get("R_phase", 0.0),
            phase=info.get("phase", 0),
            resolution=info.get("resolution", []),
        )
        self._current.steps.append(step)
        self._current.total_reward += reward
        self._current.num_turns += 1

    def end_episode(self, final_state: ESCState) -> EpisodeRecord:
        """
        Finalise the current episode and store the record.

        Parameters
        ----------
        final_state : Last ESCState of the episode

        Returns
        -------
        Completed EpisodeRecord
        """
        if self._current is None:
            raise RuntimeError("end_episode() called without a matching start_episode()")

        self._current.end_time = time.time()
        self._current.final_phase = final_state.current_phase
        self._current.final_resolution = (
            final_state.causal_graph.resolution_tensor().tolist()
        )
        record = self._current
        self._episodes.append(record)
        self._current = None
        return record

    # ------------------------------------------------------------------
    # Summaries and persistence
    # ------------------------------------------------------------------

    def episode_summary(self, episode_idx: int = -1) -> dict[str, Any]:
        """
        Return a summary dict for the episode at the given list index.

        Parameters
        ----------
        episode_idx : Python list index (-1 = last episode)
        """
        rec = self._episodes[episode_idx]
        strategy_counts: dict[str, int] = {}
        for step in rec.steps:
            strategy_counts[step.strategy_name] = (
                strategy_counts.get(step.strategy_name, 0) + 1
            )
        return {
            "episode_id": rec.episode_id,
            "total_reward": round(rec.total_reward, 4),
            "num_turns": rec.num_turns,
            "mean_reward_per_turn": round(rec.mean_reward_per_turn, 4),
            "duration_s": round(rec.duration_seconds, 3),
            "final_phase": rec.final_phase,
            "final_resolution": [round(r, 3) for r in rec.final_resolution],
            "strategy_counts": strategy_counts,
        }

    def all_summaries(self) -> list[dict[str, Any]]:
        """Return summaries for every recorded episode."""
        return [self.episode_summary(i) for i in range(len(self._episodes))]

    def save(self, path: str) -> None:
        """
        Persist all episode records to a JSON file.

        Parameters
        ----------
        path : Output file path (directories are created if needed)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = []
        for rec in self._episodes:
            d = asdict(rec)
            data.append(d)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EpisodeLogger":
        """
        Reconstruct a logger from a saved JSON file (read-only replay).

        Parameters
        ----------
        path : Path to the JSON log file

        Returns
        -------
        EpisodeLogger with all episodes populated (no active episode)
        """
        logger = cls()
        with open(path) as f:
            data = json.load(f)
        for d in data:
            steps = [StepRecord(**s) for s in d.pop("steps")]
            rec = EpisodeRecord(**d)
            rec.steps = steps
            logger._episodes.append(rec)
        return logger

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodeLogger(episodes={len(self._episodes)})"
