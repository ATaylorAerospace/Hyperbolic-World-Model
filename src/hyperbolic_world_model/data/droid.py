"""DROID loader: episodes of frames + 7-DoF actions with embodiment/task metadata.

DROID (Khazatsky et al., 2024, https://droid-dataset.github.io) is the primary real-robot source.
We read the RLDS / LeRobot export placed under ``DATA_ROOT/droid`` (see ``data/README.md``); the
dataset is not downloaded by this repository.

Implemented here (pure tensor utilities, unit-testable without data):
    * :func:`normalise_actions` - per-dimension standardisation with stored statistics.
    * :func:`chunk_episode` - sliding windows of ``horizon + 1`` frames.

TODO(phase 2):
    * :class:`DroidDataset` reading episodes lazily from disk and decoding frames.
    * Action statistics computed once over the train split and cached to ``DATA_ROOT/droid/action_stats.pt``.
    * Metadata extraction (language instruction -> task; robot id -> embodiment; skill label ->
      primitive) feeding :mod:`hyperbolic_world_model.data.hierarchies`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ActionStats:
    """Per-dimension mean and std used to normalise actions."""

    mean: Tensor
    std: Tensor

    @classmethod
    def from_actions(cls, actions: Tensor, eps: float = 1e-6) -> ActionStats:
        """Compute stats over all leading dimensions of ``actions`` ``(..., action_dim)``."""
        flat = actions.reshape(-1, actions.shape[-1]).to(torch.float64)
        mean = flat.mean(0)
        # A single row has no sample std (NaN, which clamp_min leaves alone): use unit scale.
        std = flat.std(0).clamp_min(eps) if flat.shape[0] > 1 else torch.ones_like(mean)
        return cls(mean=mean.to(actions.dtype), std=std.to(actions.dtype))


def normalise_actions(actions: Tensor, stats: ActionStats) -> Tensor:
    """``(a - mean) / std`` broadcast over leading dims."""
    return (actions - stats.mean) / stats.std


def chunk_episode(
    frames: Tensor, actions: Tensor, horizon: int, stride: int = 1
) -> list[dict[str, Tensor]]:
    """Cut one episode into overlapping windows of ``horizon + 1`` frames and ``horizon`` actions.

    Args:
        frames: ``(T, C, H, W)``.
        actions: ``(T - 1, action_dim)``; ``actions[t]`` is taken between ``frames[t]`` and ``frames[t+1]``.
        horizon: number of predicted steps per chunk.
        stride: window stride.
    """
    if frames.shape[0] != actions.shape[0] + 1:
        raise ValueError("expected len(frames) == len(actions) + 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    chunks = []
    for start in range(0, frames.shape[0] - horizon, stride):
        chunks.append(
            {
                "frames": frames[start : start + horizon + 1],
                "actions": actions[start : start + horizon],
            }
        )
    return chunks


class DroidDataset(Dataset):
    """Windowed DROID episodes.

    TODO(phase 2): implement ``__init__`` (index episodes under ``root``), ``__len__`` and
    ``__getitem__`` (decode frames, apply :func:`normalise_actions`, attach metadata).
    """

    def __init__(self, root: Path, split: str, horizon: int, stride: int, image_size: int) -> None:
        raise NotImplementedError(
            "DroidDataset is not implemented yet (phase 2). Use data=synthetic for smoke runs; "
            "see the module docstring in data/droid.py for the plan."
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], split: str = "train") -> DroidDataset:
        return cls(
            root=Path(cfg["root"]),
            split=split,
            horizon=int(cfg["horizon"]),
            stride=int(cfg.get("stride", 1)),
            image_size=int(cfg.get("image_size", 224)),
        )


__all__ = ["ActionStats", "DroidDataset", "chunk_episode", "normalise_actions"]
