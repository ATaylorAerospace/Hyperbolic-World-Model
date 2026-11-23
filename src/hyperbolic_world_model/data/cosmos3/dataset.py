"""Loader for Cosmos-generated trajectories (``manifest.jsonl``), in the shared batch format.

Each item is one window of ``horizon + 1`` frames and ``horizon`` actions from one rollout::

    {"frames": float(T, 3, H, W) in [0, 1], "actions": float(T - 1, a), "meta": {...}}

with ``meta`` carrying ``prompt_id``, ``branch_id``, ``embodiment``, ``task``, ``primitive``, the
window offset and, when extracted, ``latents_path``. Frames are resized to ``image_size`` with
bilinear interpolation (torch only).

Actions are standardised per dimension, as the shared batch format promises, with statistics
computed once over the rollouts of the **train** split and applied to every split
(:attr:`Cosmos3TrajectoryDataset.action_stats`). The stored rows are the raw layout Cosmos was
conditioned on (``droid_lerobot``: position deltas, 6-D rotation deltas and a gripper value),
whose dimensions are on different scales. ``normalise_actions=False`` returns the raw rows.

What the two tasks need is exposed directly:

* :meth:`Cosmos3TrajectoryDataset.branches` groups item indices by ``prompt_id``. Every branch of a
  prompt shares its start frame, so the long-horizon consistency task can pair windows at offset 0
  across branches and measure how latent divergence grows with horizon.
* :meth:`Cosmos3TrajectoryDataset.split_by_combination` returns (seen, held-out) index lists for a
  set of ``(embodiment, primitive)`` pairs, which is the compositional-generalisation split.
* :meth:`Cosmos3TrajectoryDataset.latents` loads the tokenizer latents of a rollout for the
  delta-hyperbolicity probe.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from hyperbolic_world_model.data.cosmos3 import default_root
from hyperbolic_world_model.data.cosmos3.generate import read_manifest
from hyperbolic_world_model.data.droid import ActionStats, normalise_actions
from hyperbolic_world_model.data.splits import COMBINATION_KEYS, split_by_combination

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    rollout: int  # index into the manifest records
    start: int  # first frame index of the window


class Cosmos3TrajectoryDataset(Dataset):
    """Windowed generated trajectories with branching structure.

    Args:
        root: generated data root containing ``manifest.jsonl``.
        horizon: predicted steps per window (``horizon + 1`` frames). ``None`` uses whole
            rollouts truncated to the shortest one (one window each), so items still stack.
        stride: window stride in frames.
        image_size: output ``(H, W)`` or a single int; ``None`` keeps the stored resolution.
        split: ``"train"``, ``"val"``, ``"test"`` or ``"all"``; rollouts are assigned to splits by a
            deterministic hash of ``prompt_id`` with ``split_fractions``.
        split_fractions: ``(train, val)`` fractions; the rest is ``test``.
        cache: keep decoded rollouts in memory (fine for a few thousand short rollouts).
        normalise_actions: standardise actions with the train split's per-dimension mean and
            std (every rollout's, if the train split is empty); ``False`` returns the raw rows.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        horizon: int | None = 8,
        stride: int = 1,
        image_size: int | tuple[int, int] | None = None,
        split: str = "all",
        split_fractions: tuple[float, float] = (0.8, 0.1),
        cache: bool = True,
        normalise_actions: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.records: list[dict[str, Any]] = read_manifest(self.root)
        if not self.records:
            raise FileNotFoundError(f"no manifest.jsonl under {self.root}; run generate.py first")
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be >= 1 or None")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.horizon = horizon
        self.stride = int(stride)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.split = split
        self.cache = cache
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

        keep = self._split_indices(split, split_fractions)
        if not keep:
            log.warning(
                "Cosmos3TrajectoryDataset split=%r is empty under %s (%d rollouts in the manifest; "
                "the split is a hash of prompt_id, so tiny datasets can lose a split entirely)",
                split,
                self.root,
                len(self.records),
            )
        if horizon is None:
            # Whole rollouts, truncated to the shortest one in the *manifest* (not the split), so
            # every item and batch stacks and train/val/test items all have the same horizon.
            lengths = [int(r["num_frames"]) for r in self.records]
            self.effective_horizon = min(lengths) - 1
        else:
            self.effective_horizon = int(horizon)
        h = self.effective_horizon
        self.windows: list[Window] = []
        for i in keep:
            n = int(self.records[i]["num_frames"])
            if n < h + 1:
                continue  # rollout too short for this horizon
            starts = [0] if horizon is None else range(0, n - h, self.stride)
            for start in starts:
                self.windows.append(Window(i, start))
        self.action_dim = int(self.records[0]["action_dim"])
        #: Per-dimension mean and std applied to every action, or ``None`` for raw actions.
        self.action_stats: ActionStats | None = (
            self._compute_action_stats(split_fractions) if normalise_actions else None
        )

    # ------------------------------------------------------------------ splits and groups
    def _split_indices(self, split: str, fractions: tuple[float, float]) -> list[int]:
        if split == "all":
            return list(range(len(self.records)))
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, test or all")
        import zlib

        out = []
        for i, r in enumerate(self.records):
            u = (zlib.crc32(str(r["prompt_id"]).encode()) % 10_000) / 10_000
            name = (
                "train"
                if u < fractions[0]
                else "val"
                if u < fractions[0] + fractions[1]
                else "test"
            )
            if name == split:
                out.append(i)
        return out

    def _compute_action_stats(self, fractions: tuple[float, float]) -> ActionStats:
        """Mean and std of the train split's actions, whatever split this instance serves.

        Train, val and test must share one normalisation and the statistics must not look at
        held-out rollouts, so they always come from the train rollouts of the manifest. Only the
        ``actions`` member of each ``frames.npz`` is read (frames stay compressed on disk).
        """
        rollouts = self._split_indices("train", fractions) or list(range(len(self.records)))
        rows = []
        for i in rollouts:
            with np.load(self.root / self.records[i]["frames_path"]) as z:
                rows.append(torch.from_numpy(z["actions"]).float())
        return ActionStats.from_actions(torch.cat(rows))

    def branches(self, start_only: bool = True) -> dict[str, list[int]]:
        """``prompt_id -> item indices`` of its branches (windows at offset 0 by default)."""
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, w in enumerate(self.windows):
            if start_only and w.start != 0:
                continue
            groups[str(self.records[w.rollout]["prompt_id"])].append(idx)
        return dict(groups)

    def branch_pairs(self) -> list[tuple[int, int]]:
        """All unordered pairs of start windows that share a start frame (for divergence curves)."""
        pairs = []
        for idxs in self.branches().values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pairs.append((idxs[a], idxs[b]))
        return pairs

    def combination(self, idx: int) -> tuple[str, str]:
        r = self.records[self.windows[idx].rollout]
        return str(r["embodiment"]), str(r["primitive"])

    def combinations(self) -> set[tuple[str, str]]:
        return {self.combination(i) for i in range(len(self))}

    def split_by_combination(self, holdout: Sequence[Sequence[str]]) -> tuple[list[int], list[int]]:
        """``(seen, held_out)`` item indices; ``holdout`` is a list of ``[embodiment, primitive]`` pairs.

        Delegates to :func:`~hyperbolic_world_model.data.splits.split_by_combination`, which raises
        if a held-out combination never occurs or if holding it out removes an embodiment or a
        primitive entirely.
        """
        combos = [
            dict(zip(COMBINATION_KEYS, self.combination(i), strict=True)) for i in range(len(self))
        ]
        return split_by_combination(combos, holdout)

    def metadata(self) -> list[dict[str, str]]:
        """Per-item ``{embodiment, task, primitive}`` for :func:`data.hierarchies.build_hierarchy`."""
        return [
            {k: str(self.records[w.rollout][k]) for k in ("embodiment", "task", "primitive")}
            for w in self.windows
        ]

    # ------------------------------------------------------------------ loading
    def _load_rollout(self, rollout: int) -> tuple[Tensor, Tensor]:
        if rollout in self._cache:
            return self._cache[rollout]
        rec = self.records[rollout]
        with np.load(self.root / rec["frames_path"]) as z:
            frames = (
                torch.from_numpy(z["frames"]).permute(0, 3, 1, 2).float() / 255.0
            )  # (T, 3, H, W)
            actions = torch.from_numpy(z["actions"]).float()
        if self.action_stats is not None:
            actions = normalise_actions(actions, self.action_stats)
        if self.image_size is not None and tuple(frames.shape[-2:]) != tuple(self.image_size):
            frames = F.interpolate(
                frames,
                size=tuple(self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.cache:
            self._cache[rollout] = (frames, actions)
        return frames, actions

    def latents(self, idx: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Tokenizer latents ``(T', D)`` of the rollout behind item ``idx`` (requires extract_latents)."""
        from hyperbolic_world_model.data.cosmos3.extract_latents import load_latents

        rec = self.records[self.windows[idx].rollout]
        if not rec.get("latents_path"):
            raise FileNotFoundError(
                f"no latents for {rec['prompt_id']}/{rec['branch_id']}; run extract_latents.py"
            )
        return load_latents(self.root / rec["latents_path"])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        w = self.windows[idx]
        frames, actions = self._load_rollout(w.rollout)
        h = self.effective_horizon
        rec = self.records[w.rollout]
        meta = {
            "prompt_id": str(rec["prompt_id"]),
            "branch_id": str(rec["branch_id"]),
            "embodiment": str(rec["embodiment"]),
            "task": str(rec["task"]),
            "primitive": str(rec["primitive"]),
            "start": w.start,
            "latents_path": rec.get("latents_path"),
        }
        return {
            "frames": frames[w.start : w.start + h + 1],
            "actions": actions[w.start : w.start + h],
            "meta": meta,
        }

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], split: str = "train") -> Cosmos3TrajectoryDataset:
        size = cfg.get("image_size")
        return cls(
            root=cfg.get("root"),
            horizon=None if cfg.get("horizon") in (None, "null") else int(cfg["horizon"]),
            stride=int(cfg.get("stride", 1)),
            image_size=None
            if size is None
            else (int(size) if isinstance(size, int) else tuple(int(s) for s in size)),
            split=split,
            split_fractions=tuple(cfg.get("split_fractions", (0.8, 0.1))),
            cache=bool(cfg.get("cache", True)),
            normalise_actions=bool(cfg.get("normalise_actions", True)),
        )

    def __repr__(self) -> str:
        return f"Cosmos3TrajectoryDataset(root={self.root}, split={self.split}, rollouts={len(self.records)}, windows={len(self)})"


def load_manifest_as_json(root: Path) -> str:
    """Convenience for reports: the manifest as a JSON array string."""
    return json.dumps(read_manifest(root))


__all__ = ["Cosmos3TrajectoryDataset", "Window", "load_manifest_as_json"]
