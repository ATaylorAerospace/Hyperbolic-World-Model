"""Synthetic action-conditioned trajectories for the CPU smoke experiment.

A hidden state ``h_t in R^k`` evolves as ``h_{t+1} = h_t + B a_t`` with a small fixed nonlinearity,
and frames are a fixed linear rendering of ``h_t`` into ``(C, H, W)`` plus noise. Because both the
renderer and :class:`~hyperbolic_world_model.models.encoders.synthetic.SyntheticEncoder` are linear,
the composed latent dynamics are learnable by a small head, so a smoke run can show error going
down, which is the property CI checks. Metadata carries a two-level hierarchy (two embodiments x
two primitives) so hierarchy tasks have something to chew on.

With ``n_branches > 1`` consecutive episodes form a prompt: they share the initial hidden state
and the rendered start frame and differ only in their actions, which is the branching structure
the long-horizon consistency task needs (:meth:`SyntheticTrajectoryDataset.branch_pairs`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticSpec:
    n_episodes: int = 64
    horizon: int = 8
    state_dim: int = 8
    action_dim: int = 4
    frame_shape: tuple[int, int, int] = (1, 8, 8)
    noise_std: float = 0.01
    seed: int = 0
    n_branches: int = 1  # episodes per prompt sharing the start state and start frame


class SyntheticTrajectoryDataset(Dataset):
    """Deterministic in-memory trajectories; see module docstring."""

    def __init__(self, spec: SyntheticSpec, split: str = "train") -> None:
        self.spec = spec
        seed = spec.seed + {"train": 0, "val": 1, "test": 2}[split]
        gen = torch.Generator().manual_seed(seed)
        k, a = spec.state_dim, spec.action_dim
        c, h, w = spec.frame_shape
        # Shared dynamics/rendering (same across splits): seeded from spec.seed only.
        shared = torch.Generator().manual_seed(spec.seed)
        self.B = torch.randn(a, k, generator=shared) / a**0.5
        self.render = torch.randn(k, c * h * w, generator=shared) / k**0.5

        if spec.n_branches < 1 or spec.n_episodes % spec.n_branches:
            raise ValueError("n_branches must be >= 1 and divide n_episodes")
        n_prompts = spec.n_episodes // spec.n_branches
        h0 = torch.randn(n_prompts, k, generator=gen).repeat_interleave(spec.n_branches, dim=0)
        states = [h0]
        actions = torch.randn(spec.n_episodes, spec.horizon, a, generator=gen)
        for t in range(spec.horizon):
            s = states[-1]
            s_next = s + actions[:, t] @ self.B - 0.05 * torch.tanh(s)
            states.append(s_next)
        st = torch.stack(states, dim=1)  # (N, T+1, k)
        frames = st @ self.render + spec.noise_std * torch.randn(
            spec.n_episodes, spec.horizon + 1, c * h * w, generator=gen
        )
        frames = torch.sigmoid(frames).reshape(spec.n_episodes, spec.horizon + 1, c, h, w)
        # Branches of a prompt share the start frame exactly (same state, same noise draw).
        first = frames[:: spec.n_branches, 0].repeat_interleave(spec.n_branches, dim=0)
        frames[:, 0] = first

        self.frames: Tensor = frames
        self.actions: Tensor = actions
        self.states: Tensor = st
        emb = ["arm_a", "arm_b"]
        prim = ["reach", "grasp"]
        # Embodiment and task describe the scene, so they are prompt-level and shared by every
        # branch of a prompt (as in the Cosmos data); only the primitive varies per branch.
        # Otherwise a held-out (embodiment, task) combination could share its start frame and
        # hidden state with a sibling branch in the seen set.
        self.meta = [
            {
                "embodiment": emb[p % 2],
                "task": f"task_{(p // 2) % 4}",
                "primitive": prim[(p // 8 + b) % 2],
                "prompt_id": f"p{p}",
                "branch_id": f"b{b}",
            }
            for i in range(spec.n_episodes)
            for p, b in [divmod(i, spec.n_branches)]
        ]

    def branches(self) -> dict[str, list[int]]:
        """``prompt_id -> episode indices`` sharing a start frame (one entry each if unbranched)."""
        groups: dict[str, list[int]] = {}
        for i, rec in enumerate(self.meta):
            groups.setdefault(rec["prompt_id"], []).append(i)
        return groups

    def branch_pairs(self) -> list[tuple[int, int]]:
        """All unordered pairs of episodes that share a start frame (empty when ``n_branches == 1``)."""
        pairs = []
        for idxs in self.branches().values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pairs.append((idxs[a], idxs[b]))
        return pairs

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], split: str = "train"
    ) -> SyntheticTrajectoryDataset:
        fields = {k: cfg[k] for k in SyntheticSpec.__dataclass_fields__ if k in cfg}
        if "frame_shape" in fields:
            fields["frame_shape"] = tuple(int(x) for x in fields["frame_shape"])
        return cls(SyntheticSpec(**fields), split=split)

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    def __len__(self) -> int:
        return self.spec.n_episodes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"frames": self.frames[idx], "actions": self.actions[idx], "meta": self.meta[idx]}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack frames and actions; keep metadata as a list of dicts."""
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


__all__ = ["SyntheticSpec", "SyntheticTrajectoryDataset", "collate"]
