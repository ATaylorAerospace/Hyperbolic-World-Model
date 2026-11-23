"""Evaluation tasks. Each task consumes a :class:`~hyperbolic_world_model.models.ModelBundle` and a
dataset and returns a :class:`~hyperbolic_world_model.tasks.base.TaskResult` whose distance-based
numbers are computed with ``bundle.manifold``.

The four tasks and the hypothesis each tests are documented in ``docs/methodology.md``:

* ``latent_rollout``: open-loop geodesic error versus horizon with the static baseline;
* ``hierarchy_reconstruction``: distortion, mAP and depth correlation against the metadata tree;
* ``long_horizon_consistency``: latent divergence of branching rollouts versus video divergence;
* ``compositional_generalization``: rollout error gap on held-out (embodiment, primitive) pairs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperbolic_world_model.tasks.base import (
    EmbeddedBatch,
    Task,
    TaskResult,
    check_rollout_horizon,
    dataset_horizon,
    frechet_mean,
    spearman,
)
from hyperbolic_world_model.tasks.compositional_generalization import (
    CompositionalGeneralizationTask,
)
from hyperbolic_world_model.tasks.hierarchy_reconstruction import HierarchyReconstructionTask
from hyperbolic_world_model.tasks.latent_rollout import LatentRolloutTask
from hyperbolic_world_model.tasks.long_horizon_consistency import LongHorizonConsistencyTask

TASKS: dict[str, type[Task]] = {
    "latent_rollout": LatentRolloutTask,
    "hierarchy_reconstruction": HierarchyReconstructionTask,
    "long_horizon_consistency": LongHorizonConsistencyTask,
    "compositional_generalization": CompositionalGeneralizationTask,
}


def build_task(task_cfg: Mapping[str, Any]) -> Task:
    """Instantiate the task named by ``task_cfg['name']`` with the remaining keys as kwargs."""
    cfg = dict(task_cfg)
    name = cfg.pop("name")
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return TASKS[name](**cfg)


__all__ = [
    "TASKS",
    "CompositionalGeneralizationTask",
    "EmbeddedBatch",
    "HierarchyReconstructionTask",
    "LatentRolloutTask",
    "LongHorizonConsistencyTask",
    "Task",
    "TaskResult",
    "build_task",
    "check_rollout_horizon",
    "dataset_horizon",
    "frechet_mean",
    "spearman",
]
