"""Compositional generalisation: held-out (embodiment, primitive) combinations.

Hypothesis: if the latent space encodes the hierarchy, a head trained on all embodiments and all
primitives but *not* on every combination will roll out unseen combinations with an error gap
(unseen minus seen) that is smaller in hyperbolic space than in Euclidean space. Falsified if the
gap is not smaller for the best swept curvature, or if it is smaller only because seen-combination
error got worse.

The split is the dataset's own ``split_by_combination`` when it has one
(:class:`~hyperbolic_world_model.data.cosmos3.dataset.Cosmos3TrajectoryDataset`) and otherwise
:func:`~hyperbolic_world_model.data.splits.split_by_combination` over the per-item metadata; both
paths run the same shared safety checks (every held-out combination must occur, and holding it
out must not remove an embodiment or a primitive entirely). The trainer applies the same split to
the training set (``data.holdout_combinations``), so the evaluation here measures generalisation
to combinations the head never saw.

Both subsets are scored by :class:`~hyperbolic_world_model.tasks.latent_rollout.LatentRolloutTask`
in the head's native geometry; the reported gaps use the normalised error so they are comparable
across curvatures.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from hyperbolic_world_model.data.splits import (
    COMBINATION_KEYS,
    normalise_holdout,
    split_by_combination,
)
from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult, check_rollout_horizon
from hyperbolic_world_model.tasks.latent_rollout import LatentRolloutTask


def split_dataset_by_combination(
    dataset: Dataset, holdout: Iterable[Sequence[str]]
) -> tuple[list[int], list[int]]:
    """Use the dataset's own ``split_by_combination`` if it has one, else its item metadata."""
    holdout = normalise_holdout(holdout)
    if hasattr(dataset, "split_by_combination"):
        seen, held = dataset.split_by_combination([list(c) for c in holdout])
        return [int(i) for i in seen], [int(i) for i in held]
    metadata = [dataset[i]["meta"] for i in range(len(dataset))]
    return split_by_combination(metadata, holdout)


class CompositionalGeneralizationTask(Task):
    """Rollout error on seen versus held-out (embodiment, primitive) combinations.

    Args:
        horizon: rollout horizon.
        batch_size: evaluation batch size.
        holdout: held-out ``[embodiment, primitive]`` pairs; must be non-empty.
    """

    name = "compositional_generalization"

    def __init__(
        self,
        horizon: int = 8,
        batch_size: int = 32,
        holdout: Iterable[Sequence[str]] | None = None,
    ) -> None:
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.holdout = normalise_holdout(holdout)
        if not self.holdout:
            raise ValueError(
                "compositional_generalization needs at least one held-out [embodiment, primitive] "
                "combination; set data.holdout_combinations"
            )

    def split(self, dataset: Dataset) -> tuple[list[int], list[int]]:
        seen, held = split_dataset_by_combination(dataset, self.holdout)
        if not seen or not held:
            raise ValueError("both the seen and the held-out subset must be non-empty")
        return seen, held

    def check_dataset(self, dataset: Dataset) -> None:
        """The horizon fits and the held-out split is valid, checked before any training."""
        check_rollout_horizon(self.name, self.horizon, dataset)
        self.split(dataset)

    @torch.no_grad()
    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        seen, held = self.split(dataset)
        rollout = LatentRolloutTask(horizon=self.horizon, batch_size=self.batch_size)
        e_seen = rollout.evaluate(bundle, dataset, seen, device)
        e_unseen = rollout.evaluate(bundle, dataset, held, device)
        s, u = e_seen.summary("seen_"), e_unseen.summary("unseen_")
        gaps = {
            "gap_geodesic_error_hmax": u["unseen_geodesic_error_hmax"]
            - s["seen_geodesic_error_hmax"],
            "gap_geodesic_error_mean": u["unseen_geodesic_error_mean"]
            - s["seen_geodesic_error_mean"],
            "gap_normalised_error_hmax": u["unseen_normalised_error_hmax"]
            - s["seen_normalised_error_hmax"],
            "gap_normalised_error_mean": u["unseen_normalised_error_mean"]
            - s["seen_normalised_error_mean"],
            "unseen_to_seen_ratio_hmax": u["unseen_normalised_error_hmax"]
            / max(s["seen_normalised_error_hmax"], torch.finfo(torch.float32).eps),
        }
        res = self.result(
            bundle,
            **s,
            **u,
            **gaps,
            n_seen=e_seen.n,
            n_unseen=e_unseen.n,
            n_holdout_combinations=len(self.holdout),
        )
        cs, cu = e_seen.curves(), e_unseen.curves()
        res.curves = pd.DataFrame(
            {
                "horizon": cs["horizon"],
                "seen_error": cs["geodesic_error"],
                "unseen_error": cu["geodesic_error"],
                "seen_static_baseline_error": cs["static_baseline_error"],
                "unseen_static_baseline_error": cu["static_baseline_error"],
                "seen_normalised_error": cs["normalised_error"],
                "unseen_normalised_error": cu["normalised_error"],
                "gap_normalised_error": cu["normalised_error"] - cs["normalised_error"],
            }
        )
        return res


__all__ = [
    "COMBINATION_KEYS",
    "CompositionalGeneralizationTask",
    "normalise_holdout",
    "split_by_combination",
    "split_dataset_by_combination",
]
