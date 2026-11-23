"""Latent rollout: open-loop prediction error versus horizon.

Hypothesis: for the same latent dimension, the hyperbolic head has lower geodesic rollout error
than the Euclidean head at long horizons because branching futures spread out exponentially in
hyperbolic space and so remain distinguishable. Falsified if, after sweeping curvature, no
curvature beats the Euclidean baseline's normalised error at any horizon beyond the first.

The task rolls the head out open-loop from the embedded first frame with the true actions and
scores every step with the geodesic distance of ``bundle.manifold`` against the embedded true
frames. The "nothing moves" static baseline (predict the context latent at every horizon) is
scored the same way; the normalised error is the ratio of the two and equals 1 for a static
predictor in every geometry, which is what makes it comparable across curvatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hyperbolic_world_model.metrics.geodesic_error import geodesic_error, static_baseline_error
from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult, check_rollout_horizon


@dataclass
class RolloutErrors:
    """Per-sample, per-horizon errors of a rollout evaluation (kept unreduced for bootstrapping).

    Attributes:
        error: ``(N, H)`` geodesic error of the open-loop rollout.
        static: ``(N, H)`` geodesic distance the true latent travelled from the context, i.e.
            the error of the static baseline.
    """

    error: Tensor
    static: Tensor

    @property
    def n(self) -> int:
        return int(self.error.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.error.shape[1])

    def per_horizon(self) -> Tensor:
        """``(H,)`` mean rollout error at each step."""
        return self.error.mean(0)

    def static_per_horizon(self) -> Tensor:
        """``(H,)`` mean static-baseline error at each step."""
        return self.static.mean(0)

    def normalised_per_horizon(self) -> Tensor:
        """``(H,)`` ratio of the two means (1 for a static predictor in every geometry)."""
        eps = torch.finfo(self.error.dtype).eps
        return self.per_horizon() / self.static_per_horizon().clamp_min(eps)

    def curves(self) -> pd.DataFrame:
        """Tidy ``horizon, geodesic_error, static_baseline_error, normalised_error`` table."""
        return pd.DataFrame(
            {
                "horizon": range(1, self.horizon + 1),
                "geodesic_error": self.per_horizon().cpu().numpy(),
                "static_baseline_error": self.static_per_horizon().cpu().numpy(),
                "normalised_error": self.normalised_per_horizon().cpu().numpy(),
            }
        )

    def summary(self, prefix: str = "") -> dict[str, float]:
        """Scalar metrics with an optional ``prefix`` (used by the compositional task)."""
        per_h, norm = self.per_horizon(), self.normalised_per_horizon()
        return {
            f"{prefix}geodesic_error_h1": float(per_h[0]),
            f"{prefix}geodesic_error_hmax": float(per_h[-1]),
            f"{prefix}geodesic_error_mean": float(per_h.mean()),
            f"{prefix}normalised_error_h1": float(norm[0]),
            f"{prefix}normalised_error_hmax": float(norm[-1]),
            f"{prefix}normalised_error_mean": float(norm.mean()),
        }


class LatentRolloutTask(Task):
    """Open-loop rollout scored with the geodesic error of ``bundle.manifold``.

    Args:
        horizon: number of steps to roll out (``<=`` dataset horizon).
        batch_size: evaluation batch size.
    """

    name = "latent_rollout"

    def __init__(self, horizon: int = 8, batch_size: int = 32) -> None:
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)

    def check_dataset(self, dataset: Dataset) -> None:
        check_rollout_horizon(self.name, self.horizon, dataset)

    @torch.no_grad()
    def evaluate(
        self,
        bundle: ModelBundle,
        dataset: Dataset,
        indices: Sequence[int] | None = None,
        device: str = "cpu",
    ) -> RolloutErrors:
        """Roll out every item (or the given ``indices``) and return unreduced errors."""
        head = bundle.predictor.eval()
        m = bundle.manifold
        errs, static_errs = [], []
        for batch in self.iter_embedded(bundle, dataset, self.batch_size, indices, device):
            z = batch.latents
            if z.shape[1] - 1 < self.horizon:
                raise ValueError(
                    f"task horizon {self.horizon} exceeds the dataset horizon {z.shape[1] - 1}"
                )
            actions = batch.actions[:, : self.horizon]
            target = z[:, 1 : self.horizon + 1]
            pred = head.rollout(z[:, 0], actions)
            errs.append(geodesic_error(m, pred, target))
            static_errs.append(static_baseline_error(m, z[:, 0], target))
        if not errs:
            raise ValueError("no items to evaluate")
        return RolloutErrors(error=torch.cat(errs), static=torch.cat(static_errs))

    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        errors = self.evaluate(bundle, dataset, device=device)
        res = self.result(bundle, **errors.summary(), n_items=errors.n)
        res.curves = errors.curves()
        return res


__all__ = ["LatentRolloutTask", "RolloutErrors"]
