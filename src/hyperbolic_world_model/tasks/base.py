"""Task interface and the helpers every task shares.

A task evaluates a :class:`~hyperbolic_world_model.models.registry.ModelBundle` on a dataset and
returns a :class:`TaskResult`. Everything distance-based inside a task goes through
``bundle.manifold`` (the geometry the head was trained in), and every result carries that
geometry and curvature so a table can never present a number without its space.

Shared helpers:

* :meth:`Task.iter_embedded` streams the dataset through the frozen encoder and the head's frozen
  embedding in batches, so every task sees ``(B, T, d)`` manifold points and the batch metadata
  without touching the data source.
* :func:`frechet_mean` pools points on the manifold (a Karcher mean by iterated
  ``logmap``/``expmap``) instead of averaging coordinates, which would leave the manifold or bias
  towards the origin.
* :func:`spearman` is the rank correlation used by the hierarchy and long-horizon tasks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from hyperbolic_world_model.data.synthetic import collate
from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.models.registry import ModelBundle

log = logging.getLogger(__name__)
#: Armijo constant of the Fréchet-mean line search (fraction of the first-order decrease).
ARMIJO = 0.5


@dataclass
class TaskResult:
    """Scalar metrics plus optional per-horizon curves.

    Attributes:
        task: task name.
        geometry: ``manifold.name`` the metrics were computed in (always recorded so a table can
            never present a number without its geometry).
        curvature: curvature the metrics were computed at.
        metrics: flat ``{name: value}`` mapping of scalars.
        curves: optional tidy DataFrame (e.g. ``horizon, error``) for plotting.
    """

    task: str
    geometry: str
    curvature: float
    metrics: dict[str, float] = field(default_factory=dict)
    curves: pd.DataFrame | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "geometry": self.geometry,
            "curvature": self.curvature,
            "metrics": self.metrics,
        }


@dataclass
class EmbeddedBatch:
    """One batch after the frozen encoder and the head's frozen embedding.

    Attributes:
        latents: ``(B, T, d)`` points on ``bundle.manifold`` (``d`` is the ambient dimension).
        encoder_latents: ``(B, T, D)`` pooled frozen-encoder outputs (Euclidean, the encoder's
            own space).
        actions: ``(B, T - 1, a)`` actions between consecutive frames.
        frames: ``(B, T, C, H, W)`` input frames, on ``device``.
        meta: per-item metadata dicts.
        indices: dataset indices of the items in this batch.
    """

    latents: Tensor
    encoder_latents: Tensor
    actions: Tensor
    frames: Tensor
    meta: list[dict[str, Any]]
    indices: list[int]


class Task(ABC):
    """A task is a callable that evaluates a bundle on a dataset."""

    name: str = "abstract"

    @abstractmethod
    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        """Evaluate and return metrics in the bundle's native geometry."""

    def check_dataset(self, dataset: Dataset) -> None:
        """Raise early if ``dataset`` cannot support this task (called before training starts).

        The default only refuses an empty dataset; tasks with further requirements (a rollout
        horizon, branch pairs, a held-out split) extend it so a sweep fails in seconds rather
        than after every epoch has run.
        """
        dataset_horizon(dataset)

    def result(self, bundle: ModelBundle, **metrics: float) -> TaskResult:
        """Helper stamping the geometry and curvature onto a result."""
        return TaskResult(
            task=self.name,
            geometry=bundle.manifold.name,
            curvature=bundle.manifold.curvature,
            metrics={k: float(v) for k, v in metrics.items()},
        )

    @staticmethod
    def loader(
        dataset: Dataset, batch_size: int, indices: Sequence[int] | None = None
    ) -> DataLoader:
        """Deterministic (unshuffled) loader over ``dataset`` or a subset of its indices."""
        ds = dataset if indices is None else Subset(dataset, list(indices))
        return DataLoader(ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate)

    @staticmethod
    @torch.no_grad()
    def iter_embedded(
        bundle: ModelBundle,
        dataset: Dataset,
        batch_size: int,
        indices: Sequence[int] | None = None,
        device: str = "cpu",
    ) -> Iterator[EmbeddedBatch]:
        """Stream ``dataset`` through the frozen encoder and the head's embedding.

        Asserts the encoder is frozen first and puts the head in eval mode; both are invariants
        every task relies on.
        """
        bundle.encoder.assert_frozen()
        head = bundle.predictor.eval()
        idx = list(range(len(dataset))) if indices is None else list(indices)
        loader = Task.loader(dataset, batch_size, idx)
        offset = 0
        for batch in loader:
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            enc = bundle.encoder.encode(frames)  # (B, T, D)
            z = head.embed(enc)  # (B, T, d) on the manifold
            n = frames.shape[0]
            yield EmbeddedBatch(
                latents=z,
                encoder_latents=enc,
                actions=actions,
                frames=frames,
                meta=batch["meta"],
                indices=idx[offset : offset + n],
            )
            offset += n


def dataset_horizon(dataset: Dataset) -> int:
    """Number of actions per item of ``dataset`` (read from its first item).

    Raises:
        ValueError: if the dataset is empty, which no task can evaluate.
    """
    if len(dataset) == 0:  # type: ignore[arg-type]
        raise ValueError(
            f"{type(dataset).__name__} is empty; there is nothing to evaluate (for generated "
            "data, check the split fractions and the horizon against the rollout lengths)"
        )
    return int(dataset[0]["actions"].shape[0])


def check_rollout_horizon(task: str, horizon: int, dataset: Dataset) -> None:
    """Raise if ``dataset`` items are shorter than the ``horizon`` a rollout task needs."""
    available = dataset_horizon(dataset)
    if horizon > available:
        raise ValueError(f"{task}: task horizon {horizon} exceeds the dataset horizon {available}")


def frechet_mean(
    manifold: Manifold,
    points: Tensor,
    dim: int = -2,
    n_iter: int = 100,
    tol: float = 1e-5,
    weights: Tensor | None = None,
    max_backtracks: int = 20,
) -> Tensor:
    """Fréchet (Karcher) mean of ``points`` along ``dim`` on ``manifold``.

    Minimises ``f(x) = sum_i w_i d(x, p_i)^2`` by Riemannian gradient descent: starting from the
    exponential map of the tangent-space mean at the origin, each iteration proposes the classic
    update ``x <- exp_x(sum_i w_i log_x(p_i))`` and halves the step until the Armijo sufficient
    decrease condition holds (``f`` must fall by at least a quarter of the first-order
    prediction). Both safeguards matter in hyperbolic space: for points more than about two units
    from the current iterate the Hessian of ``d^2`` exceeds two, so the unit step overshoots, and
    a step that merely does not increase ``f`` can land mirror-symmetric across the minimiser and
    bounce forever. ``f`` is geodesically convex on every geometry here, so the descent converges
    to the unique mean; the loop stops when the Riemannian gradient is shorter than ``tol`` and
    warns if ``n_iter`` runs out first. On the Euclidean manifold the first unit step lands on
    the arithmetic mean exactly.

    Tasks use this to pool latents over time and over trajectories: a coordinate average would
    leave the hyperboloid and pull ball points towards the origin.

    The descent runs in float64 whatever the dtype of ``points`` and the result is cast back.
    In float32 the distance noise of the curved geometries is above ``tol``, so the loop cannot
    see its own convergence: it runs all ``n_iter`` iterations, warns on nearly every call and
    returns a mean about ``1e-3`` geodesic units off for points near ``max_radius``. Latents
    are small, so the promotion costs less than the iterations it saves.

    Args:
        manifold: geometry of ``points``.
        points: ``(..., n, d)`` points on ``manifold`` (with ``dim=-2``; any ``dim`` other than
            the last is allowed).
        dim: axis to reduce.
        n_iter: maximum number of descent iterations.
        tol: stop when every residual Riemannian gradient is shorter than ``tol`` (geodesic
            units); a warning is logged if ``n_iter`` is exhausted first.
        weights: optional non-negative weights broadcastable to ``points.shape[:-1]``; ``None``
            is uniform.
        max_backtracks: maximum step halvings per iteration.

    Returns:
        ``points`` with ``dim`` removed.
    """
    if points.ndim < 2:
        raise ValueError("points must have at least a reduction axis and a coordinate axis")
    dim = dim % points.ndim
    if dim == points.ndim - 1:
        raise ValueError("dim must not be the coordinate (last) axis")
    if points.shape[dim] == 0:
        raise ValueError("cannot take the mean of zero points")
    out_dtype = points.dtype
    if out_dtype != torch.float64 and points.device.type != "mps":  # MPS has no float64
        # Re-project after the cast so the constraint holds to float64 accuracy, not float32's.
        points = manifold.proj(points.to(torch.float64))
    if weights is None:
        w = torch.full(points.shape[:-1], 1.0, dtype=points.dtype, device=points.device)
    else:
        w = weights.to(points.dtype).expand(points.shape[:-1])
    w = w / w.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(points.dtype).eps)

    def objective(x: Tensor) -> Tensor:
        return (w * manifold.dist(x.unsqueeze(dim), points) ** 2).sum(dim=dim)

    x = manifold.proj(manifold.expmap0((w.unsqueeze(-1) * manifold.logmap0(points)).sum(dim=dim)))
    f = objective(x)
    gnorm = torch.zeros_like(f)
    for _ in range(n_iter):
        u = (w.unsqueeze(-1) * manifold.logmap(x.unsqueeze(dim), points)).sum(dim=dim)
        u = manifold.proj_tan(x, u)
        # |u|_x in the Riemannian metric is the geodesic length of exp_x(u); grad f = -2u, so the
        # directional derivative of f along u is -2 |u|_x^2.
        gnorm = manifold.dist(x, manifold.proj(manifold.expmap(x, u)))
        if float(gnorm.max()) < tol:
            break
        t = torch.ones_like(f)
        accepted = torch.zeros_like(f, dtype=torch.bool)
        x_try, f_try = x, f
        for _ in range(max_backtracks):
            x_try = manifold.proj(manifold.expmap(x, t.unsqueeze(-1) * u))
            f_try = objective(x_try)
            # Armijo sufficient decrease (c = 1/4 of the first-order prediction 2 t |u|^2): a
            # step that merely does not increase f, such as a unit step landing mirror-symmetric
            # across the minimiser, is rejected instead of starting a period-2 bounce.
            accepted = f_try <= f - ARMIJO * t * gnorm**2
            if bool(accepted.all()):
                break
            t = torch.where(accepted, t, t / 2)
        if not bool(accepted.any()):
            break  # every element is at the numerical floor: no step can still decrease f
        keep = accepted.unsqueeze(-1)
        x = torch.where(keep, x_try, x)
        f = torch.where(accepted, f_try, f)
    else:
        log.warning(
            "frechet_mean did not converge in %d iterations: largest residual gradient %.2e "
            "(tol %.0e); increase n_iter or check the points",
            n_iter,
            float(gnorm.max()),
            tol,
        )
    return x if x.dtype == out_dtype else manifold.proj(x.to(out_dtype))


def spearman(a: Sequence[float] | Tensor, b: Sequence[float] | Tensor) -> float:
    """Spearman rank correlation (average ranks for ties); ``NaN`` if either input is constant."""
    sa = pd.Series(torch.as_tensor(a).detach().cpu().to(torch.float64).numpy())
    sb = pd.Series(torch.as_tensor(b).detach().cpu().to(torch.float64).numpy())
    if len(sa) != len(sb):
        raise ValueError("inputs must have the same length")
    if len(sa) < 2 or sa.nunique() < 2 or sb.nunique() < 2:
        return float("nan")
    return float(sa.corr(sb, method="spearman"))


__all__ = [
    "EmbeddedBatch",
    "Task",
    "TaskResult",
    "check_rollout_horizon",
    "dataset_horizon",
    "frechet_mean",
    "spearman",
]
