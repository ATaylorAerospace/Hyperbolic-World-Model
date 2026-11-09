"""Long-horizon consistency on branching futures (Cosmos 3 generated trajectories).

Hypothesis: two rollouts that share a start frame and diverge in their action sequences should
diverge in latent space at a rate that matches the divergence of the generated videos. In
hyperbolic space a geodesic ball's volume grows exponentially, so many branches fit without
collapsing onto one another. Falsified if the hyperbolic head's correlation is not higher than
the Euclidean head's, or if its latent divergence saturates earlier.

For every pair of branches sharing a start frame (``dataset.branch_pairs()``, offered by
:class:`~hyperbolic_world_model.data.cosmos3.dataset.Cosmos3TrajectoryDataset`) the task rolls
the head out open-loop from the shared start latent with each branch's actions and measures, at
every horizon ``h``:

* ``latent_divergence``: geodesic distance (``bundle.manifold``) between the two rollouts;
* ``embedded_divergence``: geodesic distance between the embedded *true* frames of the two
  branches (a property of the frozen embedding, reported as a control);
* ``encoder_divergence``: Euclidean distance between the frozen encoder's pooled outputs for the
  two generated frames. The encoder is a Euclidean token-space model, so this is *its* native
  geometry; it is the ground-truth video divergence the latent divergence is correlated with;
* ``pixel_divergence``: RMSE between the two generated frames.

Scalars: the pooled Spearman correlation between latent and encoder divergence over all
``(pair, h)`` samples, the horizon at which the mean latent divergence first reaches
``saturation_fraction`` of its maximum (later is better), and the divergence at the first and
last horizon.

Branch collapse. A world model can fail by rolling distinct action sequences toward one mean
trajectory. Both divergences above are geodesic distances in the same manifold, so their ratio is
unit-free and comparable across geometries and curvatures:

* ``branch_separation_ratio`` (per horizon): mean latent divergence over mean embedded
  divergence. ``1`` means the rollouts separate exactly as far as the true embedded branches;
  below ``1`` is collapse, above ``1`` is over-separation. It has no preferred direction, so it
  is reported but never ranked. ``branch_separation_ratio_median`` is the per-pair median, which
  a few pairs with nearly identical true frames cannot dominate;
* ``branch_separation_log_error``: ``|log ratio|`` averaged over horizons, the ranked form
  (lower is better, ``0`` is perfect, symmetric in collapse and over-separation). A ratio of
  zero (a static head) is clamped at :data:`RATIO_FLOOR` so it ranks worst instead of infinite;
* ``branch_collapse_fraction`` (per horizon): fraction of pairs whose latent divergence is below
  ``collapse_threshold`` times their embedded divergence (lower is better).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hyperbolic_world_model.metrics.geodesic_error import geodesic_error
from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult, spearman

DIVERGENCES = ("latent_divergence", "embedded_divergence", "encoder_divergence", "pixel_divergence")

#: Embedded divergences at or below this are treated as "the true branches did not separate":
#: the pair has no defined ratio at that horizon. Also the floor of a ratio inside the log error.
RATIO_FLOOR = 1e-12


def branch_separation(
    latent: Sequence[float] | Tensor, embedded: Sequence[float] | Tensor, collapse_threshold: float
) -> dict[str, float]:
    """Separation statistics of one horizon from per-pair latent and embedded divergences.

    Returns ``ratio`` (mean latent over mean embedded), ``median`` (median per-pair ratio over
    pairs whose embedded divergence exceeds :data:`RATIO_FLOOR`) and ``collapse_fraction``
    (pairs with ``latent < collapse_threshold * embedded``). ``ratio`` and ``median`` are NaN
    when the true branches did not separate at all.
    """
    lat = torch.as_tensor(latent, dtype=torch.float64)
    emb = torch.as_tensor(embedded, dtype=torch.float64)
    if lat.shape != emb.shape or lat.ndim != 1 or lat.numel() == 0:
        raise ValueError("latent and embedded must be non-empty 1-D sequences of equal length")
    mean_emb = float(emb.mean())
    defined = emb > RATIO_FLOOR
    return {
        "ratio": float(lat.mean()) / mean_emb if mean_emb > RATIO_FLOOR else math.nan,
        "median": float((lat[defined] / emb[defined]).median())
        if bool(defined.any())
        else math.nan,
        "collapse_fraction": float((lat < collapse_threshold * emb).to(torch.float64).mean()),
    }


def separation_log_error(ratios: Sequence[float]) -> float:
    """Mean ``|log ratio|`` over the horizons with a defined ratio (NaN if there are none)."""
    logs = [abs(math.log(max(r, RATIO_FLOOR))) for r in ratios if not math.isnan(r)]
    return sum(logs) / len(logs) if logs else math.nan


def branch_pairs_of(dataset: Dataset) -> list[tuple[int, int]]:
    """``dataset.branch_pairs()``: index pairs of windows that share a start frame."""
    if not hasattr(dataset, "branch_pairs"):
        raise TypeError(
            f"{type(dataset).__name__} has no branch_pairs(); long_horizon_consistency needs "
            "branching rollouts (data=cosmos3_generated)"
        )
    pairs = [(int(a), int(b)) for a, b in dataset.branch_pairs()]
    if not pairs:
        raise ValueError("dataset has no branch pairs; generate more than one branch per prompt")
    return pairs


def saturation_horizon(curve: Sequence[float] | Tensor, fraction: float) -> int:
    """First horizon (1-based) at which ``curve`` reaches ``fraction`` of its maximum."""
    c = torch.as_tensor(curve, dtype=torch.float64)
    if c.numel() == 0:
        raise ValueError("empty curve")
    return int((c >= fraction * c.max()).nonzero(as_tuple=True)[0][0]) + 1


class LongHorizonConsistencyTask(Task):
    """Latent divergence between branching rollouts versus the divergence of their frames.

    Args:
        horizon: maximum number of steps to compare (capped by the shortest branch in a batch).
        batch_size: number of pairs per batch.
        saturation_fraction: fraction of the maximum mean latent divergence that defines
            saturation.
        start_atol: largest per-pixel difference (frames in ``[0, 1]``) tolerated between the
            start frames of a pair. Branches of a real prompt are decoded from separate video
            files and resized, so their start frames agree only to a few uint8 quanta; the
            default allows five (``5 / 255``), far below the difference between distinct frames.
        collapse_threshold: a pair counts as collapsed at a horizon when its latent divergence
            is below this fraction of its embedded divergence.
    """

    name = "long_horizon_consistency"

    def __init__(
        self,
        horizon: int = 32,
        batch_size: int = 16,
        saturation_fraction: float = 0.95,
        start_atol: float = 5 / 255,
        collapse_threshold: float = 0.25,
    ) -> None:
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        if not 0 < float(saturation_fraction) <= 1:
            raise ValueError("saturation_fraction must be in (0, 1]")
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.saturation_fraction = float(saturation_fraction)
        if not 0 < float(collapse_threshold) <= 1:
            raise ValueError("collapse_threshold must be in (0, 1]")
        self.start_atol = float(start_atol)
        self.collapse_threshold = float(collapse_threshold)

    def _load_chunk(
        self, dataset: Dataset, pairs: list[tuple[int, int]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Load the distinct windows of a chunk of pairs once; return stacked frames, actions, and
        the index of each pair's two windows into them, plus the common horizon."""
        unique = sorted({i for pair in pairs for i in pair})
        items = {i: dataset[i] for i in unique}
        h = min(self.horizon, *(int(it["actions"].shape[0]) for it in items.values()))
        if h < 1:
            raise ValueError("branches must have at least one action")
        frames = torch.stack([items[i]["frames"][: h + 1] for i in unique])
        actions = torch.stack([items[i]["actions"][:h] for i in unique])
        pos = {i: k for k, i in enumerate(unique)}
        ia = torch.tensor([pos[a] for a, _ in pairs])
        ib = torch.tensor([pos[b] for _, b in pairs])
        return frames, actions, ia, ib, h

    def check_dataset(self, dataset: Dataset) -> None:
        branch_pairs_of(dataset)

    @torch.no_grad()
    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        bundle.encoder.assert_frozen()
        head = bundle.predictor.eval()
        m = bundle.manifold
        pairs = branch_pairs_of(dataset)

        sums = {k: torch.zeros(self.horizon, dtype=torch.float64) for k in DIVERGENCES}
        counts = torch.zeros(self.horizon, dtype=torch.float64)
        per_h: dict[str, list[list[float]]] = {
            k: [[] for _ in range(self.horizon)] for k in DIVERGENCES
        }
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            frames, actions, ia, ib, h = self._load_chunk(dataset, chunk)
            frames, actions = frames.to(device), actions.to(device)
            ia, ib = ia.to(device), ib.to(device)
            # Every distinct window is encoded, embedded and rolled out exactly once per chunk;
            # a window shared by several pairs (n - 1 pairs for n branches) is only indexed.
            enc = bundle.encoder.encode(frames)  # (U, h+1, D)
            z = head.embed(enc)  # (U, h+1, d) on the manifold
            roll = head.rollout(z[:, 0], actions)  # (U, h, d)
            fa, fb = frames[ia], frames[ib]
            gap = float((fa[:, 0] - fb[:, 0]).abs().max())
            if gap > self.start_atol:
                raise ValueError(
                    f"paired branches must share their start frame (max pixel gap {gap:.4f} > "
                    f"start_atol {self.start_atol:.4f})"
                )
            enc_a, enc_b = enc[ia], enc[ib]
            z_a, z_b = z[ia], z[ib]
            roll_a, roll_b = roll[ia], roll[ib]
            div = {
                "latent_divergence": geodesic_error(m, roll_a, roll_b),
                "embedded_divergence": geodesic_error(m, z_a[:, 1:], z_b[:, 1:]),
                # Encoder outputs live in the encoder's own (Euclidean) space; this is not a
                # distance between head latents and never goes through bundle.manifold.
                "encoder_divergence": torch.linalg.vector_norm(enc_a[:, 1:] - enc_b[:, 1:], dim=-1),
                "pixel_divergence": (fa[:, 1:] - fb[:, 1:]).flatten(2).pow(2).mean(-1).sqrt(),
            }
            for k, v in div.items():
                v = v.to(torch.float64).cpu()
                sums[k][:h] += v.sum(0)
                for t in range(h):
                    per_h[k][t].extend(v[:, t].tolist())
            counts[:h] += fa.shape[0]

        valid = counts > 0
        h_eff = int(valid.sum())
        mean = {k: (sums[k][valid] / counts[valid]) for k in DIVERGENCES}
        pooled = {k: [x for t in range(h_eff) for x in per_h[k][t]] for k in DIVERGENCES}
        rho_per_h = [
            spearman(per_h["latent_divergence"][t], per_h["encoder_divergence"][t])
            for t in range(h_eff)
        ]
        sat = saturation_horizon(mean["latent_divergence"], self.saturation_fraction)
        lat = mean["latent_divergence"]
        sep = [
            branch_separation(
                per_h["latent_divergence"][t],
                per_h["embedded_divergence"][t],
                self.collapse_threshold,
            )
            for t in range(h_eff)
        ]
        ratios = [s["ratio"] for s in sep]
        defined = [r for r in ratios if not math.isnan(r)]
        res = self.result(
            bundle,
            divergence_spearman=spearman(pooled["latent_divergence"], pooled["encoder_divergence"]),
            divergence_spearman_pixels=spearman(
                pooled["latent_divergence"], pooled["pixel_divergence"]
            ),
            embedded_divergence_spearman=spearman(
                pooled["embedded_divergence"], pooled["encoder_divergence"]
            ),
            saturation_horizon=sat,
            saturation_fraction_of_horizon=sat / h_eff,
            latent_divergence_h1=float(lat[0]),
            latent_divergence_hmax=float(lat[-1]),
            latent_divergence_growth=float(lat[-1] / lat[0]) if float(lat[0]) > 0 else float("nan"),
            branch_separation_ratio_hmax=ratios[-1],
            branch_separation_ratio_mean=sum(defined) / len(defined) if defined else math.nan,
            branch_separation_log_error=separation_log_error(ratios),
            branch_collapse_fraction_hmax=sep[-1]["collapse_fraction"],
            branch_collapse_fraction_mean=sum(s["collapse_fraction"] for s in sep) / h_eff,
            n_pairs=len(pairs),
            horizon=h_eff,
        )
        res.curves = pd.DataFrame(
            {
                "horizon": range(1, h_eff + 1),
                **{k: mean[k].numpy() for k in DIVERGENCES},
                "spearman": rho_per_h,
                "branch_separation_ratio": ratios,
                "branch_separation_ratio_median": [s["median"] for s in sep],
                "branch_collapse_fraction": [s["collapse_fraction"] for s in sep],
                "n_pairs": counts[valid].to(torch.int64).numpy(),
            }
        )
        return res


__all__ = [
    "DIVERGENCES",
    "RATIO_FLOOR",
    "LongHorizonConsistencyTask",
    "branch_pairs_of",
    "branch_separation",
    "saturation_horizon",
    "separation_log_error",
]
