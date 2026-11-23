"""Task base class: registry, result stamping, batched embedding, Fréchet mean, Spearman."""

from __future__ import annotations

import math

import pytest
import torch

from hyperbolic_world_model.geometry import Euclidean, Lorentz, Manifold, PoincareBall
from hyperbolic_world_model.tasks import (
    TASKS,
    CompositionalGeneralizationTask,
    HierarchyReconstructionTask,
    LatentRolloutTask,
    LongHorizonConsistencyTask,
    Task,
    TaskResult,
    build_task,
    frechet_mean,
    spearman,
)
from tests.tasks.helpers import GEOMETRIES, synthetic_dataset, tiny_bundle

MANIFOLDS = [
    pytest.param(Euclidean(), id="euclidean"),
    pytest.param(PoincareBall(c=-1.0), id="poincare"),
    pytest.param(PoincareBall(c=-2.0), id="poincare(c=-2)"),
    pytest.param(Lorentz(c=-1.0), id="lorentz"),
    pytest.param(Lorentz(c=-0.5), id="lorentz(c=-0.5)"),
]


def _points(m: Manifold, *shape: int, scale: float = 0.7) -> torch.Tensor:
    v = torch.randn(*shape, dtype=torch.float64) * scale
    return m.expmap0(m.tangent0_from_euclidean(v))


def test_registry_covers_the_four_tasks_and_build_task_validates() -> None:
    assert set(TASKS) == {
        "latent_rollout",
        "hierarchy_reconstruction",
        "long_horizon_consistency",
        "compositional_generalization",
    }
    assert all(issubclass(t, Task) and t.name == name for name, t in TASKS.items())
    assert isinstance(build_task({"name": "latent_rollout", "horizon": 3}), LatentRolloutTask)
    assert isinstance(
        build_task({"name": "hierarchy_reconstruction", "n_seeds": 0}), HierarchyReconstructionTask
    )
    assert isinstance(
        build_task({"name": "long_horizon_consistency", "horizon": 4}), LongHorizonConsistencyTask
    )
    t = build_task(
        {"name": "compositional_generalization", "holdout": [["arm_b", "grasp"]], "horizon": 2}
    )
    assert isinstance(t, CompositionalGeneralizationTask) and t.holdout == (("arm_b", "grasp"),)
    with pytest.raises(KeyError, match="unknown task"):
        build_task({"name": "nope"})
    with pytest.raises(TypeError):
        build_task({"name": "latent_rollout", "bogus": 1})


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_result_stamps_geometry_and_curvature(geom: tuple[str, float]) -> None:
    bundle = tiny_bundle(*geom)
    res = LatentRolloutTask(horizon=2).result(bundle, a=1, b=torch.tensor(2.5))
    assert isinstance(res, TaskResult)
    assert res.task == "latent_rollout"
    assert res.geometry == geom[0] and res.curvature == geom[1]
    assert res.metrics == {"a": 1.0, "b": 2.5} and all(
        isinstance(v, float) for v in res.metrics.values()
    )
    assert res.to_json_dict() == {
        "task": "latent_rollout",
        "geometry": geom[0],
        "curvature": geom[1],
        "metrics": {"a": 1.0, "b": 2.5},
    }
    assert res.curves is None


def test_task_is_abstract() -> None:
    with pytest.raises(TypeError):
        Task()  # type: ignore[abstract]


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_iter_embedded_streams_manifold_points_with_metadata(geom: tuple[str, float]) -> None:
    bundle, ds = tiny_bundle(*geom), synthetic_dataset(n_episodes=10, horizon=3)
    batches = list(Task.iter_embedded(bundle, ds, batch_size=4))
    assert [len(b.indices) for b in batches] == [4, 4, 2]
    assert [b.indices for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    m = bundle.manifold
    for b in batches:
        n = len(b.indices)
        assert b.latents.shape == (n, 4, bundle.predictor.ambient_dim)
        assert b.encoder_latents.shape == (n, 4, bundle.encoder.embed_dim)
        assert b.actions.shape == (n, 3, ds.action_dim)
        assert b.frames.shape[:2] == (n, 4)
        assert m.check_point(b.latents).all()
        assert [x["embodiment"] for x in b.meta] == [ds.meta[i]["embodiment"] for i in b.indices]
    # A subset of indices is honoured, in the given order.
    sub = list(Task.iter_embedded(bundle, ds, batch_size=8, indices=[7, 2, 5]))
    assert len(sub) == 1 and sub[0].indices == [7, 2, 5]
    assert torch.allclose(sub[0].latents[1], batches[0].latents[2])
    assert not bundle.predictor.training


def test_iter_embedded_refuses_a_thawed_encoder() -> None:
    bundle, ds = tiny_bundle(), synthetic_dataset(n_episodes=4, horizon=2)
    bundle.encoder.weight.requires_grad_(True)
    bundle.encoder.register_parameter("leak", torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(RuntimeError, match="frozen"):
        next(Task.iter_embedded(bundle, ds, batch_size=2))


def test_frechet_mean_is_the_arithmetic_mean_in_flat_space() -> None:
    m = Euclidean()
    pts = torch.randn(3, 7, 4, dtype=torch.float64)
    assert torch.allclose(frechet_mean(m, pts, dim=1), pts.mean(1))
    assert torch.allclose(frechet_mean(m, pts, dim=0), pts.mean(0))
    w = torch.rand(3, 7, dtype=torch.float64)
    expected = (w.unsqueeze(-1) * pts).sum(1) / w.sum(1, keepdim=True)
    assert torch.allclose(frechet_mean(m, pts, dim=1, weights=w), expected)


@pytest.mark.parametrize("m", MANIFOLDS)
def test_frechet_mean_properties_on_curved_manifolds(m: Manifold) -> None:
    pts = _points(m, 4, 9, 3)
    mu = frechet_mean(m, pts, dim=1)
    assert mu.shape == (4, pts.shape[-1]) and m.check_point(mu).all()
    # Stationarity: the mean of the log maps at the mean vanishes.
    u = m.logmap(mu.unsqueeze(1), pts).mean(1)
    assert torch.linalg.vector_norm(u, dim=-1).max() < 1e-5
    # It minimises the sum of squared geodesic distances better than the naive alternatives.
    obj = lambda x: (m.dist(x.unsqueeze(1), pts) ** 2).sum(1)  # noqa: E731
    naive = m.proj(m.expmap0(m.logmap0(pts).mean(1)))
    assert torch.all(obj(mu) <= obj(naive) + 1e-9)
    assert torch.all(obj(mu) <= obj(pts[:, 0]) + 1e-9)
    # Degenerate cases: one point, or copies of the same point, return that point.
    one = _points(m, 2, 1, 3)
    assert torch.allclose(frechet_mean(m, one, dim=1), one[:, 0], atol=1e-6)
    same = one.expand(2, 5, one.shape[-1])
    assert torch.allclose(frechet_mean(m, same, dim=1), one[:, 0], atol=1e-6)
    # Symmetric points around the origin average to the origin.
    v = torch.randn(6, 3, dtype=torch.float64) * 0.5
    sym = m.expmap0(m.tangent0_from_euclidean(torch.cat([v, -v])))
    origin = m.origin(sym.shape[-1], dtype=sym.dtype)
    assert float(m.dist(frechet_mean(m, sym, dim=0), origin)) < 1e-6


def test_frechet_mean_is_isometry_invariant_between_ball_and_hyperboloid() -> None:
    ball, hyp = PoincareBall(c=-1.0), Lorentz(c=-1.0)
    v = torch.randn(8, 3, dtype=torch.float64) * 0.9
    # Divide by lambda0 so both exponential maps travel the same geodesic distance |v|.
    pb = ball.expmap0(v / ball.lambda0)
    ph = hyp.expmap0(hyp.tangent0_from_euclidean(v / hyp.lambda0))
    sq = lambda m, x, pts: (m.dist(x.unsqueeze(0), pts) ** 2).sum()  # noqa: E731
    mb, mh = frechet_mean(ball, pb, dim=0), frechet_mean(hyp, ph, dim=0)
    assert float(sq(ball, mb, pb)) == pytest.approx(float(sq(hyp, mh, ph)), rel=1e-6)
    # The means correspond under the isometry (compared through the origin log maps).
    assert torch.allclose(
        ball.logmap0(mb) * ball.lambda0,
        hyp.euclidean_from_tangent0(hyp.logmap0(mh)) * hyp.lambda0,
        atol=1e-5,
    )


def test_frechet_mean_validation() -> None:
    m = Euclidean()
    with pytest.raises(ValueError, match="at least"):
        frechet_mean(m, torch.zeros(3))
    with pytest.raises(ValueError, match="coordinate"):
        frechet_mean(m, torch.zeros(2, 3), dim=-1)
    with pytest.raises(ValueError, match="zero points"):
        frechet_mean(m, torch.zeros(0, 3), dim=0)


def test_spearman() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 3.0, 2.0])) == pytest.approx(
        0.5
    )
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman([1.0], [2.0]))
    with pytest.raises(ValueError, match="same length"):
        spearman([1, 2], [1, 2, 3])


@pytest.mark.parametrize("m", [PoincareBall(c=-1.0), Lorentz(c=-1.0)])
def test_frechet_mean_converges_where_the_unit_step_bounces(
    m: Manifold, caplog: pytest.LogCaptureFixture
) -> None:
    """Groups of points ~2 units out: a plain 'does not increase f' line search accepted unit
    steps that bounced across the minimiser and returned unconverged means."""
    torch.manual_seed(0)
    v = torch.randn(64, 8, 3, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True) * (2.0 + 0.5 * torch.rand(64, 8, 1, dtype=torch.float64))
    pts = m.expmap0(m.tangent0_from_euclidean(v / m.lambda0))
    with caplog.at_level("WARNING"):
        mu = frechet_mean(m, pts, dim=1)
    assert "did not converge" not in caplog.text
    # Stationary everywhere: the residual Riemannian gradient is below the tolerance.
    u = m.logmap(mu.unsqueeze(1), pts).mean(1)
    assert float(m.dist(mu, m.expmap(mu, u)).max()) < 1e-5
    # And at least as good as a slow, heavily damped reference descent.
    x = m.expmap0(m.logmap0(pts).mean(1))
    for _ in range(3000):
        x = m.proj(m.expmap(x, 0.2 * m.logmap(x.unsqueeze(1), pts).mean(1)))
    sq = lambda y: (m.dist(y.unsqueeze(1), pts) ** 2).sum(1)  # noqa: E731
    assert torch.all(sq(mu) <= sq(x) + 1e-9)
    assert float(m.dist(mu, x).max()) < 1e-4
    # An exhausted budget is reported, never silent.
    with caplog.at_level("WARNING"):
        frechet_mean(m, pts, dim=1, n_iter=1)
    assert "did not converge" in caplog.text


@pytest.mark.parametrize(
    "m", [PoincareBall(c=-1.0), PoincareBall(c=-4.0), Lorentz(c=-1.0), Lorentz(c=-0.1)]
)
def test_frechet_mean_of_float32_points_converges_in_float64(
    m: Manifold, caplog: pytest.LogCaptureFixture
) -> None:
    """Float32 distance noise is above the tolerance, so a float32 descent ran out of iterations
    and warned on nearly every call; the mean is computed in float64 and cast back."""
    torch.manual_seed(0)
    v = torch.randn(16, 9, 6)
    v = v / v.norm(dim=-1, keepdim=True) * (0.5 + 2.5 * torch.rand(16, 9, 1))
    pts = m.expmap0(m.tangent0_from_euclidean(v / m.lambda0))
    assert pts.dtype == torch.float32
    with caplog.at_level("WARNING"):
        mu = frechet_mean(m, pts, dim=1)
    assert "did not converge" not in caplog.text
    assert mu.dtype == torch.float32 and mu.shape == (16, pts.shape[-1])
    assert bool(m.check_point(mu).all())
    ref = frechet_mean(m, m.proj(pts.double()), dim=1)
    assert ref.dtype == torch.float64
    assert float(m.dist(m.proj(mu.double()), ref).max()) < 1e-5


def test_check_dataset_refuses_empty_data_and_short_horizons() -> None:
    from torch.utils.data import Subset

    from hyperbolic_world_model.tasks import check_rollout_horizon, dataset_horizon

    ds = synthetic_dataset()
    horizon = dataset_horizon(ds)
    assert horizon == ds.spec.horizon
    LatentRolloutTask(horizon=horizon).check_dataset(ds)
    HierarchyReconstructionTask().check_dataset(ds)
    with pytest.raises(ValueError, match=f"horizon {horizon + 1} exceeds the dataset horizon"):
        LatentRolloutTask(horizon=horizon + 1).check_dataset(ds)
    with pytest.raises(ValueError, match="my_task: task horizon"):
        check_rollout_horizon("my_task", horizon + 1, ds)
    empty = Subset(ds, [])
    for task in (LatentRolloutTask(horizon=1), HierarchyReconstructionTask()):
        with pytest.raises(ValueError, match="is empty"):
            task.check_dataset(empty)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        CompositionalGeneralizationTask(horizon=0, holdout=[["arm_a", "grasp"]])
