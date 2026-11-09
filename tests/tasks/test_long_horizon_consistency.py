"""Long-horizon consistency: latent divergence of branching rollouts vs frame divergence."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from hyperbolic_world_model.data.cosmos3.dataset import Cosmos3TrajectoryDataset
from hyperbolic_world_model.data.synthetic import SyntheticSpec, SyntheticTrajectoryDataset
from hyperbolic_world_model.tasks import LongHorizonConsistencyTask
from hyperbolic_world_model.tasks.long_horizon_consistency import (
    DIVERGENCES,
    RATIO_FLOOR,
    branch_pairs_of,
    branch_separation,
    saturation_horizon,
    separation_log_error,
)
from tests.tasks.helpers import (
    GEOMETRIES,
    BranchingDataset,
    synthetic_dataset,
    tiny_bundle,
    write_cosmos_fixture,
)

METRICS = {
    "divergence_spearman",
    "divergence_spearman_pixels",
    "embedded_divergence_spearman",
    "saturation_horizon",
    "saturation_fraction_of_horizon",
    "latent_divergence_h1",
    "latent_divergence_hmax",
    "latent_divergence_growth",
    "branch_separation_ratio_hmax",
    "branch_separation_ratio_mean",
    "branch_separation_log_error",
    "branch_collapse_fraction_hmax",
    "branch_collapse_fraction_mean",
    "n_pairs",
    "horizon",
}
CURVE_COLS = [
    "horizon",
    *DIVERGENCES,
    "spearman",
    "branch_separation_ratio",
    "branch_separation_ratio_median",
    "branch_collapse_fraction",
    "n_pairs",
]


def test_saturation_horizon() -> None:
    assert saturation_horizon([0.1, 0.5, 0.94, 0.96, 1.0], 0.95) == 4
    assert saturation_horizon([1.0, 0.5], 0.95) == 1
    assert saturation_horizon(torch.tensor([0.0, 0.0]), 0.95) == 1
    with pytest.raises(ValueError, match="empty"):
        saturation_horizon([], 0.95)


def test_branch_pairs_validation() -> None:
    with pytest.raises(TypeError, match="branch_pairs"):
        branch_pairs_of(TensorDataset(torch.zeros(2, 3)))
    with pytest.raises(ValueError, match="no branch pairs"):
        branch_pairs_of(BranchingDataset(n_prompts=2, n_branches=1))
    with pytest.raises(ValueError, match="no branch pairs"):
        branch_pairs_of(synthetic_dataset(n_episodes=2, horizon=2))  # unbranched by default
    assert branch_pairs_of(BranchingDataset(n_prompts=2, n_branches=3)) == [
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
    ]


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_run_on_in_memory_branches(geom: tuple[str, float]) -> None:
    ds = BranchingDataset(n_prompts=3, n_branches=3, horizon=5)
    bundle = tiny_bundle(*geom, perturb=0.3)
    res = LongHorizonConsistencyTask(horizon=4, batch_size=4).run(bundle, ds)
    assert res.geometry == geom[0] and res.curvature == geom[1]
    assert set(res.metrics) == METRICS
    assert res.metrics["n_pairs"] == 9 and res.metrics["horizon"] == 4
    assert list(res.curves.columns) == CURVE_COLS
    assert res.curves["horizon"].tolist() == [1, 2, 3, 4]
    assert res.curves["n_pairs"].tolist() == [9, 9, 9, 9]
    for col in DIVERGENCES:
        assert (res.curves[col] > 0).all()
    assert 1 <= res.metrics["saturation_horizon"] <= 4
    assert res.metrics["saturation_fraction_of_horizon"] == res.metrics["saturation_horizon"] / 4
    assert res.metrics["latent_divergence_h1"] == pytest.approx(
        res.curves["latent_divergence"].iloc[0]
    )
    assert res.metrics["latent_divergence_hmax"] == pytest.approx(
        res.curves["latent_divergence"].iloc[-1]
    )
    assert res.metrics["latent_divergence_growth"] == pytest.approx(
        res.metrics["latent_divergence_hmax"] / res.metrics["latent_divergence_h1"]
    )
    for key in (
        "divergence_spearman",
        "divergence_spearman_pixels",
        "embedded_divergence_spearman",
    ):
        assert -1 <= res.metrics[key] <= 1
    assert res.curves["spearman"].between(-1, 1).all()


def test_static_head_has_zero_latent_divergence_and_undefined_correlation() -> None:
    ds = BranchingDataset(n_prompts=2, n_branches=2, horizon=3)
    res = LongHorizonConsistencyTask(horizon=3).run(tiny_bundle("poincare", -1.0), ds)
    assert (res.curves["latent_divergence"] == 0).all()
    assert (res.curves["embedded_divergence"] > 0).all()  # the frames do differ
    assert math.isnan(res.metrics["divergence_spearman"])
    assert math.isnan(res.metrics["latent_divergence_growth"])
    assert res.metrics["saturation_horizon"] == 1
    # A static head is total branch collapse: zero separation, every pair collapsed, and a
    # finite worst-case log error (the ratio is floored, not infinite) so it still ranks.
    assert (res.curves["branch_separation_ratio"] == 0).all()
    assert (res.curves["branch_collapse_fraction"] == 1).all()
    assert res.metrics["branch_separation_ratio_hmax"] == 0
    assert res.metrics["branch_collapse_fraction_hmax"] == 1
    assert res.metrics["branch_collapse_fraction_mean"] == 1
    assert res.metrics["branch_separation_log_error"] == pytest.approx(-math.log(RATIO_FLOOR))


def test_rollouts_and_divergences_match_a_manual_computation() -> None:
    ds = BranchingDataset(n_prompts=1, n_branches=2, horizon=3)
    bundle = tiny_bundle("poincare", -1.0, perturb=0.3)
    res = LongHorizonConsistencyTask(horizon=3, batch_size=1).run(bundle, ds)
    m, head = bundle.manifold, bundle.predictor
    a, b = ds[0], ds[1]
    with torch.no_grad():
        ea = bundle.encoder.encode(a["frames"][None])
        eb = bundle.encoder.encode(b["frames"][None])
        za, zb = head.embed(ea), head.embed(eb)
        ra = head.rollout(za[:, 0], a["actions"][None])
        rb = head.rollout(zb[:, 0], b["actions"][None])
    assert torch.allclose(
        torch.tensor(res.curves["latent_divergence"].to_numpy(), dtype=torch.float32),
        m.dist(ra, rb)[0],
        atol=1e-5,
    )
    assert torch.allclose(
        torch.tensor(res.curves["embedded_divergence"].to_numpy(), dtype=torch.float32),
        m.dist(za[:, 1:], zb[:, 1:])[0],
        atol=1e-5,
    )
    assert torch.allclose(
        torch.tensor(res.curves["encoder_divergence"].to_numpy(), dtype=torch.float32),
        (ea[:, 1:] - eb[:, 1:]).norm(dim=-1)[0],
        atol=1e-5,
    )
    assert torch.allclose(
        torch.tensor(res.curves["pixel_divergence"].to_numpy(), dtype=torch.float32),
        (a["frames"][1:] - b["frames"][1:]).flatten(1).pow(2).mean(1).sqrt(),
        atol=1e-5,
    )
    assert res.curves["spearman"].isna().all()  # one pair per horizon: undefined


def test_horizon_is_capped_by_the_shortest_branch() -> None:
    ds = BranchingDataset(n_prompts=2, n_branches=2, horizon=3)
    res = LongHorizonConsistencyTask(horizon=10).run(tiny_bundle(perturb=0.2), ds)
    assert res.metrics["horizon"] == 3 and len(res.curves) == 3


def test_pairs_must_share_their_start_frame_up_to_video_quantisation() -> None:
    ds = BranchingDataset(n_prompts=1, n_branches=2, horizon=2)
    ds.items[1]["frames"] = ds.items[1]["frames"].clone()
    ds.items[1]["frames"][0] += 2 / 255  # two uint8 quanta, as separate mp4 decodes can differ
    res = LongHorizonConsistencyTask(horizon=2).run(tiny_bundle(perturb=0.2), ds)
    assert res.metrics["n_pairs"] == 1
    ds.items[1]["frames"][0] += 0.5  # a different frame altogether
    with pytest.raises(ValueError, match="share their start frame"):
        LongHorizonConsistencyTask(horizon=2).run(tiny_bundle(), ds)
    with pytest.raises(ValueError, match="share their start frame"):
        LongHorizonConsistencyTask(horizon=2, start_atol=0.0).run(tiny_bundle(), ds)


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="horizon"):
        LongHorizonConsistencyTask(horizon=0)
    with pytest.raises(ValueError, match="saturation_fraction"):
        LongHorizonConsistencyTask(saturation_fraction=1.5)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="collapse_threshold"):
            LongHorizonConsistencyTask(collapse_threshold=bad)


def test_run_on_the_cosmos_dataset_layout(tmp_path: Path) -> None:
    """The real loader's branch pairs (windows at offset 0 sharing a start frame) drive the task."""
    root = write_cosmos_fixture(tmp_path / "gen", n_prompts=2, n_branches=3, n_frames=6, size=8)
    ds = Cosmos3TrajectoryDataset(root, horizon=4, image_size=8)
    bundle = tiny_bundle("poincare", -0.5, frame_shape=(3, 8, 8), perturb=0.3)
    res = LongHorizonConsistencyTask(horizon=8, batch_size=2).run(bundle, ds)
    assert res.metrics["n_pairs"] == len(ds.branch_pairs()) == 6
    assert res.metrics["horizon"] == 4 and (res.curves["latent_divergence"] > 0).all()


def test_synthetic_branches_share_a_start_frame_and_drive_the_task() -> None:
    spec = SyntheticSpec(n_episodes=12, horizon=4, n_branches=3, frame_shape=(1, 8, 8))
    ds = SyntheticTrajectoryDataset(spec, split="val")
    assert len(ds.branches()) == 4 and all(len(v) == 3 for v in ds.branches().values())
    pairs = ds.branch_pairs()
    assert len(pairs) == 4 * 3
    for a, b in pairs:
        assert torch.equal(ds[a]["frames"][0], ds[b]["frames"][0])
        assert not torch.equal(ds[a]["frames"][1], ds[b]["frames"][1])
        assert ds.meta[a]["prompt_id"] == ds.meta[b]["prompt_id"]
    assert ds.meta[0]["branch_id"] == "b0" and ds.meta[2]["branch_id"] == "b2"
    with pytest.raises(ValueError, match="n_branches"):
        SyntheticTrajectoryDataset(SyntheticSpec(n_episodes=10, n_branches=3))
    res = LongHorizonConsistencyTask(horizon=4, batch_size=5).run(tiny_bundle(perturb=0.2), ds)
    assert res.metrics["n_pairs"] == 12 and res.metrics["horizon"] == 4
    # Unbranched data keeps the previous episode layout exactly.
    plain = SyntheticTrajectoryDataset(SyntheticSpec(n_episodes=12, horizon=4), split="val")
    assert plain.branch_pairs() == [] and plain.meta[5]["prompt_id"] == "p5"


def test_each_window_is_encoded_once_per_chunk() -> None:
    ds = BranchingDataset(n_prompts=2, n_branches=5, horizon=3)
    bundle = tiny_bundle(perturb=0.2)
    encoded = {"windows": 0}
    original = bundle.encoder.encode

    def counting(frames):
        encoded["windows"] += frames.shape[0]
        return original(frames)

    bundle.encoder.encode = counting  # type: ignore[method-assign]
    res = LongHorizonConsistencyTask(horizon=3, batch_size=100).run(bundle, ds)
    assert res.metrics["n_pairs"] == 20 and encoded["windows"] == len(ds) == 10


def test_synthetic_branches_share_embodiment_and_task() -> None:
    """Embodiment and task describe the scene (prompt-level); only the primitive varies per branch."""
    from hyperbolic_world_model.data.splits import split_by_combination

    spec = SyntheticSpec(n_episodes=32, horizon=2, n_branches=2, frame_shape=(1, 4, 4))
    ds = SyntheticTrajectoryDataset(spec)
    for idxs in ds.branches().values():
        assert len({(ds.meta[i]["embodiment"], ds.meta[i]["task"]) for i in idxs}) == 1
        assert len({ds.meta[i]["primitive"] for i in idxs}) == 2
    assert {m["embodiment"] for m in ds.meta} == {"arm_a", "arm_b"}
    assert len({m["task"] for m in ds.meta}) == 4
    seen, held = split_by_combination(ds.meta, [["arm_b", "grasp"]])
    assert len(held) == 8 and len(seen) == 24
    # Every held-out episode is arm_b, and no seen episode is an arm_b grasp.
    assert all(ds.meta[i]["embodiment"] == "arm_b" for i in held)
    assert ("arm_b", "grasp") not in {
        (ds.meta[i]["embodiment"], ds.meta[i]["primitive"]) for i in seen
    }
    # Unbranched data keeps the original per-episode labelling exactly.
    plain = SyntheticTrajectoryDataset(
        SyntheticSpec(n_episodes=16, horizon=2, frame_shape=(1, 4, 4))
    )
    assert [m["embodiment"] for m in plain.meta[:4]] == ["arm_a", "arm_b", "arm_a", "arm_b"]
    assert plain.meta[9]["task"] == "task_0" and plain.meta[9]["primitive"] == "grasp"


def test_branch_separation_statistics() -> None:
    # Ratio of means, per-pair median and collapse fraction on a hand-computed example.
    s = branch_separation([0.1, 1.0, 3.0], [1.0, 1.0, 2.0], collapse_threshold=0.25)
    assert s["ratio"] == pytest.approx(4.1 / 4.0)
    assert s["median"] == pytest.approx(1.0)  # per-pair ratios 0.1, 1.0, 1.5
    assert s["collapse_fraction"] == pytest.approx(1 / 3)  # only 0.1 < 0.25 * 1.0
    # The threshold is strict and scales with each pair's own embedded divergence.
    assert branch_separation([0.25, 0.5], [1.0, 2.0], 0.25)["collapse_fraction"] == 0
    assert branch_separation([0.2, 0.4], [1.0, 2.0], 0.25)["collapse_fraction"] == 1
    # A pair whose true frames coincide has no ratio: it cannot blow up the median or count
    # as collapsed, and the ratio of means stays finite.
    s = branch_separation([0.5, 0.5], [0.0, 1.0], 0.25)
    assert s["ratio"] == pytest.approx(1.0) and s["median"] == pytest.approx(0.5)
    assert s["collapse_fraction"] == 0
    # If no true branch separated at all, the ratios are undefined rather than huge.
    s = branch_separation([0.3, 0.2], [0.0, 0.0], 0.25)
    assert math.isnan(s["ratio"]) and math.isnan(s["median"]) and s["collapse_fraction"] == 0
    with pytest.raises(ValueError, match="equal length"):
        branch_separation([1.0], [1.0, 2.0], 0.25)
    with pytest.raises(ValueError, match="non-empty"):
        branch_separation([], [], 0.25)


def test_separation_log_error_is_symmetric_and_ranks_collapse_worst() -> None:
    assert separation_log_error([1.0, 1.0]) == 0
    # Collapse to a half and over-separation by two are equally wrong.
    assert separation_log_error([0.5]) == pytest.approx(separation_log_error([2.0]))
    assert separation_log_error([0.5, 2.0]) == pytest.approx(math.log(2))
    assert separation_log_error([0.0]) == pytest.approx(-math.log(RATIO_FLOOR))
    assert separation_log_error([math.nan, 2.0]) == pytest.approx(math.log(2))
    assert math.isnan(separation_log_error([math.nan]))


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_branch_separation_matches_the_divergence_curves(geom: tuple[str, float]) -> None:
    """The reported ratio is mean latent over mean embedded divergence, in every geometry."""
    ds = BranchingDataset(n_prompts=3, n_branches=3, horizon=4)
    res = LongHorizonConsistencyTask(horizon=4, batch_size=4).run(
        tiny_bundle(*geom, perturb=0.3), ds
    )
    c = res.curves
    expected = c["latent_divergence"] / c["embedded_divergence"]
    assert c["branch_separation_ratio"].to_numpy() == pytest.approx(expected.to_numpy())
    assert (c["branch_separation_ratio_median"] > 0).all()
    assert c["branch_collapse_fraction"].between(0, 1).all()
    m = res.metrics
    assert m["branch_separation_ratio_hmax"] == pytest.approx(expected.iloc[-1])
    assert m["branch_separation_ratio_mean"] == pytest.approx(expected.mean())
    assert m["branch_separation_log_error"] == pytest.approx(expected.map(math.log).abs().mean())
    assert m["branch_collapse_fraction_hmax"] == c["branch_collapse_fraction"].iloc[-1]
    assert m["branch_collapse_fraction_mean"] == pytest.approx(c["branch_collapse_fraction"].mean())


def test_collapse_threshold_controls_the_collapse_fraction() -> None:
    ds = BranchingDataset(n_prompts=3, n_branches=3, horizon=3)
    bundle = tiny_bundle("poincare", -1.0, perturb=0.3)
    fractions = [
        LongHorizonConsistencyTask(horizon=3, collapse_threshold=t)
        .run(bundle, ds)
        .metrics["branch_collapse_fraction_mean"]
        for t in (1e-6, 0.25, 1.0)
    ]
    assert fractions[0] == 0 and fractions == sorted(fractions)
    # The other metrics do not depend on the threshold.
    a = LongHorizonConsistencyTask(horizon=3, collapse_threshold=0.1).run(bundle, ds)
    b = LongHorizonConsistencyTask(horizon=3, collapse_threshold=0.9).run(bundle, ds)
    assert a.metrics["branch_separation_log_error"] == b.metrics["branch_separation_log_error"]
