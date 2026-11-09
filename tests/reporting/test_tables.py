"""Report tables are pure, sorted functions of the run payloads."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from hyperbolic_world_model.reporting.tables import (
    best_curvature_table,
    collect_runs,
    dimension_efficiency_table,
    format_mean_std,
    is_result_metric,
    lower_is_better,
    results_table,
    summary_table,
    sweep_table,
    to_markdown,
)
from tests.reporting.helpers import CURVATURES, DIMS, SEEDS, write_outputs_tree


@pytest.fixture
def summary(tmp_path: Path) -> pd.DataFrame:
    return summary_table(results_table(collect_runs(write_outputs_tree(tmp_path / "outputs"))))


def test_collect_and_results_table(tmp_path: Path) -> None:
    runs = collect_runs(write_outputs_tree(tmp_path / "outputs"))
    n_runs = len(DIMS) * len(SEEDS) * (1 + len(CURVATURES))
    assert len(runs) == n_runs and all("_path" in r for r in runs)
    table = results_table(runs)
    assert list(table.columns) == [
        "experiment",
        "model",
        "geometry",
        "curvature",
        "latent_dim",
        "seed",
        "task",
        "metric",
        "value",
        "path",
    ]
    assert len(table) == n_runs * 5  # 2 + 3 metrics per run
    assert (table[["geometry", "curvature"]].notna()).all().all()
    # Sorted by every key, so regeneration is byte-stable.
    keys = ["experiment", "model", "geometry", "curvature", "latent_dim", "seed", "task", "metric"]
    assert table[keys].apply(tuple, axis=1).is_monotonic_increasing
    assert results_table([]).empty and collect_runs(tmp_path / "nothing") == []


def test_summary_table_aggregates_seeds(summary: pd.DataFrame) -> None:
    assert (summary["n_seeds"] == len(SEEDS)).all()
    row = summary[
        (summary["geometry"] == "poincare")
        & (summary["curvature"] == -1.0)
        & (summary["latent_dim"] == 8)
        & (summary["metric"] == "geodesic_error_hmax")
    ].iloc[0]
    assert row["mean"] == pytest.approx(1 / 3 - 0.2 + 0.005)
    assert row["std"] == pytest.approx(0.01 / math.sqrt(2))
    assert summary_table(pd.DataFrame()).empty


def test_sweep_table_is_a_curvature_by_dimension_grid(summary: pd.DataFrame) -> None:
    grid = sweep_table(summary, "latent_rollout", "geodesic_error_hmax")
    assert list(grid.columns) == ["model", "geometry", "curvature", "dim=8", "dim=16"]
    assert (grid["model"] == "synthetic").all()
    assert grid["geometry"].tolist() == ["euclidean", "poincare", "poincare"]
    assert grid["curvature"].tolist() == [0.0, -1.0, -0.5]
    assert all("±" in cell for cell in grid["dim=8"])
    assert (
        format_mean_std(1.0, 0.5, 1) == "1.0000"
        and format_mean_std(1.0, 0.5, 3) == "1.0000 ± 0.5000"
    )


def test_best_curvature_table_applies_the_one_std_rule(summary: pd.DataFrame) -> None:
    best = best_curvature_table(summary, "latent_rollout", "geodesic_error_hmax")
    assert best["latent_dim"].tolist() == [8, 16]
    assert (best["best_geometry"] == "poincare").all() and (best["best_curvature"] == -1.0).all()
    assert (best["improvement"] > 0.19).all() and best["exceeds_one_std"].all()
    assert (best["n_seeds"] == len(SEEDS)).all()
    # A single seed per side has no spread to exceed: the flag is never raised.
    one_seed = summary.copy()
    one_seed["n_seeds"] = 1
    single = best_curvature_table(one_seed, "latent_rollout", "geodesic_error_hmax")
    assert (single["improvement"] > 0.19).all() and not single["exceeds_one_std"].any()
    # Higher-is-better metrics pick the maximum and count the improvement the other way.
    best_map = best_curvature_table(summary, "hierarchy_reconstruction", "map")
    assert (best_map["best_curvature"] == -1.0).all() and (best_map["improvement"] > 0).all()
    # Without a Euclidean run the comparison columns are NaN and nothing "exceeds".
    only_hyp = summary[summary["geometry"] != "euclidean"]
    partial = best_curvature_table(only_hyp, "latent_rollout", "geodesic_error_hmax")
    assert partial["euclidean_mean"].isna().all() and not partial["exceeds_one_std"].any()
    assert best_curvature_table(summary, "nope", "nope").empty


def test_dimension_efficiency_table(summary: pd.DataFrame) -> None:
    eff = dimension_efficiency_table(summary, "latent_rollout", "geodesic_error_hmax")
    assert eff["label"].tolist() == ["euclidean", "poincare(K=-0.5)", "poincare(K=-1.0)"]
    assert (eff["n_dims"] == 2).all()
    euc = eff[eff["label"] == "euclidean"].iloc[0]
    assert euc["dim_to_reach"] == 16  # the threshold is its own value at the largest dimension
    assert eff[eff["label"] == "poincare(K=-1.0)"]["dim_to_reach"].iloc[0] == 8
    assert eff["auc_log2"].idxmin() == eff.index[eff["label"] == "poincare(K=-1.0)"][0]
    assert dimension_efficiency_table(summary, "nope", "nope").empty


def test_metric_directions_and_bookkeeping() -> None:
    assert lower_is_better("geodesic_error_hmax") and not lower_is_better("map")
    assert not lower_is_better("divergence_spearman") and not lower_is_better("saturation_horizon")
    assert is_result_metric("map") and not is_result_metric("map_std")
    assert not is_result_metric("n_items") and not is_result_metric("n_pairs")


def test_to_markdown_formatting() -> None:
    df = pd.DataFrame(
        {
            "latent_dim": [8.0, float("nan")],
            "value": [0.123456, 1.0],
            "ok": [True, False],
            "s": ["a", None],
        }
    )
    md = to_markdown(df)
    lines = md.splitlines()
    assert lines[0] == "| latent_dim | value | ok | s |" and lines[1] == "|---|---|---|---|"
    assert lines[2] == "| 8 | 0.1235 | yes | a |"
    assert lines[3] == "|  | 1.0000 | no |  |"
    assert to_markdown(pd.DataFrame()) == "_(no runs found)_"


def test_best_row_never_picks_a_nan_configuration(summary: pd.DataFrame) -> None:
    """A NaN mean (an undefined correlation, say) sorts last, which used to win higher-is-better."""
    s = summary.copy()
    nan_cfg = (s["task"] == "hierarchy_reconstruction") & (s["metric"] == "map")
    nan_cfg &= (s["geometry"] == "poincare") & (s["curvature"] == -0.5)
    s.loc[nan_cfg, "mean"] = math.nan
    best = best_curvature_table(s, "hierarchy_reconstruction", "map")
    assert (best["best_curvature"] == -1.0).all() and best["best_mean"].notna().all()
    # If every hyperbolic mean is NaN there is no best row, not a NaN one.
    s.loc[(s["task"] == "hierarchy_reconstruction") & (s["geometry"] != "euclidean"), "mean"] = (
        math.nan
    )
    none = best_curvature_table(s, "hierarchy_reconstruction", "map")
    assert (none["best_geometry"] == "").all() and not none["exceeds_one_std"].any()


def test_sweep_table_pools_the_spread_across_experiments() -> None:
    """Two single-seed experiments at one configuration are not reported as ``± 0``."""
    base = {
        "model": "m", "geometry": "poincare", "curvature": -1.0, "latent_dim": 8,
        "task": "t", "metric": "err",
    }  # fmt: skip
    s = pd.DataFrame(
        [
            {**base, "experiment": "e1", "mean": 1.0, "std": 0.0, "n_seeds": 1},
            {**base, "experiment": "e2", "mean": 3.0, "std": 0.0, "n_seeds": 1},
        ]
    )
    cell = sweep_table(s, "t", "err")["dim=8"].iloc[0]
    assert cell == "2.0000 ± 1.0000"
    # With within-experiment spread the pooled std combines both components.
    s["std"] = [0.5, 0.5]
    assert sweep_table(s, "t", "err")["dim=8"].iloc[0] == f"2.0000 ± {(1.25) ** 0.5:.4f}"


def test_unranked_metrics_get_no_best_or_efficiency_row() -> None:
    from hyperbolic_world_model.reporting.tables import UNRANKED_METRICS, is_ranked_metric

    assert UNRANKED_METRICS == {
        "latent_divergence_h1",
        "latent_divergence_hmax",
        "branch_separation_ratio_hmax",
        "branch_separation_ratio_mean",
    }
    # The ranked forms of branch separation are errors: lower is better.
    for metric in (
        "branch_separation_log_error",
        "branch_collapse_fraction_hmax",
        "branch_collapse_fraction_mean",
    ):
        assert is_ranked_metric(metric) and lower_is_better(metric)
    assert all(is_result_metric(m) and not is_ranked_metric(m) for m in UNRANKED_METRICS)
    assert is_ranked_metric("map") and is_ranked_metric("geodesic_error_hmax")
    assert not is_ranked_metric("n_items") and not is_ranked_metric("map_std")
