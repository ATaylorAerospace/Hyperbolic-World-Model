"""Runs configs/experiments/smoke.yaml end to end on CPU and checks the harness invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hyperbolic_world_model.geometry import Euclidean
from hyperbolic_world_model.models.predictors import EuclideanHead, HyperbolicHead
from hyperbolic_world_model.training.train_predictor import CONFIG_DIR, run

pytestmark = pytest.mark.smoke


def _compose(tmp_path: Path, *overrides: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["experiments=smoke", f"output_dir={tmp_path}", *overrides],
        )
    return cfg


def test_smoke_config_is_tiny_and_cpu() -> None:
    cfg = _compose(Path("outputs/smoke-test"))
    assert cfg.models.name == "synthetic" and cfg.data.name == "synthetic"
    assert cfg.device == "cpu" and cfg.training.epochs <= 20
    assert cfg.geometry.name == "poincare" and cfg.geometry.curvature < 0


@pytest.mark.parametrize("geometry", ["poincare", "lorentz", "euclidean"])
def test_smoke_runs_end_to_end(tmp_path: Path, geometry: str) -> None:
    overrides = [f"geometry={geometry}"]
    if geometry == "euclidean":
        overrides.append("models.head.type=euclidean")
    cfg = _compose(tmp_path / geometry, *overrides)
    out = run(cfg)
    history = out["history"]
    assert history[-1]["loss"] < history[0]["loss"], "training must reduce the geodesic loss"
    (res,) = out["results"]
    assert res.geometry == geometry
    assert res.metrics["geodesic_error_h1"] < res.metrics["geodesic_error_hmax"]
    payload = json.loads((Path(out["output_dir"]) / "metrics.json").read_text())
    assert payload["geometry"] == geometry and "curvature" in payload
    assert (Path(out["output_dir"]) / "latent_rollout_curves.csv").exists()
    saved = OmegaConf.load(Path(out["output_dir"]) / "config.yaml")
    assert saved.geometry.name == geometry


def test_latent_cache_encodes_the_dataset_once(tmp_path: Path) -> None:
    from hyperbolic_world_model.data import build_dataset
    from hyperbolic_world_model.models.registry import build_model
    from hyperbolic_world_model.training.train_predictor import train

    cfg = _compose(tmp_path)
    ds = build_dataset(cfg.data, split="train")
    n_batches = -(-len(ds) // int(cfg.training.batch_size))
    for cache in (True, False):
        bundle = build_model(cfg, action_dim=ds.action_dim)
        calls = {"n": 0}
        orig = bundle.encoder.forward

        def counting(frames, _orig=orig, _calls=calls):
            _calls["n"] += 1
            return _orig(frames)

        bundle.encoder.forward = counting  # type: ignore[method-assign]
        cfg.training.cache_latents = cache
        history = train(bundle, ds, cfg, "cpu")
        assert history[-1]["loss"] < history[0]["loss"]
        expected = n_batches if cache else n_batches * int(cfg.training.epochs)
        assert calls["n"] == expected, (cache, calls["n"], expected)


def test_encoder_stays_frozen(tmp_path: Path) -> None:
    """The frozen-encoder invariant: build_encoder refuses frozen=false and assert_frozen catches drift."""
    from hyperbolic_world_model.models.registry import build_encoder

    cfg = OmegaConf.to_container(_compose(tmp_path).models, resolve=True)
    enc = build_encoder(cfg)
    assert all(not p.requires_grad for p in enc.parameters())
    cfg["encoder"]["frozen"] = False
    with pytest.raises(ValueError, match="frozen"):
        build_encoder(cfg)


def test_hyperbolic_head_on_euclidean_manifold_matches_euclidean_head() -> None:
    """With the flat manifold, the generic head must reduce to the baseline exactly."""
    m = Euclidean()
    kw = dict(encoder_dim=16, latent_dim=8, action_dim=4, hidden_dim=32, n_layers=2, seed=3)
    a, b = EuclideanHead(manifold=m, **kw), HyperbolicHead(manifold=m, max_step=1e9, **kw)
    b.load_state_dict(a.state_dict())
    z, act = torch.randn(5, 4, 16), torch.randn(5, 3, 4)
    pa, ta = a(z, act)
    pb, tb = b(z, act)
    assert torch.allclose(pa, pb, atol=1e-6) and torch.allclose(ta, tb)
    assert torch.allclose(a.loss(pa, ta), b.loss(pb, tb))


def test_smoke_runs_all_four_tasks_on_branched_synthetic_data(tmp_path: Path) -> None:
    """``tasks=all`` needs branching rollouts and a held-out combination; both exist on CPU."""
    cfg = _compose(
        tmp_path,
        "tasks=all",
        "data.n_branches=2",
        "data.holdout_combinations=[[arm_b,grasp]]",
        "training.epochs=3",
    )
    out = run(cfg)
    assert out["n_train"] == 48
    names = [r.task for r in out["results"]]
    assert names == [
        "latent_rollout",
        "hierarchy_reconstruction",
        "long_horizon_consistency",
        "compositional_generalization",
    ]
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["seed"] == 0 and set(payload["tasks"]) == set(names)
    assert (
        payload["tasks"]["long_horizon_consistency"]["metrics"]["n_pairs"] == 32
    )  # 64 / 2 prompts
    for name in names:
        assert (tmp_path / f"{name}_curves.csv").exists()


def test_each_run_saves_its_own_head_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sweep runs share experiment_name; the output_dir leaf keeps their checkpoints apart."""
    monkeypatch.setenv("CKPT_ROOT", str(tmp_path / "ckpt"))
    paths = []
    for seed in (0, 1):
        out_dir = tmp_path / f"run_seed{seed}"  # stands in for geometry=...,K=...,dim=...,seed=...
        cfg = _compose(out_dir, f"seed={seed}", "training.epochs=1", "training.save_head=true")
        run(cfg)
        payload = json.loads((out_dir / "metrics.json").read_text())
        paths.append(Path(payload["head_checkpoint"]))
    assert paths[0] != paths[1] and all(p.exists() for p in paths)
    assert {p.parent.name for p in paths} == {"run_seed0", "run_seed1"}
    assert paths[0].parent.parent.name == "smoke"  # checkpoints/predictors/<experiment>/<run>/
    assert set(torch.load(paths[0])) == set(torch.load(paths[1]))


def test_config_dir_resolves_from_a_checkout_or_the_wheel(tmp_path: Path) -> None:
    from hyperbolic_world_model.training.train_predictor import CONFIG_DIR, resolve_config_dir

    assert (Path(CONFIG_DIR) / "config.yaml").exists()
    # Installed layout: <site-packages>/hyperbolic_world_model/{training/train_predictor.py, configs/}
    pkg = tmp_path / "site" / "hyperbolic_world_model"
    (pkg / "training").mkdir(parents=True)
    (pkg / "configs").mkdir()
    (pkg / "configs" / "config.yaml").write_text("defaults: []\n")
    module = pkg / "training" / "train_predictor.py"
    module.write_text("")
    assert resolve_config_dir(module) == str(pkg / "configs")
    (pkg / "configs" / "config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="no Hydra config directory"):
        resolve_config_dir(module)


def test_experiment_seed_changes_the_head_initialisation(tmp_path: Path) -> None:
    """A seed sweep must vary the head init, not just the data order (same across geometries)."""
    from hyperbolic_world_model.models.registry import build_model

    def head_state(seed: int, geometry: str) -> dict:
        cfg = _compose(tmp_path / f"{geometry}{seed}", f"seed={seed}", f"geometry={geometry}")
        assert int(cfg.models.head.seed) == seed
        return {
            k: v.clone() for k, v in build_model(cfg, action_dim=3).predictor.state_dict().items()
        }

    s0, s1 = head_state(0, "poincare"), head_state(1, "poincare")
    assert any(not torch.equal(s0[k], s1[k]) for k in s0)
    assert not torch.equal(s0["projection"], s1["projection"])
    # The same seed gives the same projection and initial weights in the other geometries.
    for geometry in ("euclidean", "lorentz"):
        other = head_state(0, geometry)
        assert torch.equal(other["projection"], s0["projection"])
        assert all(torch.equal(other[k], s0[k]) for k in s0 if k != "projection")


def test_tasks_are_checked_against_the_data_before_training(tmp_path: Path) -> None:
    """``long_horizon_consistency`` on unbranched data fails in the constructor path, not after epochs."""
    cfg = _compose(tmp_path, "tasks=long_horizon_consistency", "training.epochs=50")
    with pytest.raises(ValueError, match="no branch pairs"):
        run(cfg)
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (["tasks.0.horizon=9"], "latent_rollout: task horizon 9 exceeds the dataset horizon 8"),
        (
            ["tasks=compositional_generalization", "data.holdout_combinations=[[arm_a,fly]]"],
            "not present in the data",
        ),
        (
            [
                "tasks=compositional_generalization",
                "data.holdout_combinations=[[arm_a,reach]]",
                "tasks.0.horizon=9",
            ],
            "compositional_generalization: task horizon 9 exceeds",
        ),
    ],
)
def test_rollout_tasks_fail_before_the_model_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: list[str], match: str
) -> None:
    """A horizon longer than the data, or an invalid hold-out, used to surface only when the
    task ran, after every epoch had been trained."""
    from hyperbolic_world_model.training import train_predictor

    def no_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model must not be built for a run that cannot be evaluated")

    monkeypatch.setattr(train_predictor, "build_model", no_model)
    cfg = _compose(tmp_path, "training.epochs=50", *overrides)
    with pytest.raises(ValueError, match=match):
        run(cfg)
    assert not (tmp_path / "metrics.json").exists()
