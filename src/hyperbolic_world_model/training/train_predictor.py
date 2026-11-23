"""Train a predictor head on top of a frozen encoder, then run the configured tasks.

Hydra entry point::

    python -m hyperbolic_world_model.training.train_predictor experiments=smoke
    python -m hyperbolic_world_model.training.train_predictor experiments=poincare_sweep --multirun

Invariants enforced here (see docs/methodology.md):
    * ``encoder.assert_frozen()`` runs before the first optimiser step and again before writing
      any checkpoint; only ``predictor.state_dict()`` is ever saved.
    * the loss is ``predictor.loss`` = squared geodesic distance in the head's geometry.
    * the geometry, curvature, latent dimension and seed are written next to every metric.
    * every configured task is built, and checked against the evaluation data, *before* the
      encoder is loaded or anything is trained, so a misconfigured task, an empty split or an
      invalid held-out combination fails in seconds.
    * when ``data.holdout_combinations`` is set, the head is trained on the seen
      (embodiment, primitive) combinations only, so the compositional-generalisation task
      evaluates combinations the head never saw.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from hyperbolic_world_model.data import build_dataset
from hyperbolic_world_model.data.synthetic import collate
from hyperbolic_world_model.models.registry import ModelBundle, build_model
from hyperbolic_world_model.tasks import Task, build_task
from hyperbolic_world_model.tasks.compositional_generalization import (
    normalise_holdout,
    split_dataset_by_combination,
)
from hyperbolic_world_model.training.riemannian_optim import build_optimizer

log = logging.getLogger(__name__)


def resolve_config_dir(module_file: str | os.PathLike[str] = __file__) -> str:
    """Hydra config directory: the source checkout's ``configs/`` or the copy shipped in the wheel.

    The wheel force-includes ``configs/`` as ``hyperbolic_world_model/configs`` so the
    ``hwm-train`` console script works from a non-editable install, not only from a checkout.
    """
    here = Path(module_file).resolve()
    candidates = [here.parents[3] / "configs", here.parents[1] / "configs"]
    for cand in candidates:
        if (cand / "config.yaml").exists():
            return str(cand)
    raise FileNotFoundError(
        "no Hydra config directory found; looked in "
        + ", ".join(str(c) for c in candidates)
        + " (a source checkout keeps configs/ at the repository root; the wheel ships a copy)"
    )


CONFIG_DIR = resolve_config_dir()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_subset(dataset: Dataset, holdout) -> Dataset:
    """The seen-combination subset of ``dataset`` when ``holdout`` is non-empty, else ``dataset``."""
    holdout = normalise_holdout(holdout)
    if not holdout:
        return dataset
    seen, held = split_dataset_by_combination(dataset, holdout)
    log.info(
        "training on %d seen items; %d items of held-out combinations %s excluded",
        len(seen),
        len(held),
        [list(c) for c in holdout],
    )
    return Subset(dataset, seen)


def build_tasks(cfg: DictConfig) -> list[Task]:
    """Instantiate every task in ``cfg.tasks`` (done before training so bad configs fail fast)."""
    return [build_task(OmegaConf.to_container(task_cfg, resolve=True)) for task_cfg in cfg.tasks]


@torch.no_grad()
def encode_dataset(
    bundle: ModelBundle, dataset: Dataset, batch_size: int, device: str
) -> TensorDataset:
    """Run the frozen encoder over the whole dataset once and return ``(latents, actions)`` tensors.

    The encoder never changes, so re-encoding every epoch only burns compute: with the 1B-parameter
    ViT-g the encoder forward dominates a head-training step. Latents live on ``device``.
    """
    bundle.encoder.assert_frozen()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    zs, acts = [], []
    for batch in loader:
        zs.append(bundle.encoder.encode(batch["frames"].to(device)))
        acts.append(batch["actions"].to(device))
    return TensorDataset(torch.cat(zs), torch.cat(acts))


def train(bundle: ModelBundle, dataset, cfg: DictConfig, device: str) -> list[dict[str, float]]:
    """Optimise the head only. Returns a per-epoch log.

    With ``training.cache_latents`` (default true) the dataset is encoded once by
    :func:`encode_dataset` and the epochs iterate over cached latents; otherwise frames are
    re-encoded every step (only useful when the latents do not fit in memory).
    """
    bundle.encoder.assert_frozen()
    head = bundle.predictor.train()
    opt = build_optimizer(
        head.trainable_parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    batch_size = int(cfg.training.batch_size)
    cache = bool(cfg.training.get("cache_latents", True))
    if cache:
        loader = DataLoader(
            encode_dataset(bundle, dataset, batch_size, device), batch_size=batch_size, shuffle=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False
        )
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.training.epochs)):
        total, n = 0.0, 0
        t0 = time.time()
        for batch in loader:
            if cache:
                z, actions = batch
            else:
                frames = batch["frames"].to(device)
                actions = batch["actions"].to(device)
                with torch.no_grad():
                    z = bundle.encoder.encode(frames)
            pred, target = head(z, actions)
            loss = head.loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.training.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    head.trainable_parameters(), float(cfg.training.grad_clip)
                )
            opt.step()
            total += loss.item() * z.shape[0]
            n += z.shape[0]
        rec = {"epoch": epoch, "loss": total / max(n, 1), "seconds": time.time() - t0}
        history.append(rec)
        log.info("epoch %d  loss=%.5f  (%.1fs)", epoch, rec["loss"], rec["seconds"])
    bundle.encoder.assert_frozen()
    return history


def evaluate(
    bundle: ModelBundle,
    dataset,
    cfg: DictConfig,
    device: str,
    tasks: list[Task] | None = None,
) -> list[Any]:
    """Run every task listed in ``cfg.tasks`` (or the pre-built ``tasks``) and return the results."""
    out = []
    for task in build_tasks(cfg) if tasks is None else tasks:
        res = task.run(bundle, dataset, device=device)
        log.info("task %s (%s, K=%s): %s", res.task, res.geometry, res.curvature, res.metrics)
        out.append(res)
    return out


def write_outputs(
    out_dir: Path,
    cfg: DictConfig,
    bundle: ModelBundle,
    history,
    results,
    n_train: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    payload = {
        "experiment": cfg.experiment_name,
        "geometry": bundle.manifold.name,
        "curvature": bundle.manifold.curvature,
        "latent_dim": bundle.predictor.latent_dim,
        "seed": int(cfg.seed),
        "model": bundle.name,
        "n_train": n_train,
        "train_history": history,
        "tasks": {r.task: r.to_json_dict() for r in results},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    for r in results:
        if r.curves is not None:
            r.curves.to_csv(out_dir / f"{r.task}_curves.csv", index=False)
    if cfg.training.save_head:
        bundle.encoder.assert_frozen()
        # One file per run: the sweep's 90 runs share experiment_name, so the output_dir leaf
        # (geometry=...,K=...,dim=...,seed=...) keeps them from overwriting each other.
        ckpt_dir = (
            Path(os.environ.get("CKPT_ROOT") or "checkpoints")
            / "predictors"
            / cfg.experiment_name
            / Path(cfg.output_dir).name
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "head.pt"
        torch.save(bundle.predictor.state_dict(), ckpt_path)
        payload["head_checkpoint"] = str(ckpt_path)
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def run(cfg: DictConfig) -> dict[str, Any]:
    """Programmatic entry point (used by ``tests/test_smoke_experiment.py``)."""
    device = os.environ.get("HWM_DEVICE", cfg.get("device", "cpu"))
    set_seed(int(cfg.seed))
    tasks = build_tasks(cfg)
    train_ds = build_dataset(cfg.data, split="train")
    eval_ds = build_dataset(cfg.data, split="val")
    for task in tasks:  # fail before training if the evaluation data cannot support a task
        task.check_dataset(eval_ds)
    # Also before the encoder is loaded: an invalid hold-out or an empty split needs no model.
    train_subset = training_subset(train_ds, cfg.data.get("holdout_combinations"))
    if len(train_subset) == 0:
        raise ValueError(
            f"the training split of data={cfg.data.name} is empty; there is nothing to train on"
        )
    bundle = build_model(cfg, action_dim=int(train_ds.action_dim), device=device)
    history = train(bundle, train_subset, cfg, device)
    results = evaluate(bundle, eval_ds, cfg, device, tasks=tasks)
    out_dir = Path(cfg.output_dir)
    write_outputs(out_dir, cfg, bundle, history, results, n_train=len(train_subset))
    return {
        "history": history,
        "results": results,
        "output_dir": str(out_dir),
        "n_train": len(train_subset),
    }


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
