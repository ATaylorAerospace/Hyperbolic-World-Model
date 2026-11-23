"""Save the Cosmos video tokenizer's latents for every generated rollout.

Usage::

    python -m hyperbolic_world_model.data.cosmos3.extract_latents \\
        --root "$DATA_ROOT/cosmos3_generated" [--pooling mean|flatten]

For each manifest entry the rollout video is encoded by the tokenizer encoder (decoder never
loaded) and saved to ``latents/<prompt_id>/<branch_id>.npz`` with ``latents`` float16 ``(T', D)``
and ``meta`` (JSON). The manifest is rewritten after every rollout with ``latents_path``,
``latents_pooling`` and ``latent_shape`` filled in, so an interrupted run resumes where it
stopped. A rollout is skipped only when its stored latents use the requested ``--pooling``;
asking for the other pooling re-encodes it.

These latents answer a question independent of any head we train: *does a large generative world
model already organise its latent space in a tree-like way?* (:mod:`metrics.gromov_hyperbolicity`).
Cosmos 3's vision tokenizer is a causal Wan 2.2 VAE with 4x temporal and 16x spatial compression
(``cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py``): ``model.encode(x)`` on a
``(1, 3, T, H, W)`` video in ``[-1, 1]`` with ``T = 4n + 1`` returns ``(1, z, 1 + (T - 1) // 4, H/16, W/16)``.
``pooling="mean"`` averages the spatial grid to ``(T', z)``, ``pooling="flatten"`` keeps it as
``(T', z * H' * W')``.

The model call is isolated in :class:`Cosmos3TokenizerBackend`: it pads ``T`` to ``4n + 1`` by
repeating the last frame, crops ``H`` and ``W`` to multiples of 16, and calls the model's ``encode``
(the decoder is never used). Everything else is numpy and is tested with a fake backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hyperbolic_world_model.data.cosmos3 import default_root
from hyperbolic_world_model.data.cosmos3.generate import (
    DEFAULT_MODEL,
    read_manifest,
    write_manifest,
)


class TokenizerBackend(Protocol):
    """What :func:`extract_latents` needs: a frozen video tokenizer encoder."""

    model_info: dict[str, Any]

    def encode(self, video: np.ndarray) -> np.ndarray:
        """``(T, H, W, 3)`` uint8 video -> latent video ``(C, T', H', W')`` float32."""
        ...


class Cosmos3TokenizerBackend:
    """Adapter over the vision VAE bundled with a Cosmos 3 checkpoint (encoder only).

    Args:
        checkpoint: ``--checkpoint-path`` value used to build the pipeline (``Cosmos3-Nano``).
        model: an object with ``encode(x)`` (tests inject a fake); built lazily from
            ``cosmos_framework`` otherwise via :func:`load_cosmos3_model`.
        device: where the input video is sent.
    """

    def __init__(
        self, checkpoint: str = DEFAULT_MODEL, model: Any | None = None, device: str | None = None
    ) -> None:
        self.checkpoint = checkpoint
        self._model = model
        self.device = device
        self.model_info = {
            "backend": "cosmos_framework OmniMoTModel.encode (Wan2.2 VAE 4x16x16)",
            "model": checkpoint,
        }

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = load_cosmos3_model(self.checkpoint)
        return self._model

    @staticmethod
    def prepare(video: np.ndarray) -> np.ndarray:
        """Pad ``T`` to ``4n + 1`` (repeat last frame) and crop ``H``, ``W`` to multiples of 16."""
        if video.ndim != 4 or video.dtype != np.uint8:
            raise ValueError("video must be (T, H, W, 3) uint8")
        t, h, w, _ = video.shape
        h16, w16 = h - h % 16, w - w % 16
        if h16 < 16 or w16 < 16:
            raise ValueError(f"video too small for a 16x spatial VAE: {(h, w)}")
        video = video[:, :h16, :w16]
        pad = (-(t - 1)) % 4
        if pad:
            video = np.concatenate([video, np.repeat(video[-1:], pad, axis=0)], axis=0)
        return video

    def encode(self, video: np.ndarray) -> np.ndarray:
        import torch

        video = self.prepare(video)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        x = (
            torch.from_numpy(np.ascontiguousarray(video)).permute(3, 0, 1, 2).unsqueeze(0)
        )  # (1, 3, T, H, W)
        x = (x.float() / 127.5 - 1.0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            latent = self.model.encode(x)  # (1, z, T', H', W')
        return latent[0].float().cpu().numpy()


def load_cosmos3_model(checkpoint: str, work_dir: str | Path | None = None) -> Any:
    """Build the framework pipeline for ``checkpoint`` and return its model (which owns the VAE).

    Mirrors ``cosmos_framework/scripts/inference.py`` at the pinned commit: setup overrides ->
    ``build_setup()`` -> ``get_inference_cls().create(setup)``; the model exposes ``encode``
    (``OmniMoTModel.encode`` -> ``tokenizer_vision_gen.encode``). Not exercised in this repository's
    tests (requires the framework and a GPU).
    """
    try:
        from cosmos_framework.inference.args import OmniSetupOverrides  # type: ignore[import-not-found]
        from cosmos_framework.inference.common.init import init_output_dir  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "NVIDIA's cosmos-framework is not installed; it is not a dependency of hyperbolic-world-model. "
            "Install it per https://github.com/nvidia/cosmos-framework (GPU required) before extracting latents."
        ) from e
    import tempfile

    out = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="cosmos3_latents_"))
    setup = OmniSetupOverrides.model_construct(
        checkpoint_path=checkpoint, output_dir=out, guardrails=False
    ).build_setup()
    init_output_dir(setup.output_dir)
    pipe = setup.get_inference_cls().create(setup)
    model = getattr(pipe, "model", None)
    if model is None or not hasattr(model, "encode"):
        raise RuntimeError(
            "cosmos-framework pipeline exposes no model.encode; the framework API may have changed"
        )
    return model


POOLINGS: tuple[str, ...] = ("mean", "flatten")


def pool_latent(latent: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """``(C, T', H', W')`` -> ``(T', D)`` with ``D = C`` (mean over space) or ``C * H' * W'`` (flatten)."""
    if latent.ndim != 4:
        raise ValueError(f"latent must be (C, T', H', W'), got {latent.shape}")
    t = latent.shape[1]
    if pooling == "mean":
        return latent.mean(axis=(2, 3)).T.astype(np.float32)  # (T', C)
    if pooling == "flatten":
        return latent.transpose(1, 0, 2, 3).reshape(t, -1).astype(np.float32)
    raise ValueError(f"pooling must be 'mean' or 'flatten', got {pooling!r}")


def save_latents(
    root: Path, record: dict[str, Any], pooled: np.ndarray, pooling: str, model_info: dict[str, Any]
) -> Path:
    d = Path(root) / "latents" / record["prompt_id"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{record['branch_id']}.npz"
    meta = {
        "prompt_id": record["prompt_id"],
        "branch_id": record["branch_id"],
        "embodiment": record["embodiment"],
        "task": record["task"],
        "primitive": record["primitive"],
        "num_frames": record["num_frames"],
        "pooling": pooling,
        "latent_shape": list(pooled.shape),
        "model_info": model_info,
    }
    np.savez_compressed(path, latents=pooled.astype(np.float16), meta=json.dumps(meta))
    return path


def load_latents(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one latents file: ``(T', D)`` float32 and its meta dict."""
    with np.load(path) as z:
        return z["latents"].astype(np.float32), json.loads(str(z["meta"]))


def rewrite_manifest(root: Path, records: list[dict[str, Any]]) -> Path:
    """Replace the manifest atomically (see :func:`generate.write_manifest`)."""
    return write_manifest(root, records)


def stored_pooling(root: Path, record: dict[str, Any]) -> str | None:
    """Pooling of the latents already saved for ``record``, or ``None`` if there are none.

    Read from the manifest record, falling back to the file's own meta for manifests written
    before ``latents_pooling`` was recorded.
    """
    rel = record.get("latents_path")
    if not rel or not (Path(root) / rel).exists():
        return None
    if record.get("latents_pooling"):
        return str(record["latents_pooling"])
    with np.load(Path(root) / rel) as z:
        return json.loads(str(z["meta"])).get("pooling")


def extract_latents(
    root: Path, backend: TokenizerBackend, pooling: str = "mean", skip_existing: bool = True
) -> list[Path]:
    """Encode every rollout in the manifest; returns the latents files written.

    With ``skip_existing`` a rollout is skipped only if its saved latents were pooled the way
    ``pooling`` asks; latents of the other pooling are replaced, never silently kept. The
    manifest is rewritten after every rollout, so a run that is interrupted (the encoder needs a
    GPU job that can be pre-empted) keeps what it has done and resumes from there.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    root = Path(root)
    records = read_manifest(root)
    if not records:
        raise FileNotFoundError(f"no manifest.jsonl under {root}; run generate.py first")
    written: list[Path] = []
    for rec in records:
        if skip_existing and stored_pooling(root, rec) == pooling:
            continue
        with np.load(root / rec["frames_path"]) as z:
            frames = z["frames"]
        latent = backend.encode(frames)
        pooled = pool_latent(latent, pooling)
        path = save_latents(root, rec, pooled, pooling, backend.model_info)
        rec["latents_path"] = path.relative_to(root).as_posix()
        rec["latents_pooling"] = pooling
        rec["latent_shape"] = list(pooled.shape)
        written.append(path)
        write_manifest(root, records)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="generated data root; default $DATA_ROOT/cosmos3_generated",
    )
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_MODEL,
        help="cosmos-framework --checkpoint-path (Cosmos3-Nano)",
    )
    p.add_argument("--pooling", choices=list(POOLINGS), default="mean")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, backend: TokenizerBackend | None = None) -> list[Path]:
    args = parse_args(argv)
    root = args.root or default_root()
    be = backend or Cosmos3TokenizerBackend(checkpoint=args.checkpoint)
    written = extract_latents(root, be, pooling=args.pooling, skip_existing=not args.overwrite)
    print(f"wrote {len(written)} latent files under {Path(root) / 'latents'}")
    return written


if __name__ == "__main__":
    main()

__all__ = [
    "POOLINGS",
    "Cosmos3TokenizerBackend",
    "TokenizerBackend",
    "extract_latents",
    "load_cosmos3_model",
    "load_latents",
    "main",
    "pool_latent",
    "rewrite_manifest",
    "save_latents",
    "stored_pooling",
]
