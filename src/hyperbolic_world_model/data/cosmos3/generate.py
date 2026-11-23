"""Batch generation: start frame + action sequence -> video rollout with Cosmos, action-conditioned.

Standalone usage (see ``scripts/generate_cosmos3_trajectories.sh``)::

    python -m hyperbolic_world_model.data.cosmos3.generate \\
        --start-frames data/cosmos3_prompts/frames \\
        --actions data/cosmos3_prompts/actions.json \\
        --out "$DATA_ROOT/cosmos3_generated" --seed 0
    python -m ... --dry-run          # validate inputs and write the generation plan, no model

Inputs
------
``--start-frames``: a directory of images (``.png``/``.jpg``) or ``.npy`` arrays ``(H, W, 3)`` uint8.
``--actions``: a JSON file::

    {
      "action_dim": 10,
      "prompts": [
        {"prompt_id": "ep0001", "start_frame": "ep0001.png",
         "embodiment": "franka", "task": "pick_cup", "prompt": "Pick up the cup and place it on the tray.",
         "branches": [
            {"branch_id": "b0", "primitive": "reach", "actions": [[dx, dy, dz, r6d_1, ..., r6d_6, gripper], ...]},
            {"branch_id": "b1", "primitive": "grasp", "actions": [...]}
         ]}
      ]
    }

Every branch of a prompt shares the same start frame: that is what gives the long-horizon
consistency task its branching futures. Action rows are the raw per-domain layout of the chosen
``--domain-name`` (Cosmos 3's ``droid_lerobot`` domain is 10-D ``[pos_delta (3), rot6d_delta (6),
gripper (1)]``); ``action_dim`` must match that domain's width.

Outputs (under ``--out``): ``rollouts/<prompt_id>/<branch_id>/frames.npz`` (``frames`` uint8
``(T, H, W, 3)``, ``actions`` float32 ``(T - 1, a)``), ``meta.json``, an optional ``rollout.mp4``
preview when ``torchvision`` is importable, and one line per rollout appended to ``manifest.jsonl``.

The model call
--------------
Everything except the model call is plain numpy and is tested with a fake generator.
:class:`Cosmos3Generator` drives **Cosmos 3 Nano** through NVIDIA's ``cosmos-framework``
(https://github.com/nvidia/cosmos-framework, ``python -m cosmos_framework.scripts.inference``) in its
``forward_dynamics`` mode: one sample is ``{vision_path: <observation image>, action_path: <JSON rows>,
domain_name, action_chunk_size, image_size, fps, view_point, prompt, seed}`` and produces
``<out>/<name>/vision.mp4`` with ``action_chunk_size + 1`` frames, the first being the observation
(``cosmos_framework/inference/action.py::build_action_batch`` at the pinned commit). Sequences longer
than one chunk are rolled out autoregressively by feeding the last generated frame back as the next
observation; every chunk level of every rollout is batched into one JSONL so the model is loaded once
per level rather than once per chunk. The checkpoint is ``--checkpoint-path Cosmos3-Nano``
(``nvidia/Cosmos3-Nano`` on the Hub). Actions are raw per-domain rows, e.g. ``droid_lerobot``:
10-D ``[pos_delta (3), rot6d_delta (6), gripper (1)]`` (``docs/action_fd_droid_posttrain.md``). The
framework is **not** a dependency of this repository; install it per NVIDIA's setup guide on a GPU
machine, together with ``ffmpeg`` (its own prerequisite, which this module also uses to read and
write mp4). Cosmos is loaded in inference mode only; no optimiser ever sees it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import platform
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hyperbolic_world_model.data.cosmos3 import default_root

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Cosmos3-Nano"
DEFAULT_DOMAIN = "droid_lerobot"
DEFAULT_CHUNK_SIZE = 16
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LEVEL_KEYS = ("embodiment", "task", "primitive")


# ---------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Branch:
    branch_id: str
    actions: np.ndarray  # (T - 1, action_dim) float32
    primitive: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    start_frame_path: Path
    embodiment: str
    task: str
    branches: tuple[Branch, ...]
    #: Free-text task description handed to the world model as its text prompt.
    text: str = ""


def load_start_frame(path: Path) -> np.ndarray:
    """Read a start frame as ``(H, W, 3)`` uint8 from ``.png``/``.jpg`` (via Pillow) or ``.npy``."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        from PIL import Image  # Pillow ships with matplotlib, an allowed dependency

        arr = np.asarray(Image.open(path).convert("RGB"))
    else:
        raise ValueError(f"unsupported start frame {path}: expected .png, .jpg or .npy")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"start frame {path} must be (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"start frame {path} must be uint8, got {arr.dtype}")
    return arr


def load_action_spec(actions_json: Path, start_frames_dir: Path) -> list[Prompt]:
    """Parse and validate the action-sequence JSON against the start-frame directory."""
    spec = json.loads(Path(actions_json).read_text(encoding="utf-8"))
    action_dim = int(spec["action_dim"])
    if action_dim < 1:
        raise ValueError("action_dim must be >= 1")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for p in spec["prompts"]:
        pid = str(p["prompt_id"])
        if pid in seen:
            raise ValueError(f"duplicate prompt_id {pid!r}")
        seen.add(pid)
        frame_path = Path(start_frames_dir) / p["start_frame"]
        if not frame_path.exists():
            raise FileNotFoundError(f"prompt {pid}: start frame {frame_path} not found")
        branches = []
        bids: set[str] = set()
        for b in p["branches"]:
            bid = str(b["branch_id"])
            if bid in bids:
                raise ValueError(f"prompt {pid}: duplicate branch_id {bid!r}")
            bids.add(bid)
            actions = np.asarray(b["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != action_dim or actions.shape[0] < 1:
                raise ValueError(
                    f"prompt {pid} branch {bid}: actions must be (T-1 >= 1, {action_dim}), got {actions.shape}"
                )
            extra = {k: v for k, v in b.items() if k not in {"branch_id", "actions", "primitive"}}
            reserved = sorted(set(extra) & MANIFEST_FIELDS)
            if reserved:
                raise ValueError(
                    f"prompt {pid} branch {bid}: keys {reserved} are written by the generator and "
                    "cannot be set in the action spec"
                )
            branches.append(Branch(bid, actions, str(b["primitive"]), extra))
        if not branches:
            raise ValueError(f"prompt {pid}: no branches")
        prompts.append(
            Prompt(
                pid,
                frame_path,
                str(p["embodiment"]),
                str(p["task"]),
                tuple(branches),
                text=str(p.get("prompt", p["task"])),
            )
        )
    if not prompts:
        raise ValueError("no prompts in action spec")
    return prompts


# ---------------------------------------------------------------------------- generator interface
class ActionConditionedGenerator(Protocol):
    """What :func:`generate_rollouts` needs from a world model. Inference only."""

    #: Free-form description recorded in every ``meta.json`` (model, checkpoint, settings).
    model_info: dict[str, Any]

    def generate(self, start_frame: np.ndarray, actions: np.ndarray, seed: int) -> np.ndarray:
        """``(H, W, 3)`` uint8 start frame + ``(T - 1, a)`` actions -> ``(T, H, W, 3)`` uint8 video."""
        ...


@dataclass(frozen=True)
class RolloutJob:
    """One rollout to generate: used by :meth:`Cosmos3Generator.generate_batch`."""

    key: str
    start_frame: np.ndarray
    actions: np.ndarray
    seed: int
    text: str = ""


# ---------------------------------------------------------------------------- video I/O (ffmpeg)
def ffmpeg_available() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None


def write_video_ffmpeg(path: Path, frames: np.ndarray, fps: int) -> Path | None:
    """Encode ``(T, H, W, 3)`` uint8 frames to H.264 mp4 with the ``ffmpeg`` CLI; ``None`` if unavailable."""
    import subprocess

    if not ffmpeg_available():
        return None
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path),
    ]  # fmt: skip
    proc = subprocess.run(
        cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True, check=False
    )
    if proc.returncode != 0:
        return None
    return path


def read_video_ffmpeg(path: Path) -> np.ndarray:
    """Decode an mp4 to ``(T, H, W, 3)`` uint8 with ``ffprobe`` + ``ffmpeg`` (raw rgb24 over a pipe)."""
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None or not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg/ffprobe are required to read generated videos (a cosmos-framework prerequisite)"
        )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    w, h = (int(v) for v in probe.stdout.strip().split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3).copy()


# ---------------------------------------------------------------------------- Cosmos 3 adapter
class Cosmos3Generator:
    """Adapter over Cosmos 3 Nano's ``forward_dynamics`` mode in NVIDIA's cosmos-framework.

    Args:
        checkpoint: ``--checkpoint-path`` value (``Cosmos3-Nano``, a Hub id, or a local export).
        domain_name: action domain registered in ``cosmos_framework`` (``droid_lerobot``: 10-D).
        chunk_size: actions per model call (``action_chunk_size``); must be a multiple of 4 so the
            ``chunk_size + 1`` output frames satisfy the tokenizer's ``4n + 1`` constraint.
        image_size: action media resize bucket (``256`` or ``480``).
        fps: conditioning/output frame rate written into the sample.
        view_point: viewpoint tag injected into the action prompt.
        num_steps, guidance: sampler settings (``None`` keeps the framework defaults).
        guardrails: pass ``--no-guardrails`` when false (offline use).
        work_dir: where samples, inputs and framework outputs are written (default: a temp dir).
        python: interpreter that has ``cosmos_framework`` installed (default: ``sys.executable``).
        runner: ``(samples: list[dict], work_dir: Path) -> dict[name, frames (K+1, H, W, 3) uint8]``.
            Defaults to :meth:`run_framework`; tests inject a fake.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_MODEL,
        domain_name: str = DEFAULT_DOMAIN,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        image_size: int = 256,
        fps: int = 10,
        view_point: str = "ego_view",
        num_steps: int | None = None,
        guidance: float | None = None,
        guardrails: bool = False,
        work_dir: str | os.PathLike[str] | None = None,
        python: str | None = None,
        runner: Callable[[list[dict[str, Any]], Path], dict[str, np.ndarray]] | None = None,
    ) -> None:
        if chunk_size < 4 or chunk_size % 4:
            raise ValueError(
                "chunk_size must be a positive multiple of 4 (output frames = chunk_size + 1 = 4n + 1)"
            )
        self.checkpoint = checkpoint
        self.domain_name = domain_name
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)
        self.fps = int(fps)
        self.view_point = view_point
        self.num_steps = num_steps
        self.guidance = guidance
        self.guardrails = bool(guardrails)
        self._work_dir = Path(work_dir) if work_dir else None
        self.python = python
        self._runner = runner or self.run_framework
        self.model_info = {
            "backend": "cosmos_framework.scripts.inference forward_dynamics",
            "model": checkpoint,
            "domain_name": domain_name,
            "chunk_size": self.chunk_size,
            "image_size": self.image_size,
            "fps": self.fps,
            "view_point": view_point,
            "num_steps": num_steps,
            "guidance": guidance,
        }

    # ------------------------------------------------------------------ sample construction
    @property
    def work_dir(self) -> Path:
        if self._work_dir is None:
            import tempfile

            self._work_dir = Path(tempfile.mkdtemp(prefix="cosmos3_gen_"))
        self._work_dir.mkdir(parents=True, exist_ok=True)
        return self._work_dir

    def make_sample(
        self, name: str, image_path: Path, action_path: Path, seed: int, text: str
    ) -> dict[str, Any]:
        """One ``forward_dynamics`` sample as the framework's input schema expects it."""
        sample: dict[str, Any] = {
            "name": name,
            "model_mode": "forward_dynamics",
            "vision_path": str(image_path),
            "action_path": str(action_path),
            "domain_name": self.domain_name,
            "action_chunk_size": self.chunk_size,
            "image_size": self.image_size,
            "fps": self.fps,
            "view_point": self.view_point,
            "prompt": text,
            "seed": int(seed),
        }
        if self.num_steps is not None:
            sample["num_steps"] = int(self.num_steps)
        if self.guidance is not None:
            sample["guidance"] = float(self.guidance)
        return sample

    def framework_command(self, samples_file: Path, out_dir: Path) -> list[str]:
        import sys

        cmd = [
            self.python or sys.executable,
            "-m",
            "cosmos_framework.scripts.inference",
            "-i",
            str(samples_file),
            "-o",
            str(out_dir),
            "--checkpoint-path",
            self.checkpoint,
        ]
        if not self.guardrails:
            cmd.append("--no-guardrails")
        return cmd

    def run_framework(self, samples: list[dict[str, Any]], work_dir: Path) -> dict[str, np.ndarray]:
        """Write ``samples.jsonl``, run the framework CLI once, read every ``<name>/vision.mp4``."""
        import subprocess

        samples_file = work_dir / "samples.jsonl"
        samples_file.write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
        out_dir = work_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.framework_command(samples_file, out_dir)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"cosmos_framework inference failed (interpreter not found: {cmd[0]}); "
                f"is NVIDIA's cosmos-framework installed in {self.python or 'this interpreter'} "
                "and is a GPU available?"
            ) from exc
        if proc.returncode != 0:
            tail = proc.stderr[-2000:]
            raise RuntimeError(
                f"cosmos_framework inference failed (exit {proc.returncode}); is NVIDIA's cosmos-framework installed "
                f"in {self.python or 'this interpreter'} and is a GPU available?\n{tail}"
            )
        results = {}
        for s in samples:
            d = out_dir / s["name"]
            status = (
                json.loads((d / "sample_outputs.json").read_text())
                if (d / "sample_outputs.json").exists()
                else {}
            )
            video = d / "vision.mp4"
            if not video.exists():
                raise RuntimeError(f"no vision.mp4 for sample {s['name']}: {status}")
            results[s["name"]] = read_video_ffmpeg(video)
        return results

    # ------------------------------------------------------------------ rollouts
    def generate_batch(self, jobs: list[RolloutJob]) -> dict[str, np.ndarray]:
        """Roll out every job, one framework invocation per chunk level (wavefront batching).

        Returns ``{job.key: (len(actions) + 1, H, W, 3) uint8}``. Frame ``t + 1`` is the result of
        action ``t``: each chunk's first frame is the observation it was conditioned on and is
        dropped for chunks after the first.
        """
        from PIL import Image

        for j in jobs:
            if j.start_frame.ndim != 3 or j.start_frame.dtype != np.uint8:
                raise ValueError(f"{j.key}: start_frame must be (H, W, 3) uint8")
            if j.actions.ndim != 2 or j.actions.shape[0] < 1:
                raise ValueError(f"{j.key}: actions must be (T - 1 >= 1, action_dim)")
        keys = {j.key for j in jobs}
        if len(keys) != len(jobs):
            raise ValueError("job keys must be unique")
        n_levels = (
            max(int(np.ceil(j.actions.shape[0] / self.chunk_size)) for j in jobs) if jobs else 0
        )
        current: dict[str, np.ndarray] = {j.key: j.start_frame for j in jobs}
        chunks: dict[str, list[np.ndarray]] = {j.key: [] for j in jobs}
        for level in range(n_levels):
            level_dir = self.work_dir / f"level_{level:03d}"
            (level_dir / "inputs").mkdir(parents=True, exist_ok=True)
            samples, active = [], []
            for j in jobs:
                start = level * self.chunk_size
                if start >= j.actions.shape[0]:
                    continue
                chunk = j.actions[start : start + self.chunk_size].astype(np.float32)
                pad = self.chunk_size - chunk.shape[0]
                if pad > 0:  # last chunk: zero actions, frames dropped after stitching
                    chunk = np.concatenate(
                        [chunk, np.zeros((pad, chunk.shape[1]), dtype=chunk.dtype)], axis=0
                    )
                name = f"{_safe(j.key)}_c{level:03d}"
                image_path = level_dir / "inputs" / f"{name}.png"
                Image.fromarray(current[j.key]).save(image_path)
                action_path = level_dir / "inputs" / f"{name}.json"
                action_path.write_text(json.dumps(chunk.tolist()))
                samples.append(
                    self.make_sample(name, image_path, action_path, j.seed + level, j.text)
                )
                active.append((j.key, name))
            names = [name for _, name in active]
            if len(set(names)) != len(names):
                dupes = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"rollout keys collide after sanitising to framework sample names: {dupes}; "
                    "use prompt/branch ids that differ in letters, digits, '-' or '_'"
                )
            outputs = self._runner(samples, level_dir)
            for key, name in active:
                frames = outputs[name]
                if (
                    frames.ndim != 4
                    or frames.shape[0] < self.chunk_size + 1
                    or frames.dtype != np.uint8
                ):
                    raise RuntimeError(
                        f"{name}: expected >= {self.chunk_size + 1} uint8 frames, got {frames.shape} {frames.dtype}"
                    )
                frames = frames[: self.chunk_size + 1]
                chunks[key].append(frames)
                current[key] = frames[-1]
        out = {}
        for j in jobs:
            parts = [chunks[j.key][0]] + [c[1:] for c in chunks[j.key][1:]]
            out[j.key] = np.concatenate(parts, axis=0)[: j.actions.shape[0] + 1]
        return out

    def generate(self, start_frame: np.ndarray, actions: np.ndarray, seed: int) -> np.ndarray:
        return self.generate_batch([RolloutJob("single", start_frame, actions, seed)])["single"]


#: Record fields written by :func:`write_rollout` or by ``extract_latents``; an action spec may
#: not set them.
MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        "prompt_id",
        "branch_id",
        "embodiment",
        "task",
        "primitive",
        "frames_path",
        "mp4_path",
        "latents_path",
        "latents_pooling",
        "latent_shape",
        "num_frames",
        "height",
        "width",
        "action_dim",
        "seed",
        "start_frame",
        "model_info",
        "generated_at",
        "host",
    }
)


def _safe(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)


# ---------------------------------------------------------------------------- outputs
def write_rollout(
    out_root: Path,
    prompt: Prompt,
    branch: Branch,
    frames: np.ndarray,
    seed: int,
    model_info: dict[str, Any],
    write_mp4: bool = True,
    fps: int = 10,
) -> dict[str, Any]:
    """Save ``frames.npz`` + ``meta.json`` (+ optional mp4) and return the manifest record."""
    if (
        frames.ndim != 4
        or frames.shape[0] != branch.actions.shape[0] + 1
        or frames.dtype != np.uint8
    ):
        raise ValueError(
            f"frames must be (T = len(actions) + 1, H, W, 3) uint8, got {frames.shape} {frames.dtype}"
        )
    d = Path(out_root) / "rollouts" / prompt.prompt_id / branch.branch_id
    d.mkdir(parents=True, exist_ok=True)
    frames_path = d / "frames.npz"
    np.savez_compressed(frames_path, frames=frames, actions=branch.actions.astype(np.float32))
    mp4_path: str | None = None
    if write_mp4:
        mp4_path = _maybe_write_mp4(d / "rollout.mp4", frames, fps)
    record = {
        "prompt_id": prompt.prompt_id,
        "branch_id": branch.branch_id,
        "embodiment": prompt.embodiment,
        "task": prompt.task,
        "primitive": branch.primitive,
        "frames_path": frames_path.relative_to(out_root).as_posix(),
        "mp4_path": Path(mp4_path).relative_to(out_root).as_posix() if mp4_path else None,
        "latents_path": None,
        "latents_pooling": None,
        "latent_shape": None,
        "num_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "action_dim": int(branch.actions.shape[1]),
        "seed": int(seed),
        "start_frame": str(prompt.start_frame_path.name),
        "model_info": model_info,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
    }
    for key, value in branch.extra.items():  # provenance fields always win over spec extras
        record.setdefault(key, value)
    (d / "meta.json").write_text(json.dumps(record, indent=2))
    return record


def _maybe_write_mp4(path: Path, frames: np.ndarray, fps: int) -> str | None:
    """Preview mp4 via ffmpeg; ``None`` (never an error) when ffmpeg is unavailable. npz is canonical."""
    try:
        written = write_video_ffmpeg(path, frames, fps)
    except Exception:
        written = None
    if written is None and ffmpeg_available():
        # yuv420p needs even dimensions; a failed encode is otherwise invisible in the manifest.
        log.warning("no mp4 preview for %s (ffmpeg failed; frames are %s)", path, frames.shape)
    return None if written is None else str(written)


def append_manifest(out_root: Path, records: Iterable[dict[str, Any]]) -> Path:
    path = Path(out_root) / "manifest.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def write_manifest(out_root: Path, records: Iterable[dict[str, Any]]) -> Path:
    """Replace ``manifest.jsonl`` with ``records`` atomically (temporary file, then rename).

    The manifest is the source of truth for the dataset, so a rewrite interrupted half-way must
    leave the previous file intact rather than a truncated one.
    """
    path = Path(out_root) / "manifest.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _drop_from_manifest(out_root: Path, keys: set[tuple[str, str]]) -> None:
    """Rewrite ``manifest.jsonl`` without the records whose ``(prompt_id, branch_id)`` is in ``keys``."""
    if not (Path(out_root) / "manifest.jsonl").exists():
        return
    kept = [r for r in read_manifest(out_root) if (r["prompt_id"], r["branch_id"]) not in keys]
    write_manifest(out_root, kept)


def read_manifest(out_root: Path) -> list[dict[str, Any]]:
    path = Path(out_root) / "manifest.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def branch_seed(base_seed: int, prompt_id: str, branch_id: str) -> int:
    """Deterministic per-branch seed: same inputs -> same seed, different branches -> different seeds."""
    import zlib

    return (int(base_seed) * 1_000_003 + zlib.crc32(f"{prompt_id}/{branch_id}".encode())) % (
        2**31 - 1
    )


# ---------------------------------------------------------------------------- driver
def generate_rollouts(
    prompts: list[Prompt],
    generator: ActionConditionedGenerator,
    out_root: Path,
    seed: int = 0,
    write_mp4: bool = True,
    fps: int = 10,
    skip_existing: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate every branch of every prompt and append the manifest. Idempotent with ``skip_existing``."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    done = (
        {(r["prompt_id"], r["branch_id"]) for r in read_manifest(out_root)}
        if skip_existing
        else set()
    )
    todo: list[tuple[Prompt, Branch, RolloutJob]] = []
    for prompt in prompts:
        start = load_start_frame(prompt.start_frame_path)
        for branch in prompt.branches:
            if (prompt.prompt_id, branch.branch_id) in done:
                continue
            s = branch_seed(seed, prompt.prompt_id, branch.branch_id)
            todo.append(
                (
                    prompt,
                    branch,
                    RolloutJob(
                        f"{prompt.prompt_id}/{branch.branch_id}",
                        start,
                        branch.actions,
                        s,
                        prompt.text,
                    ),
                )
            )
    if not todo:
        return []
    batch = getattr(generator, "generate_batch", None)
    if callable(batch):  # Cosmos 3: every rollout's chunk level shares one model invocation
        videos = batch([job for _, _, job in todo])
    else:
        videos = {
            job.key: generator.generate(job.start_frame, job.actions, job.seed)
            for _, _, job in todo
        }
    if not skip_existing:
        # Regenerated rollouts replace their manifest records, never duplicate them. The old
        # records go only now, once the model has produced every replacement: a generation that
        # fails (no GPU, a framework error) leaves the manifest and its rollouts as they were.
        _drop_from_manifest(out_root, {(p.prompt_id, b.branch_id) for p, b, _ in todo})
    records = []
    for prompt, branch, job in todo:
        frames = videos[job.key]
        rec = write_rollout(
            out_root, prompt, branch, frames, job.seed, generator.model_info, write_mp4, fps
        )
        append_manifest(out_root, [rec])
        records.append(rec)
        if progress:
            progress(f"{job.key}: {frames.shape[0]} frames (seed {job.seed})")
    return records


def write_plan(prompts: list[Prompt], out_root: Path, seed: int, model: str) -> Path:
    """Dry run: record what would be generated without loading any model."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": model,
        "seed": seed,
        "rollouts": [
            {
                "prompt_id": p.prompt_id,
                "branch_id": b.branch_id,
                "num_frames": int(b.actions.shape[0]) + 1,
                "action_dim": int(b.actions.shape[1]),
                "seed": branch_seed(seed, p.prompt_id, b.branch_id),
                "embodiment": p.embodiment,
                "task": p.task,
                "primitive": b.primitive,
            }
            for p in prompts
            for b in p.branches
        ],
    }
    path = out_root / "generation_plan.json"
    path.write_text(json.dumps(plan, indent=2))
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--start-frames",
        type=Path,
        required=True,
        help="directory of start frames (.png/.jpg/.npy)",
    )
    p.add_argument(
        "--actions", type=Path, required=True, help="JSON file of action sequences (see module doc)"
    )
    p.add_argument(
        "--out", type=Path, default=None, help="output root; default $DATA_ROOT/cosmos3_generated"
    )
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_MODEL,
        help="cosmos-framework --checkpoint-path (Cosmos3-Nano)",
    )
    p.add_argument(
        "--domain-name", default=DEFAULT_DOMAIN, help="action domain (droid_lerobot: 10-D rows)"
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="actions per model call (multiple of 4)",
    )
    p.add_argument(
        "--image-size", type=int, default=256, help="action media resize bucket (256 or 480)"
    )
    p.add_argument(
        "--fps",
        type=int,
        default=10,
        help="conditioning/output fps (also used for the mp4 preview)",
    )
    p.add_argument("--view-point", default="ego_view")
    p.add_argument(
        "--num-steps", type=int, default=None, help="denoising steps; default: framework preset"
    )
    p.add_argument(
        "--guidance", type=float, default=None, help="guidance; default: framework preset"
    )
    p.add_argument(
        "--guardrails",
        action="store_true",
        help="keep NVIDIA's guardrails on (default: --no-guardrails)",
    )
    p.add_argument(
        "--work-dir", type=Path, default=None, help="scratch dir for framework inputs/outputs"
    )
    p.add_argument("--python", default=None, help="interpreter with cosmos_framework installed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument(
        "--overwrite", action="store_true", help="regenerate rollouts already in the manifest"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="validate inputs and write generation_plan.json only"
    )
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None, generator: ActionConditionedGenerator | None = None
) -> list[dict[str, Any]]:
    args = parse_args(argv)
    out_root = args.out or default_root()
    prompts = load_action_spec(args.actions, args.start_frames)
    if args.dry_run:
        path = write_plan(prompts, out_root, args.seed, args.checkpoint)
        print(f"dry run: {sum(len(p.branches) for p in prompts)} rollouts planned -> {path}")
        return []
    gen = generator or Cosmos3Generator(
        checkpoint=args.checkpoint,
        domain_name=args.domain_name,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        fps=args.fps,
        view_point=args.view_point,
        num_steps=args.num_steps,
        guidance=args.guidance,
        guardrails=args.guardrails,
        work_dir=args.work_dir,
        python=args.python,
    )
    records = generate_rollouts(
        prompts,
        gen,
        out_root,
        seed=args.seed,
        write_mp4=not args.no_mp4,
        fps=args.fps,
        skip_existing=not args.overwrite,
        progress=print,
    )
    print(
        f"generated {len(records)} rollouts under {out_root} (manifest: {Path(out_root) / 'manifest.jsonl'})"
    )
    return records


if __name__ == "__main__":
    main()

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_DOMAIN",
    "DEFAULT_MODEL",
    "MANIFEST_FIELDS",
    "ActionConditionedGenerator",
    "Branch",
    "Cosmos3Generator",
    "Prompt",
    "RolloutJob",
    "append_manifest",
    "branch_seed",
    "ffmpeg_available",
    "generate_rollouts",
    "load_action_spec",
    "load_start_frame",
    "main",
    "read_manifest",
    "read_video_ffmpeg",
    "write_manifest",
    "write_plan",
    "write_rollout",
    "write_video_ffmpeg",
]
