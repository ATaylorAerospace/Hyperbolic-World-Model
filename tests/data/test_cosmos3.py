"""Cosmos generation pipeline, latent extraction and dataset, exercised end to end with fakes.

No model is loaded. A fake runner stands in for NVIDIA's cosmos-framework CLI so the Cosmos 3
adapter's sample construction, wavefront chunking, padding, seeding and stitching are verified
against the ``forward_dynamics`` contract it was written for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hyperbolic_world_model.data import build_dataset
from hyperbolic_world_model.data.cosmos3 import default_root
from hyperbolic_world_model.data.cosmos3 import generate as g
from hyperbolic_world_model.data.cosmos3.dataset import Cosmos3TrajectoryDataset
from hyperbolic_world_model.data.cosmos3.extract_latents import (
    Cosmos3TokenizerBackend,
    extract_latents,
    load_latents,
    pool_latent,
)
from hyperbolic_world_model.data.cosmos3.extract_latents import main as extract_main
from hyperbolic_world_model.data.hierarchies import build_hierarchy

H = W = 32
A = 10


class FakeGenerator:
    """Deterministic stand-in: each action shifts the frame's mean brightness by its first component."""

    model_info = {"backend": "fake", "model": "fake-nano"}

    def generate(self, start_frame: np.ndarray, actions: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        frames = [start_frame]
        cur = start_frame.astype(np.float32)
        for a in actions:
            cur = np.clip(cur + 10 * a[0] + rng.normal(0, 0.5, cur.shape), 0, 255)
            frames.append(cur.astype(np.uint8))
        return np.stack(frames)


class FakeTokenizer:
    model_info = {"backend": "fake-tokenizer"}

    def encode(self, video: np.ndarray) -> np.ndarray:  # (T, H, W, 3) -> (C, T', H', W')
        t = 1 + (video.shape[0] - 1) // 4
        c, h, w = 6, video.shape[1] // 8, video.shape[2] // 8
        lat = np.zeros((c, t, h, w), dtype=np.float32)
        for k in range(t):
            frame = video[min(k * 4, video.shape[0] - 1)].astype(np.float32) / 255
            lat[:3, k] = frame[::8, ::8].transpose(2, 0, 1)
            lat[3:, k] = frame[::8, ::8].transpose(2, 0, 1) ** 2
        return lat


@pytest.fixture
def prompt_dir(tmp_path: Path) -> tuple[Path, Path]:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    from PIL import Image

    rng = np.random.default_rng(0)
    for pid in ("ep0", "ep1", "ep2"):
        img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        Image.fromarray(img).save(frames_dir / f"{pid}.png")
    np.save(frames_dir / "ep3.npy", rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8))
    spec = {
        "action_dim": A,
        "prompts": [
            {
                "prompt_id": pid,
                "start_frame": f"{pid}.{'npy' if pid == 'ep3' else 'png'}",
                "embodiment": emb,
                "task": task,
                **({"prompt": "Pick up the cup."} if pid == "ep0" else {}),
                "branches": [
                    {
                        "branch_id": "b0",
                        "primitive": "reach",
                        "actions": rng.normal(size=(6, A)).tolist(),
                        "note": "x",
                    },
                    {
                        "branch_id": "b1",
                        "primitive": "grasp",
                        "actions": rng.normal(size=(9, A)).tolist(),
                    },
                ],
            }
            for pid, emb, task in (
                ("ep0", "franka", "pick"),
                ("ep1", "franka", "place"),
                ("ep2", "ur5", "pick"),
                ("ep3", "ur5", "place"),
            )
        ],
    }
    spec_path = tmp_path / "actions.json"
    spec_path.write_text(json.dumps(spec))
    return frames_dir, spec_path


# ---------------------------------------------------------------------------- inputs
def test_load_action_spec_validates(prompt_dir: tuple[Path, Path], tmp_path: Path) -> None:
    frames_dir, spec_path = prompt_dir
    prompts = g.load_action_spec(spec_path, frames_dir)
    assert [p.prompt_id for p in prompts] == ["ep0", "ep1", "ep2", "ep3"]
    assert prompts[0].branches[0].actions.shape == (6, A) and prompts[0].branches[0].extra == {
        "note": "x"
    }
    assert (
        prompts[0].text == "Pick up the cup." and prompts[1].text == "place"
    )  # falls back to the task label
    assert g.load_start_frame(prompts[0].start_frame_path).shape == (H, W, 3)
    assert g.load_start_frame(prompts[3].start_frame_path).dtype == np.uint8
    spec = json.loads(spec_path.read_text())
    bad = tmp_path / "bad.json"
    spec["prompts"][0]["branches"][0]["actions"] = [[0.0] * (A - 1)]
    bad.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="actions must be"):
        g.load_action_spec(bad, frames_dir)
    spec = json.loads(spec_path.read_text())
    spec["prompts"][0]["start_frame"] = "missing.png"
    bad.write_text(json.dumps(spec))
    with pytest.raises(FileNotFoundError):
        g.load_action_spec(bad, frames_dir)
    spec = json.loads(spec_path.read_text())
    spec["prompts"].append(dict(spec["prompts"][0]))
    bad.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        g.load_action_spec(bad, frames_dir)
    with pytest.raises(ValueError, match="unsupported"):
        g.load_start_frame(tmp_path / "x.gif")


def test_branch_seed_is_deterministic_and_distinct() -> None:
    assert g.branch_seed(0, "ep0", "b0") == g.branch_seed(0, "ep0", "b0")
    assert g.branch_seed(0, "ep0", "b0") != g.branch_seed(0, "ep0", "b1")
    assert g.branch_seed(0, "ep0", "b0") != g.branch_seed(1, "ep0", "b0")
    assert 0 <= g.branch_seed(123, "p", "b") < 2**31 - 1


# ---------------------------------------------------------------------------- generation driver
def test_generate_rollouts_writes_frames_meta_and_manifest(
    prompt_dir: tuple[Path, Path], tmp_path: Path
) -> None:
    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    prompts = g.load_action_spec(spec_path, frames_dir)
    records = g.generate_rollouts(prompts, FakeGenerator(), out, seed=0, write_mp4=False)
    assert len(records) == 8
    manifest = g.read_manifest(out)
    assert [(r["prompt_id"], r["branch_id"]) for r in manifest] == [
        (p, b) for p in ("ep0", "ep1", "ep2", "ep3") for b in ("b0", "b1")
    ]
    r = manifest[1]
    assert (
        r["num_frames"] == 10
        and r["action_dim"] == A
        and r["primitive"] == "grasp"
        and r["latents_path"] is None
    )
    assert r["seed"] == g.branch_seed(0, "ep0", "b1") and r["model_info"]["model"] == "fake-nano"
    with np.load(out / r["frames_path"]) as z:
        assert (
            z["frames"].shape == (10, H, W, 3)
            and z["frames"].dtype == np.uint8
            and z["actions"].shape == (9, A)
        )
        assert np.array_equal(
            z["frames"][0], g.load_start_frame(prompts[0].start_frame_path)
        )  # shared start frame
    meta = json.loads((out / "rollouts" / "ep0" / "b1" / "meta.json").read_text())
    assert meta["embodiment"] == "franka" and meta["task"] == "pick" and "generated_at" in meta
    assert json.loads((out / "rollouts" / "ep0" / "b0" / "meta.json").read_text())["note"] == "x"
    # Branches of one prompt share the start frame but diverge.
    with (
        np.load(out / manifest[0]["frames_path"]) as z0,
        np.load(out / manifest[1]["frames_path"]) as z1,
    ):
        assert np.array_equal(z0["frames"][0], z1["frames"][0]) and not np.array_equal(
            z0["frames"][1], z1["frames"][1]
        )
    # Idempotent: a second run skips everything already in the manifest; --overwrite regenerates identically.
    assert g.generate_rollouts(prompts, FakeGenerator(), out, seed=0, write_mp4=False) == []
    again = g.generate_rollouts(
        prompts, FakeGenerator(), out, seed=0, write_mp4=False, skip_existing=False
    )
    assert len(again) == 8
    # Regenerated rollouts replace their manifest records instead of being appended twice.
    assert len(g.read_manifest(out)) == 8
    assert sorted((r["prompt_id"], r["branch_id"]) for r in g.read_manifest(out)) == sorted(
        (r["prompt_id"], r["branch_id"]) for r in manifest
    )
    with np.load(out / again[3]["frames_path"]) as z:
        with np.load(out / manifest[3]["frames_path"]) as z_old:
            assert np.array_equal(z["frames"], z_old["frames"])  # same seed -> same rollout


def test_write_rollout_rejects_bad_frames(prompt_dir: tuple[Path, Path], tmp_path: Path) -> None:
    frames_dir, spec_path = prompt_dir
    p = g.load_action_spec(spec_path, frames_dir)[0]
    with pytest.raises(ValueError, match="frames must be"):
        g.write_rollout(
            tmp_path, p, p.branches[0], np.zeros((3, H, W, 3), np.uint8), 0, {}, write_mp4=False
        )


def test_cli_dry_run_and_injected_generator(
    prompt_dir: tuple[Path, Path], tmp_path: Path, capsys
) -> None:
    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    assert (
        g.main(
            [
                "--start-frames",
                str(frames_dir),
                "--actions",
                str(spec_path),
                "--out",
                str(out),
                "--dry-run",
            ]
        )
        == []
    )
    plan = json.loads((out / "generation_plan.json").read_text())
    assert (
        len(plan["rollouts"]) == 8
        and plan["model"] == g.DEFAULT_MODEL
        and not (out / "manifest.jsonl").exists()
    )
    records = g.main(
        [
            "--start-frames",
            str(frames_dir),
            "--actions",
            str(spec_path),
            "--out",
            str(out),
            "--no-mp4",
            "--seed",
            "3",
        ],
        generator=FakeGenerator(),
    )
    assert len(records) == 8 and records[0]["seed"] == g.branch_seed(3, "ep0", "b0")
    assert "generated 8 rollouts" in capsys.readouterr().out


def test_default_root_follows_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert default_root() == tmp_path / "cosmos3_generated"
    assert default_root("/x") == Path("/x/cosmos3_generated")


# ---------------------------------------------------------------------------- Cosmos 3 adapter logic
class FakeFrameworkRunner:
    """Mimics ``Cosmos3Generator.run_framework``: records the samples it was given and returns
    ``chunk_size + 1`` frames per sample, the first being the observation image, the rest brightened
    by the first action component so chaining and stitching are observable."""

    def __init__(self, extra_frames: int = 0) -> None:
        self.calls: list[list[dict]] = []
        self.extra_frames = extra_frames

    def __call__(self, samples: list[dict], work_dir: Path) -> dict[str, np.ndarray]:
        from PIL import Image

        self.calls.append(samples)
        out = {}
        for s in samples:
            assert s["model_mode"] == "forward_dynamics"
            img = np.asarray(Image.open(s["vision_path"]).convert("RGB"))
            actions = np.asarray(json.loads(Path(s["action_path"]).read_text()), dtype=np.float32)
            assert actions.shape == (s["action_chunk_size"], A)
            frames = [img]
            cur = img.astype(np.float32)
            for a in actions:
                cur = np.clip(cur + 8 * a[0], 0, 255)
                frames.append(cur.astype(np.uint8))
            frames += [frames[-1]] * self.extra_frames
            out[s["name"]] = np.stack(frames)
        return out


def test_cosmos3_generator_wavefront_chunking_samples_and_stitching(tmp_path: Path) -> None:
    runner = FakeFrameworkRunner(extra_frames=3)
    gen = g.Cosmos3Generator(
        chunk_size=4,
        image_size=256,
        fps=5,
        num_steps=7,
        guidance=2.5,
        work_dir=tmp_path / "w",
        runner=runner,
    )
    start_a = np.full((H, W, 3), 50, dtype=np.uint8)
    start_b = np.full((H, W, 3), 90, dtype=np.uint8)
    jobs = [
        g.RolloutJob(
            "ep0/b0", start_a, np.ones((10, A), np.float32), seed=100, text="pick"
        ),  # 3 chunks (4, 4, 2+pad)
        g.RolloutJob("ep0/b1", start_a, np.ones((4, A), np.float32), seed=200),  # 1 chunk
        g.RolloutJob(
            "ep1/b0", start_b, np.ones((5, A), np.float32), seed=300
        ),  # 2 chunks (4, 1+pad)
    ]
    videos = gen.generate_batch(jobs)
    assert {k: v.shape for k, v in videos.items()} == {
        "ep0/b0": (11, H, W, 3),
        "ep0/b1": (5, H, W, 3),
        "ep1/b0": (6, H, W, 3),
    }
    # One framework invocation per chunk level; each level batches every active rollout.
    assert [len(c) for c in runner.calls] == [3, 2, 1]
    s0 = runner.calls[0][0]
    assert (
        s0["name"] == "ep0_b0_c000"
        and s0["domain_name"] == "droid_lerobot"
        and s0["action_chunk_size"] == 4
    )
    assert (
        s0["image_size"] == 256
        and s0["fps"] == 5
        and s0["view_point"] == "ego_view"
        and s0["prompt"] == "pick"
    )
    assert s0["seed"] == 100 and s0["num_steps"] == 7 and s0["guidance"] == 2.5
    assert (
        runner.calls[1][0]["seed"] == 101 and runner.calls[2][0]["seed"] == 102
    )  # per-chunk seeds
    assert runner.calls[0][1]["prompt"] == ""  # no text given
    # Frame t + 1 is the result of action t: monotone brightness, no duplicated boundary frame.
    means = videos["ep0/b0"].astype(np.float32).mean(axis=(1, 2, 3))
    assert np.array_equal(videos["ep0/b0"][0], start_a) and np.all(np.diff(means) > 0)
    # The last chunk of ep1/b0 was zero-padded to 4 actions; padded frames are dropped.
    last_actions = json.loads(Path(runner.calls[1][1]["action_path"]).read_text())
    assert runner.calls[1][1]["name"] == "ep1_b0_c001" and last_actions[1:] == [[0.0] * A] * 3
    # Each later chunk is conditioned on the previous chunk's last frame.
    from PIL import Image

    cond = np.asarray(Image.open(runner.calls[1][0]["vision_path"]).convert("RGB"))
    assert np.array_equal(cond, videos["ep0/b0"][4])
    assert gen.model_info["model"] == "Cosmos3-Nano" and gen.model_info["chunk_size"] == 4
    assert gen.generate(start_a, np.ones((3, A), np.float32), 1).shape == (4, H, W, 3)


def test_cosmos3_generator_validation_and_command() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        g.Cosmos3Generator(chunk_size=6)
    gen = g.Cosmos3Generator(
        runner=lambda s, d: {}, work_dir="/tmp/unused", guardrails=False, python="py3"
    )
    cmd = gen.framework_command(Path("s.jsonl"), Path("out"))
    assert (
        cmd[:3] == ["py3", "-m", "cosmos_framework.scripts.inference"] and "--no-guardrails" in cmd
    )
    assert cmd[cmd.index("--checkpoint-path") + 1] == "Cosmos3-Nano"
    assert "--no-guardrails" not in g.Cosmos3Generator(guardrails=True).framework_command(
        Path("s"), Path("o")
    )
    with pytest.raises(ValueError, match="uint8"):
        gen.generate_batch(
            [g.RolloutJob("k", np.zeros((H, W, 3), np.float32), np.ones((4, A), np.float32), 0)]
        )
    with pytest.raises(ValueError, match="unique"):
        gen.generate_batch(
            [g.RolloutJob("k", np.zeros((H, W, 3), np.uint8), np.ones((4, A), np.float32), 0)] * 2
        )
    short = FakeFrameworkRunner()
    gen2 = g.Cosmos3Generator(
        chunk_size=8, runner=lambda s, d: {k: v[:5] for k, v in short(s, d).items()}
    )
    with pytest.raises(RuntimeError, match="expected >= 9"):
        gen2.generate(np.zeros((H, W, 3), np.uint8), np.ones((8, A), np.float32), 0)


def test_run_framework_without_the_package_raises_actionable_error(tmp_path: Path) -> None:
    gen = g.Cosmos3Generator(work_dir=tmp_path)
    sample = gen.make_sample("s", tmp_path / "i.png", tmp_path / "a.json", 0, "")
    assert sample["model_mode"] == "forward_dynamics" and "num_steps" not in sample
    with pytest.raises(RuntimeError, match="cosmos_framework inference failed"):
        gen.run_framework([sample], tmp_path)
    missing = g.Cosmos3Generator(work_dir=tmp_path, python=str(tmp_path / "no-such-python"))
    with pytest.raises(RuntimeError, match="cosmos_framework inference failed"):
        missing.run_framework([sample], tmp_path)


@pytest.mark.skipif(
    g.ffmpeg_available(), reason="ffmpeg present; the no-ffmpeg paths are exercised elsewhere"
)
def test_video_helpers_without_ffmpeg(tmp_path: Path) -> None:
    frames = np.zeros((5, H, W, 3), np.uint8)
    assert g.write_video_ffmpeg(tmp_path / "x.mp4", frames, 5) is None
    with pytest.raises(RuntimeError, match="ffmpeg"):
        g.read_video_ffmpeg(tmp_path / "x.mp4")


@pytest.mark.skipif(not g.ffmpeg_available(), reason="needs ffmpeg")
def test_video_helpers_round_trip_with_ffmpeg(tmp_path: Path) -> None:
    """Structured frames survive the lossy mp4 round trip with the right shape and channel order.

    White noise would not: 4:2:0 chroma subsampling averages colour over 2x2 blocks, which costs
    ~45/255 per pixel on noise whatever the quality setting, so the fixture is a moving gradient.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.stack(
        [
            np.stack([(xx * 8 + 10 * t) % 256, yy * 8, (xx + yy) * 4], axis=-1).astype(np.uint8)
            for t in range(9)
        ]
    )
    path = g.write_video_ffmpeg(tmp_path / "x.mp4", frames, 5)
    assert path is not None and path.exists()
    back = g.read_video_ffmpeg(path)
    assert back.shape == frames.shape
    err = np.abs(back.astype(int) - frames.astype(int)).mean()
    swapped = np.abs(back[..., ::-1].astype(int) - frames.astype(int)).mean()
    assert err < 8 < swapped  # lossy but close, and RGB (not BGR) on both sides of the pipe


def test_generate_rollouts_uses_the_batch_path(
    prompt_dir: tuple[Path, Path], tmp_path: Path
) -> None:
    frames_dir, spec_path = prompt_dir
    runner = FakeFrameworkRunner()
    gen = g.Cosmos3Generator(chunk_size=4, work_dir=tmp_path / "w", runner=runner)
    records = g.generate_rollouts(
        g.load_action_spec(spec_path, frames_dir), gen, tmp_path / "gen", write_mp4=False
    )
    assert len(records) == 8 and records[0]["model_info"]["model"] == "Cosmos3-Nano"
    assert [len(c) for c in runner.calls] == [
        8,
        8,
        4,
    ]  # 6-action branches need 2 chunks, 9-action ones need 3
    assert (
        runner.calls[0][0]["prompt"] == "Pick up the cup."
        and runner.calls[0][2]["prompt"] == "place"
    )
    with np.load(tmp_path / "gen" / records[1]["frames_path"]) as z:
        assert z["frames"].shape == (10, H, W, 3)


# ---------------------------------------------------------------------------- latents
def test_pool_latent_and_extract(prompt_dir: tuple[Path, Path], tmp_path: Path) -> None:
    lat = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    mean = pool_latent(lat, "mean")
    assert mean.shape == (3, 2) and np.allclose(mean[0], lat[:, 0].mean(axis=(1, 2)))
    flat = pool_latent(lat, "flatten")
    assert flat.shape == (3, 8) and np.allclose(flat[1], lat[:, 1].reshape(-1))
    with pytest.raises(ValueError):
        pool_latent(lat, "max")
    with pytest.raises(ValueError):
        pool_latent(lat[0], "mean")

    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    g.generate_rollouts(
        g.load_action_spec(spec_path, frames_dir), FakeGenerator(), out, write_mp4=False
    )
    written = extract_latents(out, FakeTokenizer(), pooling="mean")
    assert len(written) == 8 and all(p.exists() for p in written)
    manifest = g.read_manifest(out)
    assert all(r["latents_path"] for r in manifest)
    lat, meta = load_latents(out / manifest[1]["latents_path"])
    assert lat.shape == (1 + 9 // 4, 6) and lat.dtype == np.float32 and meta["pooling"] == "mean"
    assert meta["primitive"] == "grasp" and meta["model_info"]["backend"] == "fake-tokenizer"
    assert extract_latents(out, FakeTokenizer()) == []  # skip existing
    assert len(extract_latents(out, FakeTokenizer(), pooling="flatten", skip_existing=False)) == 8
    lat2, meta2 = load_latents(out / manifest[1]["latents_path"])
    assert lat2.shape == (3, 6 * (H // 8) * (W // 8)) and meta2["pooling"] == "flatten"
    with pytest.raises(FileNotFoundError):
        extract_latents(tmp_path / "empty", FakeTokenizer())
    assert len(extract_main(["--root", str(out), "--overwrite"], backend=FakeTokenizer())) == 8


def test_cosmos3_tokenizer_backend_prepares_and_encodes() -> None:
    class Model:
        def encode(self, x):
            assert x.shape[:2] == (1, 3) and x.dtype == torch.bfloat16 and (x.shape[2] - 1) % 4 == 0
            assert x.shape[3] % 16 == 0 and x.shape[4] % 16 == 0
            assert float(x.float().min()) >= -1 and float(x.float().max()) <= 1
            return x[:, :, ::4, ::16, ::16].float()

    be = Cosmos3TokenizerBackend(model=Model(), device="cpu")
    video = np.random.default_rng(0).integers(
        0, 255, size=(7, 40, 50, 3), dtype=np.uint8
    )  # T=7 -> pad to 9; crop to 32x48
    prepared = be.prepare(video)
    assert prepared.shape == (9, 32, 48, 3) and np.array_equal(prepared[7], prepared[6])
    lat = be.encode(video)
    assert lat.shape == (3, 3, 2, 3) and lat.dtype == np.float32
    assert be.model_info["model"] == "Cosmos3-Nano"
    with pytest.raises(ValueError):
        be.encode(video.astype(np.float32))
    with pytest.raises(ValueError, match="too small"):
        be.prepare(np.zeros((5, 8, 8, 3), np.uint8))


def test_load_cosmos3_model_without_framework_raises_actionable_error() -> None:
    from hyperbolic_world_model.data.cosmos3.extract_latents import load_cosmos3_model

    with pytest.raises(RuntimeError, match="cosmos-framework is not installed"):
        load_cosmos3_model("Cosmos3-Nano")


# ---------------------------------------------------------------------------- dataset
@pytest.fixture
def generated(prompt_dir: tuple[Path, Path], tmp_path: Path) -> Path:
    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    g.generate_rollouts(
        g.load_action_spec(spec_path, frames_dir), FakeGenerator(), out, write_mp4=False
    )
    extract_latents(out, FakeTokenizer())
    return out


def test_dataset_windows_shapes_and_meta(generated: Path) -> None:
    ds = Cosmos3TrajectoryDataset(generated, horizon=4, stride=2, image_size=8)
    # ep*/b0 has 7 frames -> starts 0, 2 ; ep*/b1 has 10 frames -> starts 0, 2, 4 ; x4 prompts
    assert len(ds) == 4 * (2 + 3) and ds.action_dim == A
    item = ds[0]
    assert item["frames"].shape == (5, 3, 8, 8) and item["actions"].shape == (4, A)
    assert (
        item["frames"].dtype == torch.float32
        and 0 <= item["frames"].min()
        and item["frames"].max() <= 1
    )
    assert (
        item["meta"]["prompt_id"] == "ep0"
        and item["meta"]["branch_id"] == "b0"
        and item["meta"]["start"] == 0
    )
    assert item["meta"]["latents_path"].endswith("ep0/b0.npz")
    whole = Cosmos3TrajectoryDataset(generated, horizon=None)
    # Whole rollouts are truncated to the shortest (7 frames) so every item and batch stacks.
    assert len(whole) == 8 and whole.effective_horizon == 6
    assert whole[1]["frames"].shape == (7, 3, H, W) and whole[1]["actions"].shape == (6, A)
    from hyperbolic_world_model.data.synthetic import collate

    batch = collate([whole[i] for i in range(len(whole))])
    assert batch["frames"].shape == (8, 7, 3, H, W) and batch["actions"].shape == (8, 6, A)
    too_long = Cosmos3TrajectoryDataset(generated, horizon=8)
    assert len(too_long) == 4 * 2  # only the 10-frame branches fit
    with pytest.raises(FileNotFoundError):
        Cosmos3TrajectoryDataset(generated / "nope")
    with pytest.raises(ValueError):
        Cosmos3TrajectoryDataset(generated, horizon=0)
    assert "windows=" in repr(ds)


def test_dataset_branches_and_pairs_share_start_frames(generated: Path) -> None:
    ds = Cosmos3TrajectoryDataset(generated, horizon=4)
    groups = ds.branches()
    assert set(groups) == {"ep0", "ep1", "ep2", "ep3"} and all(len(v) == 2 for v in groups.values())
    for a, b in ds.branch_pairs():
        assert torch.equal(ds[a]["frames"][0], ds[b]["frames"][0]) and ds[a]["meta"]["start"] == 0
    assert len(ds.branch_pairs()) == 4
    assert len(ds.branches(start_only=False)["ep0"]) > 2


def test_dataset_compositional_split_and_hierarchy(generated: Path) -> None:
    ds = Cosmos3TrajectoryDataset(generated, horizon=4)
    assert ds.combinations() == {
        ("franka", "reach"),
        ("franka", "grasp"),
        ("ur5", "reach"),
        ("ur5", "grasp"),
    }
    seen, held = ds.split_by_combination([["franka", "grasp"]])
    assert seen and held and set(seen).isdisjoint(held) and len(seen) + len(held) == len(ds)
    assert all(ds.combination(i) == ("franka", "grasp") for i in held)
    with pytest.raises(ValueError, match="not present"):
        ds.split_by_combination([["franka", "fly"]])
    with pytest.raises(ValueError, match="removes"):
        ds.split_by_combination([["franka", "reach"], ["franka", "grasp"]])  # no franka left
    tree = build_hierarchy(ds.metadata())
    assert tree.n_nodes == 1 + 2 + 4 + 8  # root, 2 embodiments, 2x2 tasks, 2 primitives each
    lat, meta = ds.latents(0)
    assert lat.ndim == 2 and meta["prompt_id"] == "ep0"


def test_dataset_splits_and_config(generated: Path) -> None:
    parts = [
        Cosmos3TrajectoryDataset(generated, horizon=4, split=s) for s in ("train", "val", "test")
    ]
    assert sum(len(p) for p in parts) == len(Cosmos3TrajectoryDataset(generated, horizon=4))
    with pytest.raises(ValueError):
        Cosmos3TrajectoryDataset(generated, split="dev")
    cfg = {
        "name": "cosmos3_generated",
        "root": str(generated),
        "horizon": 4,
        "image_size": 8,
        "stride": 1,
    }
    ds = build_dataset(cfg, split="all")
    assert isinstance(ds, Cosmos3TrajectoryDataset) and ds[0]["frames"].shape[-1] == 8
    ds2 = Cosmos3TrajectoryDataset.from_config(
        {"root": str(generated), "horizon": None, "image_size": [8, 12]}
    )
    assert ds2[0]["frames"].shape[-2:] == (8, 12)


def test_action_spec_cannot_set_manifest_fields(
    prompt_dir: tuple[Path, Path], tmp_path: Path
) -> None:
    frames_dir, spec_path = prompt_dir
    spec = json.loads(spec_path.read_text())
    spec["prompts"][0]["branches"][0]["num_frames"] = 999
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="written by the generator"):
        g.load_action_spec(bad, frames_dir)
    # Even a Branch built directly with a reserved extra cannot override the provenance fields.
    prompts = g.load_action_spec(spec_path, frames_dir)
    p = prompts[0]
    b = g.Branch(
        p.branches[0].branch_id, p.branches[0].actions, "reach", {"num_frames": 999, "note": "x"}
    )
    frames = np.zeros((b.actions.shape[0] + 1, H, W, 3), np.uint8)
    rec = g.write_rollout(tmp_path / "o", p, b, frames, 0, {}, write_mp4=False)
    assert rec["num_frames"] == frames.shape[0] and rec["note"] == "x"
    assert g.MANIFEST_FIELDS <= set(rec)


def test_sanitised_sample_names_must_be_unique(tmp_path: Path) -> None:
    gen = g.Cosmos3Generator(chunk_size=4, work_dir=tmp_path / "w", runner=FakeFrameworkRunner())
    start = np.full((H, W, 3), 50, dtype=np.uint8)
    jobs = [
        g.RolloutJob("ep.1/b0", start, np.ones((4, A), np.float32), seed=1),
        g.RolloutJob("ep_1/b0", start, np.ones((4, A), np.float32), seed=2),
    ]
    with pytest.raises(ValueError, match="collide"):
        gen.generate_batch(jobs)


def test_whole_rollout_horizon_is_shared_across_splits_and_empty_splits_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``horizon=None`` truncates to the shortest rollout in the manifest, not in the split."""
    from hyperbolic_world_model.data.cosmos3.extract_latents import rewrite_manifest
    from tests.tasks.helpers import write_cosmos_fixture

    root = write_cosmos_fixture(tmp_path / "gen", n_prompts=6, n_branches=1, n_frames=9, size=8)
    records = g.read_manifest(root)
    for i, rec in enumerate(records):  # ep0 is the shortest rollout; it lands in one split only
        rec["num_frames"] = 5 + i
        d = root / rec["frames_path"]
        data = np.load(d)
        np.savez_compressed(d, frames=data["frames"][: 5 + i], actions=data["actions"][: 4 + i])
    rewrite_manifest(root, records)
    horizons = {
        s: Cosmos3TrajectoryDataset(root, horizon=None, split=s).effective_horizon
        for s in ("train", "val", "test", "all")
    }
    assert set(horizons.values()) == {4}
    with caplog.at_level("WARNING"):
        empty = Cosmos3TrajectoryDataset(
            root, horizon=None, split_fractions=(0.0, 0.0), split="val"
        )
    assert len(empty) == 0 and "split='val' is empty" in caplog.text


def test_action_stats_with_a_single_row_have_unit_scale() -> None:
    from hyperbolic_world_model.data.droid import ActionStats, normalise_actions

    one = torch.tensor([[1.0, -2.0, 3.0]])
    stats = ActionStats.from_actions(one)
    assert torch.equal(stats.mean, one[0]) and torch.equal(stats.std, torch.ones(3))
    assert torch.equal(normalise_actions(one, stats), torch.zeros(1, 3))
    many = torch.randn(16, 3)
    assert torch.isfinite(ActionStats.from_actions(many).std).all()


def test_overwrite_keeps_the_manifest_when_generation_fails(
    prompt_dir: tuple[Path, Path], tmp_path: Path
) -> None:
    """``--overwrite`` used to drop the old records before calling the model, so a failed
    generation left rollouts on disk that the manifest (the dataset's source of truth) had lost."""
    frames_dir, spec_path = prompt_dir
    prompts = g.load_action_spec(spec_path, frames_dir)
    out = tmp_path / "gen"
    g.generate_rollouts(prompts, FakeGenerator(), out, write_mp4=False)
    before = (out / "manifest.jsonl").read_text()

    class Broken(FakeGenerator):
        def generate(self, start_frame: np.ndarray, actions: np.ndarray, seed: int) -> np.ndarray:
            raise RuntimeError("no GPU")

    with pytest.raises(RuntimeError, match="no GPU"):
        g.generate_rollouts(prompts, Broken(), out, write_mp4=False, skip_existing=False)
    assert (out / "manifest.jsonl").read_text() == before
    assert len(Cosmos3TrajectoryDataset(out, horizon=4)) > 0
    # Rewrites go through a temporary file that is renamed over the manifest.
    g.write_manifest(out, g.read_manifest(out)[:3])
    assert len(g.read_manifest(out)) == 3 and not list(out.glob("*.tmp"))


def test_extract_latents_resumes_and_respects_the_requested_pooling(
    prompt_dir: tuple[Path, Path], tmp_path: Path
) -> None:
    from hyperbolic_world_model.data.cosmos3.extract_latents import stored_pooling

    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    g.generate_rollouts(
        g.load_action_spec(spec_path, frames_dir), FakeGenerator(), out, write_mp4=False
    )

    class Preempted(FakeTokenizer):
        calls = 0

        def encode(self, video: np.ndarray) -> np.ndarray:
            Preempted.calls += 1
            if Preempted.calls == 4:
                raise RuntimeError("pre-empted")
            return super().encode(video)

    # An interrupted run keeps the three rollouts it finished; the next run does the other five.
    with pytest.raises(RuntimeError, match="pre-empted"):
        extract_latents(out, Preempted())
    done = [r for r in g.read_manifest(out) if r["latents_path"]]
    assert len(done) == 3 and all(r["latents_pooling"] == "mean" for r in done)
    assert len(extract_latents(out, FakeTokenizer())) == 5
    assert extract_latents(out, FakeTokenizer()) == []

    # Asking for the other pooling re-encodes instead of silently keeping the mean-pooled files.
    assert len(extract_latents(out, FakeTokenizer(), pooling="flatten")) == 8
    manifest = g.read_manifest(out)
    assert all(r["latents_pooling"] == "flatten" for r in manifest)
    assert load_latents(out / manifest[0]["latents_path"])[1]["pooling"] == "flatten"
    assert extract_latents(out, FakeTokenizer(), pooling="flatten") == []

    # Manifests written before latents_pooling existed fall back to the file's own meta.
    legacy = [{k: v for k, v in r.items() if k != "latents_pooling"} for r in manifest]
    g.write_manifest(out, legacy)
    assert stored_pooling(out, legacy[0]) == "flatten"
    assert extract_latents(out, FakeTokenizer(), pooling="flatten") == []
    assert stored_pooling(out, {"latents_path": None}) is None
    with pytest.raises(ValueError, match="pooling must be"):
        extract_latents(out, FakeTokenizer(), pooling="max")


def test_dataset_standardises_actions_with_train_split_statistics(generated: Path) -> None:
    """The shared batch format promises normalised actions; the loader returned the raw rows."""
    fractions = (0.5, 0.25)
    raw = Cosmos3TrajectoryDataset(
        generated, horizon=None, normalise_actions=False, split_fractions=fractions
    )
    assert raw.action_stats is None
    train_rollouts = raw._split_indices("train", fractions)
    assert 0 < len(train_rollouts) < len(raw.records)
    rows = []
    for i in train_rollouts:
        with np.load(generated / raw.records[i]["frames_path"]) as z:
            rows.append(torch.from_numpy(z["actions"]).float())
            assert torch.equal(raw._load_rollout(i)[1], rows[-1])  # raw rows, untouched
    rows = torch.cat(rows)

    parts = {
        s: Cosmos3TrajectoryDataset(generated, horizon=None, split=s, split_fractions=fractions)
        for s in ("train", "val", "test", "all")
    }
    stats = parts["train"].action_stats
    assert stats is not None and stats.mean.shape == (A,)
    assert torch.allclose(stats.mean, rows.mean(0), atol=1e-6)
    assert torch.allclose(stats.std, rows.std(0), atol=1e-6)
    # Every split is normalised with the train statistics, never its own.
    for ds in parts.values():
        assert torch.equal(ds.action_stats.mean, stats.mean)
        assert torch.equal(ds.action_stats.std, stats.std)
    normed = torch.cat([parts["train"]._load_rollout(i)[1] for i in train_rollouts])
    assert torch.allclose(normed.mean(0), torch.zeros(A), atol=1e-5)
    assert torch.allclose(normed.std(0), torch.ones(A), atol=1e-5)
    item, raw_item = parts["all"][0], raw[0]
    assert torch.allclose(item["actions"], (raw_item["actions"] - stats.mean) / stats.std)
    assert torch.equal(item["frames"], raw_item["frames"])

    # An empty train split (a tiny dataset) falls back to every rollout instead of failing.
    tiny = Cosmos3TrajectoryDataset(generated, horizon=None, split_fractions=(0.0, 0.0))
    assert torch.isfinite(tiny.action_stats.std).all() and len(tiny) == 8
    cfg = {
        "name": "cosmos3_generated",
        "root": str(generated),
        "horizon": 4,
        "normalise_actions": False,
    }
    assert build_dataset(cfg, split="all").action_stats is None
    assert build_dataset({**cfg, "normalise_actions": True}, split="all").action_stats is not None
