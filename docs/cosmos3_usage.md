# How Cosmos 3 is used, and how it is not

NVIDIA Cosmos 3 (the Nano model, `nvidia/Cosmos3-Nano`, driven through NVIDIA's
[cosmos-framework](https://github.com/nvidia/cosmos-framework)) appears in this repository in exactly two roles, both inference-only:
**a data generator** and **a latent-space probe target**. Three rules are non-negotiable:

1. **We never fine-tune, post-train, distil, quantise-for-training, or otherwise modify Cosmos.**
   No optimiser is ever constructed over its parameters. The adapters in
   `src/hyperbolic_world_model/data/cosmos3/` disable gradients before any call and there is no
   training entry point that accepts the model.
2. **We never report Cosmos's generation quality as a result.** No table or figure in the report
   scores the generated videos. Cosmos is an instrument for producing controlled inputs, not a
   subject of the evaluation. If a generated rollout is visibly wrong, the fix is to regenerate or
   exclude it, and the manifest records what was excluded; the quality itself is not a finding.
3. **No Cosmos weights are redistributed.** Users download them from the Hugging Face Hub under
   NVIDIA's terms (OpenMDW 1.1 for the Cosmos 3 release). Generated videos and latents live under
   `data/` and are gitignored.

## Role 1: action-conditioned trajectory generator

`generate.py` takes a directory of start frames and a JSON file of action sequences and writes
video rollouts plus metadata to `DATA_ROOT/cosmos3_generated/`. Every branch of a prompt shares the
same start frame, so the output contains **branching futures** that real datasets rarely have:
the long-horizon consistency task pairs branches and measures how fast latents diverge.

```bash
bash scripts/generate_cosmos3_trajectories.sh --start-frames prompts/frames --actions prompts/actions.json   # Cosmos3-Nano, droid_lerobot
# or, to validate the inputs and see the plan without loading a model:
python -m hyperbolic_world_model.data.cosmos3.generate --start-frames ... --actions ... --dry-run
```

Output layout:

```text
DATA_ROOT/cosmos3_generated/
├── rollouts/<prompt_id>/<branch_id>/frames.npz   # frames uint8 (T, H, W, 3), actions float32 (T-1, a)
├── rollouts/<prompt_id>/<branch_id>/rollout.mp4  # preview only, written if torchvision is importable
├── rollouts/<prompt_id>/<branch_id>/meta.json    # labels, seed, model key and sampler settings, timestamp
├── latents/<prompt_id>/<branch_id>.npz           # tokenizer latents float16 (T', D) + meta (after extraction)
├── manifest.jsonl                                # one line per rollout; the dataset reads only this
└── generation_plan.json                          # written by --dry-run
```

The generated videos are then encoded by the frozen V-JEPA 2-AC or DINOv2 encoder like any other
video and used to train and evaluate *our* predictor heads. Cosmos is the data source, not the
subject.

## Role 2: latent-space probe target

`extract_latents.py` runs the Cosmos video tokenizer's **encoder** (the decoder is never loaded)
on every rollout and stores pooled latents. `metrics/gromov_hyperbolicity.py` then estimates their
delta-hyperbolicity. The question is whether a large generative world model's own latent space is
tree-like; the answer is reported next to the same probe on V-JEPA 2-AC and DINOv2 latents, as a
property of the latent spaces, never as a statement about Cosmos's videos.

## Which Cosmos, and how the model is called

The adapters (`Cosmos3Generator` in `generate.py`, `Cosmos3TokenizerBackend` in `extract_latents.py`)
are written against NVIDIA's `cosmos-framework` at the commit we read
(`2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce`, September 2026):

- **Generation** uses the framework's `forward_dynamics` mode
  (`python -m cosmos_framework.scripts.inference -i samples.jsonl -o out --checkpoint-path Cosmos3-Nano`).
  One sample is an observation image, a JSON action file (rows of raw per-domain actions), a
  `domain_name` (`droid_lerobot`: 10-D `[pos_delta (3), rot6d_delta (6), gripper (1)]`), an
  `action_chunk_size`, an `image_size` bucket (256 or 480), `fps`, `view_point`, the text `prompt`
  and a `seed`; the framework writes `<name>/vision.mp4` with `action_chunk_size + 1` frames, the
  first being the observation (`cosmos_framework/inference/action.py::build_action_batch`).
  Longer action sequences are rolled out autoregressively: the last generated frame becomes the
  next observation. Every chunk level of every rollout is batched into one JSONL, so the model is
  loaded once per chunk level; per-chunk seeds are the branch seed plus the level. Chunks are
  stitched so that frame `t + 1` is the result of action `t`. `chunk_size` must be a multiple of 4
  because the tokenizer needs `4n + 1` frames.
- **Latents** come from the Cosmos 3 vision tokenizer, a causal Wan 2.2 VAE with 4x temporal and
  16x spatial compression, through the model's `encode` on a `(1, 3, T, H, W)` video in `[-1, 1]`
  with `T = 4n + 1`; the adapter pads `T` and crops `H`, `W` to multiples of 16. The decoder is
  never used.

`tests/data/test_cosmos3.py` replays the chunk loop against a fake framework runner and checks
sample construction, batching per level, zero padding, per-chunk seeds, chaining and stitching, and
the tokenizer backend's padding, cropping and value range. The framework itself, its GPU stack and
`ffmpeg` (its prerequisite, which we also use to read and write mp4) are **not** dependencies of
this repository; install them per NVIDIA's setup guide on the generation machine. Everything around
the model call (input validation, seeding, manifest, latent pooling, the dataset) runs and is
tested without them.

## Reproducibility of generation

Every rollout's `meta.json` and manifest line record the prompt id, branch id, per-branch seed
(derived deterministically from `--seed`, the prompt id and the branch id; each chunk adds its
level), the checkpoint, domain, chunk size, image size, sampler settings and timestamp. Re-running with the same inputs regenerates identical
rollouts; `--overwrite` forces it, otherwise already-generated rollouts are skipped. The old
manifest records are dropped only once the model has produced every replacement, and the manifest
is only ever replaced through a temporary file, so a failed or interrupted run leaves it usable.

Latent extraction writes the manifest after every rollout (`latents_path`, `latents_pooling`,
`latent_shape`), so an interrupted run resumes where it stopped. A rollout is skipped only when
its stored latents use the requested `--pooling`; asking for the other pooling re-encodes it.

The loader (`data/cosmos3/dataset.py`) standardises actions per dimension with the mean and std
of the train split's rollouts and applies the same statistics to every split, because the raw
`droid_lerobot` rows mix position deltas, rotation deltas and a gripper value on different
scales. `data.normalise_actions=false` returns the raw rows.

## Status

Scripts are ready and tested with fakes; **no generation has been run**. Running it requires a GPU
machine with NVIDIA's package installed and the checkpoint accepted on NVIDIA's terms.
