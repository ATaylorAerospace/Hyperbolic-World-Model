# Working rules for this repository

These are the rules from [CONTRIBUTING.md](CONTRIBUTING.md), the CI workflow, `pyproject.toml` and
`docs/reproducibility.md`, gathered in one place so every session applies them. CONTRIBUTING.md is
the source of truth; if the two ever disagree, follow it and fix this file.

## Before every commit

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Both must pass. CI runs the same on CPU (ruff, the unit tests under `tests/geometry`,
`tests/metrics`, `tests/data`, `tests/models`, `tests/tasks` and `tests/reporting`, then
`scripts/run_smoke.sh`). Never skip or disable a test to get green.

## Attribution and commit rules

### Identity

- Every commit must be authored AND committed as:
  - Name: `A Taylor`
  - Email: `112668339+ATaylorAerospace@users.noreply.github.com`
- Set this before the first commit in every session:
  ```bash
  git config user.name "A Taylor"
  git config user.email "112668339+ATaylorAerospace@users.noreply.github.com"
  ```
- Never use `ataylor@users.noreply.github.com` or any other variant. Only the `112668339+`
  address links commits to the GitHub account.

### No AI attribution

- Do NOT add `Co-Authored-By` trailers to commit messages.
- Do NOT add session links, "Generated with" lines or any AI/tool attribution to commit
  messages, PR titles or PR bodies.
- Do NOT put model names in commits, PRs or code comments.

### Verify before pushing

```bash
git log -1 --format='%an <%ae> | %cn <%ce>'
```

Both the author and committer must show the A Taylor identity above. If they don't, fix the
commit before pushing:

```bash
git commit --amend --no-edit --reset-author
```

### Branches and PRs

- Commit to a feature branch, never directly to `main`.
- Push the branch only. A Taylor opens and merges the PR.

### Message style

Taken from the history of `main`.

- **Subject line:** imperative mood, capitalised first word, no trailing full stop, no
  `feat:`/`fix:` prefix, no issue number. Say what changed specifically, e.g.
  "Fix Poincare distance symmetry, Cosmos 3 manifest paths, and UTF-8 report I/O". Aim for under
  about 90 characters, never more than 100; join the parts of a multi-part change with commas or
  semicolons. Avoid vague subjects such as "Update" or "Repo fixes".
- **Body:** a blank line, then prose wrapped at 100 columns that explains why the change was
  needed, not only what changed. For a change with several parts, group them under short headings
  such as "Results:" and "Correctness:" with one `-` bullet per item. A one-line change may have
  no body.
- **Test count:** when it changes, say so with real numbers from a `pytest` run:
  "Tests 493 -> 500", followed by what the new tests cover.
- **Docs:** end the body by naming the documents refreshed in the same commit (README,
  CONTRIBUTING, configs, the header graphic).
- **One commit per focused change.** A branch normally carries a single commit that is merged to
  `main` with a merge commit ("Merge pull request #N from ..."), never squashed or rebased onto
  `main` by hand, and never pushed to `main` directly.
- **Spelling:** British English in messages and docs (licence, capitalisation, initialisation).

## What may be committed

- Reports: only the Markdown tables and figures produced by `bash scripts/make_report.sh`, never
  raw outputs. `data/`, `checkpoints/`, `outputs/`, weights, `.npz` and `.mp4` files stay
  untracked.
- `uv.lock` is committed and never edited by hand; run `uv lock` after changing `pyproject.toml`.
- `.env` and other secrets are never committed; `.env.example` documents the variables.

## Branches and pull requests

- Branch from `main`; changes reach `main` through a pull request that A Taylor opens and
  merges, not a direct push.
- Keep each change focused: one geometry, one task, one metric, one fix.
- Add or update tests next to the code.
- If something is not implemented yet, raise `NotImplementedError` with a note on the plan rather
  than returning placeholder values.
- If you add a module, config option, task or test, update `README.md` in the same pull request.
- Opening a pull request is the contributor agreement described in CONTRIBUTING.md; there is
  nothing separate to sign.

## The rules that keep the science honest

1. **Distances are computed in each model's own geometry.** Heads, metrics and tasks take a
   `Manifold` and call its methods; nothing computes a Euclidean distance on curved coordinates.
   Adding a geometry means one file, one registry entry, one config and one line in the shared
   contract test (`tests/geometry/test_base.py`).
2. **Curvature is swept, never fixed.** Reported hyperbolic results come from the sweep configs
   and show the whole curve; single-curvature runs are for debugging.
3. **Encoders stay frozen.** Nothing trains or saves encoder weights; the trainer asserts this.
4. **Cosmos 3 is a data generator, never a subject.** It is loaded for inference only, never
   fine-tuned, and its output quality is never reported as a result.
5. **No new dependencies without an issue first.** The allowed set is in `pyproject.toml`.
6. **No hand-written numbers.** The README's test summary comes from a real `pytest` run and every
   report table from `scripts/make_report.sh`.

## Code style

- Python 3.11, line length 100, double quotes, formatted by `ruff format`.
- Ruff lint sets: `E`, `F`, `W`, `I`, `UP`, `B`, `NPY`, `RUF` (`E501` and `RUF012` ignored).
- Type hints and docstrings on public functions.
- Pytest runs with `--strict-markers`; register a marker in `pyproject.toml` before using it.

## Reproducibility

- Every run writes `outputs/<experiment>/<geometry,K,dim>/metrics.json` with the geometry and
  curvature next to the numbers.
- Pin an encoder `revision` in `configs/models/*.yaml` before reporting results.
- The licence is PolyForm Noncommercial 1.0.0; upstream code and weights are listed in
  `THIRD_PARTY_LICENSES.md` and nothing upstream is redistributed.
