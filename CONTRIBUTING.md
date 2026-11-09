# Contributing

Thanks for your interest. Issues, questions, docs fixes, extra tests and small focused pull
requests are all welcome; you do not need to understand every part of the repository to help. If
you are unsure whether something fits, open an issue and ask.

## A note on the licence

The code is under [PolyForm Noncommercial 1.0.0](LICENSE): free to use, modify and share for
research, teaching, personal and other noncommercial purposes. Commercial use needs a separate
licence from the author. Here is what that means for you as a contributor, in plain words:

- You keep the copyright on what you write and you are credited in the git history.
- You can keep using your own contribution however you like, including commercially.
- By opening a pull request you confirm the work is yours to submit and you give the project a
  perpetual, worldwide, irrevocable licence to use, modify, sublicense and relicense it, including
  under commercial terms. This lets the author offer commercial licences and, if it ever makes
  sense, move the whole project to a more permissive licence without tracking down every
  contributor. There is no separate agreement to sign; the pull request is the agreement.

## Getting set up

```bash
git clone https://github.com/ATaylorAerospace/hyperbolic-world-model
cd hyperbolic-world-model
uv sync                       # creates .venv with the locked dependencies
cp .env.example .env          # fill in HF_TOKEN only if you need gated weights
uv run pytest                 # 500 tests, CPU only, about a minute
uv run ruff check . && uv run ruff format --check .
```

Everything runs on a laptop CPU with no downloads; the real model weights are only needed for
full experiments.

## The rules that keep the science honest

These are the reasons the repository exists, so pull requests are checked against them:

1. **Distances are computed in each model's own geometry.** Heads, metrics and tasks take a
   `Manifold` and call its methods; nothing computes a Euclidean distance on curved coordinates.
   Adding a geometry means one file, one registry entry, one config and one line in the shared
   contract test (`tests/geometry/test_base.py`), which then checks it for you.
2. **Curvature is swept, never fixed.** Reported hyperbolic results come from the sweep configs
   and show the whole curve; single-curvature runs are for debugging.
3. **Encoders stay frozen.** Nothing trains or saves encoder weights; the trainer asserts this.
4. **Cosmos 3 is a data generator, never a subject.** It is loaded for inference only, never
   fine-tuned, and its output quality is never reported as a result.
5. **No new dependencies without an issue first.** The allowed set is in `pyproject.toml`.
6. **No hand-written numbers.** The README's test summary comes from a real `pytest` run and every
   report table from `scripts/make_report.sh`.

## Opening a pull request

- Branch from `main` and keep the change focused: one geometry, one task, one metric, one fix.
- Add or update tests next to the code. If something is not implemented yet, raise
  `NotImplementedError` with a note on the plan rather than returning placeholder values.
- Run `uv run pytest` and `ruff` before pushing; CI runs the same on CPU.
- If you add a module, config option, task or test, update `README.md` in the same pull request.
- Type hints and docstrings on public functions; line length is 100.

## Reporting results

Every run writes `outputs/<experiment>/<geometry,K,dim>/metrics.json` with the geometry and
curvature next to the numbers. Regenerate the report with `bash scripts/make_report.sh` and commit
only the Markdown tables and figures it produces, never raw outputs.
