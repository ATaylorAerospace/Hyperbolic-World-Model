# Methodology

This document states the hypotheses each task tests, what result would falsify each one, and the
two rules under which every number in this repository is produced.

## The two non-negotiable rules

**Rule 1: native geometry.** All distance-based metrics are reported in the native geometry of
each baseline. A Euclidean head is scored with Euclidean distance; a Poincaré head with the
Poincaré geodesic distance at the curvature it was trained with; a Lorentz head with the
hyperboloid geodesic distance. No metric ever maps one model's latents into another model's space,
because such a map would be an arbitrary choice that favours whichever geometry it targets. The
code makes this structural: every metric in `src/hyperbolic_world_model/metrics/` takes a
`Manifold` argument and calls `manifold.dist`; there is no Euclidean fallback. The task result
records `geometry` and `curvature` next to every scalar (`tasks/base.py::TaskResult`), and the
report tables never show a value without those two columns.

Where a comparison *across* geometries is needed, we compare dimensionless quantities:
the error normalised by the distance actually travelled (`normalised_geodesic_error`, which is 1
for a static predictor in every geometry), ranking metrics (mAP), scale-fitted distortion, and
relative delta-hyperbolicity.

**Rule 2: curvature is swept, never fixed.** No hyperbolic result is reported at a single
curvature. Every hyperbolic experiment is a Hydra multirun over
`geometry.curvature ∈ {-0.1, -0.25, -0.5, -1.0, -2.0, -4.0}` crossed with latent dimension and
seed (`configs/experiments/poincare_sweep.yaml`, `lorentz_sweep.yaml`). Tables report the full
curve; "best curvature" is reported alongside the curve, never instead of it. The default
`curvature: -1.0` in `configs/geometry/*.yaml` exists only so single debug runs have a value.

**Invariant: the encoder is frozen.** V-JEPA 2 and DINOv2 are loaded with every parameter set to
`requires_grad=False` and in `eval()` mode. `FrozenEncoder.assert_frozen` runs before the first
optimiser step and before any checkpoint is written; only the predictor head's state dict is ever
saved. `build_encoder` refuses a config with `encoder.frozen: false`. The test
`tests/test_smoke_experiment.py::test_encoder_stays_frozen` enforces both.

## Why hyperbolic space might help a world model

Three structures in embodied data are tree-like:

1. **Embodiment hierarchies.** Robot > arm type > gripper > primitive skill. Distances between
   siblings should be small; distances between subtrees large; the number of leaves grows
   exponentially with depth.
2. **Task decompositions.** A language instruction decomposes into sub-goals, which decompose into
   primitives. The decomposition is a tree.
3. **Branching futures.** From one state, the set of reachable states after `h` steps grows
   roughly exponentially with `h`. A latent space that must keep those futures distinguishable
   needs volume that grows exponentially with radius.

Euclidean space has polynomial volume growth, so embedding any of these with low distortion
requires dimension that grows with the size of the tree (Bourgain-type lower bounds). Hyperbolic
space has exponential volume growth and embeds trees with arbitrarily low distortion in two
dimensions (Sarkar, 2011). The hypothesis is that a predictor head whose latent lives in
hyperbolic space will (a) need fewer dimensions for the same rollout error, (b) preserve the
hierarchy better, and (c) keep branching futures separated for longer.

The counter-hypothesis is equally plausible and is what the harness must be able to show: the
frozen encoder already linearises the relevant structure, the residual dynamics are locally
Euclidean, and curvature only adds optimisation difficulty. The metrics below are chosen so that
either outcome is visible.

## Tasks, hypotheses, falsification

### Latent rollout (`tasks/latent_rollout.py`)

- **Hypothesis.** At equal latent dimension, the best swept curvature gives lower normalised
  geodesic rollout error than the Euclidean head at horizons beyond one step.
- **Metric.** `geodesic_error_{h1,hmax,mean}` and `normalised_error_{h1,hmax,mean}` in each
  model's geometry (the normalised error is the ratio of the mean rollout error to the mean
  static-baseline error at each horizon); the full curves are written next to the scalars.
- **Falsified if** no curvature in the sweep beats the Euclidean baseline's normalised error at
  any horizon `h > 1` by more than one seed standard deviation, at any latent dimension.
- **Data.** DROID (real), Cosmos 3 generated trajectories (controlled), synthetic (CI only).

### Hierarchy reconstruction (`tasks/hierarchy_reconstruction.py`)

- **Hypothesis.** Per-trajectory latents (Fréchet mean over time) recover the
  embodiment > task > primitive tree with lower average distortion and higher mAP in hyperbolic
  space than in Euclidean space at the same dimension, and `dist0` (distance from origin)
  correlates with tree depth.
- **Metric.** `average_distortion` (scale-fitted), `map` (`mean_average_precision`),
  `depth_spearman` (Spearman of tree depth against `dist0`), each with a `*_std` from
  bootstrap resamples of the trajectories within every leaf (`n_seeds`). Trajectories are pooled
  over time, and tree nodes over the trajectories in their subtree, with a Fréchet mean on the
  head's manifold (`tasks/base.py::frechet_mean`, Riemannian descent with a backtracking line
  search), never with a coordinate average.
- **Falsified if** the best curvature does not improve both distortion and mAP over Euclidean by
  more than the seed standard deviation, or if the depth correlation is not positive.
- **Data.** DROID metadata; synthetic hierarchy for CI.

### Long-horizon consistency (`tasks/long_horizon_consistency.py`)

- **Hypothesis.** For rollouts sharing a start frame but differing in actions (Cosmos 3
  branches), latent divergence tracks video divergence with higher correlation, and saturates
  later, in hyperbolic space.
- **Metric.** `divergence_spearman`: Spearman correlation, pooled over pairs and horizons,
  between the geodesic distance of the two open-loop rollouts (`latent_divergence`, in the head's
  geometry) and the Euclidean distance between the frozen encoder's outputs for the two generated
  frames (`encoder_divergence`, the encoder's own space); the per-horizon correlation is in the
  curves. `saturation_horizon`: first horizon at which the mean latent divergence reaches 95% of
  its maximum (later is better). `embedded_divergence` (the embedded true frames) and
  `pixel_divergence` (frame RMSE) are reported as controls.
- **Branch collapse.** A head can fail by rolling distinct action sequences toward one mean
  trajectory. `branch_separation_ratio` is the mean latent divergence over the mean embedded
  divergence at each horizon. Both are geodesic distances in the same manifold, so the ratio is
  unit-free and comparable across geometries and curvatures: `1` is ideal, below `1` is
  collapse, above `1` is over-separation. Because it has no preferred direction it is reported
  (`_hmax`, `_mean`, and a per-pair median in the curves) but never ranked. The ranked forms are
  `branch_separation_log_error`, the mean over horizons of `|log ratio|` (lower is better,
  symmetric in collapse and over-separation, floored so a static head ranks worst instead of
  infinite), and `branch_collapse_fraction`, the fraction of pairs whose latent divergence is
  below `collapse_threshold` (default 0.25) times their own embedded divergence (lower is
  better; `_hmax` and `_mean`). Pairs whose true frames coincide have no ratio and never count
  as collapsed.
- **Falsified if** the correlation is not higher, or saturation is not later, for the best
  curvature versus Euclidean. Branch separation is reported alongside as a diagnostic: a
  geometry whose correlation is high only because every branch collapsed shows up here.
- **Data.** Cosmos 3 generated branching trajectories (`branch_pairs`); synthetic prompts with
  `n_branches > 1` for CI.

### Compositional generalisation (`tasks/compositional_generalization.py`)

- **Hypothesis.** Holding out (embodiment, primitive) combinations, the gap between unseen and
  seen rollout error is smaller in hyperbolic space.
- **Metric.** `gap_normalised_error_hmax` = `unseen_normalised_error_hmax - seen_normalised_error_hmax`
  (native geometry, per curvature), with the raw gaps, the unseen/seen ratio and both subsets'
  full rollout summaries alongside so a smaller gap caused by worse seen error is visible. The
  trainer excludes the held-out combinations from the training set.
- **Falsified if** the gap is not smaller for the best curvature, or is smaller only because seen
  error got worse.
- **Data.** DROID or Cosmos 3 generated data with `data.holdout_combinations`; synthetic data
  with e.g. `[[arm_b, grasp]]` for CI.

### Latent-space probe: delta-hyperbolicity (`metrics/gromov_hyperbolicity.py`)

Independent of any model we train: we estimate the relative Gromov delta of (a) raw frozen
V-JEPA 2 latents, (b) raw DINOv2 latents, (c) Cosmos 3 tokenizer latents. Low delta means the
encoder already organises data tree-like, which predicts that the hyperbolic head should help;
delta near 1 predicts it should not. This is a pre-registered prediction: we report it before the
sweeps and state whether the sweep outcome agreed.

## Dimension efficiency

Every task is run across `latent_dim ∈ {8, 16, 32, 64, 128}`. `metrics/dimension_efficiency.py`
turns the results into metric-vs-dimension curves and their area under the curve (log2 dimension
axis). The headline claim, if the hypothesis holds, is "hyperbolic reaches error `e` at dimension
`d_h < d_e`", with both numbers and the full curves in the report.

## Statistical protocol

- Three seeds per configuration; tables show mean ± std.
- A difference counts only if it exceeds one std of the seeds at both configurations.
- The Euclidean baseline is run with the same head architecture, optimiser, schedule, latent
  dimensions and seeds as the hyperbolic heads; only the manifold differs.
- The embedding from encoder space onto the manifold is a frozen seeded projection in phase 1, so
  no geometry can "win" by collapsing its embedding. A learnable embedding with an anti-collapse
  regulariser is a phase-3 ablation, reported separately.
- The two heads share every module and hyper-parameter (`tests/models/test_predictor_heads.py`
  asserts identical parameter signatures and that the same seed gives identical initial weights).
  Updates are expressed in geodesic units: a proposed vector of norm `r` moves the state by
  geodesic distance `r` in every geometry (the Poincaré ball's origin metric scale of 2 is divided
  out), so `max_step`, `embed_scale` and `max_radius` mean the same thing for every head, and the
  Poincaré and Lorentz heads at equal curvature are isometric twins (tested).
- Every head applies the same numerical guard: proposed updates are clipped to `max_step` and any
  state beyond `max_radius` (default 4) is retracted along its geodesic to the origin. In flat
  space this is a norm clip. It exists because float32 hyperbolic geometries lose accuracy
  exponentially with distance (below); without it a Lorentz rollout of five maximal steps
  produced NaNs in testing.
- The V-JEPA 2-AC reference predictor (Meta's, frozen) is scored with Meta's own L1 loss on
  layer-normalised tokens: the native-geometry rule applied to a Euclidean token-space model.

## Numerical caveats (measured, see `tests/geometry`)

- The Poincaré ball clips points to `(1 - eps) * radius`, so nothing can lie farther than
  `reliable_radius(c) = (2 / sqrt(-c)) * artanh(1 - eps)` from the origin: 6.2 units at `c = -1`,
  4.4 at `c = -2`, 3.1 at `c = -4` in float32 (`geometry/utils.py::reliable_radius`). Every head
  therefore enforces `min(max_radius, reliable_radius(c))`, in every geometry, so a config value
  the ball cannot represent never turns into a silent per-curvature cap and the Poincaré and
  Lorentz heads at the same curvature keep identical guards.
- The default `max_radius` is 4 because the float32 hyperboloid's self-distance noise is 0.01 at
  4 units, 0.09 at 6 and 0.7 at 8; at 8 the squared-distance loss floor would exceed a real step.

The curved geometries delegate to geoopt (`geoopt.PoincareBall`, `geoopt.Lorentz`); we keep our
own code only for the Lorentz parallel transport (geoopt's divides by the squared distance and is
singular when the two points coincide) and the ball-to-hyperboloid isometries (geoopt's assume the
unit ball, which is wrong for any curvature other than -1). The geoopt curvature parameter is held
in float64 so float64 inputs get float64 accuracy at every curvature.

- Poincaré ball, float32, points at 0.999 of the radius: quantities that do not go through Möbius
  addition (`dist0`, `logmap0`, `lambda_x`) agree with float64 to 1e-4; pairwise `dist` and
  `logmap` between two boundary points lose about 1%; `expmap` projects back to
  `(1 - 4e-3) * radius`, so the float32 exp/log round trip only holds inside that radius
  (`test_poincare.py::test_float32_round_trip_holds_inside_the_clip_margin`). Float64 round-trips
  to 1e-5 even at 0.999 of the radius.
- Lorentz, float32, at the image of 0.999 of the radius (`x_0 ~ 1e3`): `dist0` and pairwise `dist`
  agree with float64 to 1e-5, self-distance noise is bounded by `2e-3 * x_0`, and `logmap` can
  lose up to 20%. Even in float64 the hyperboloid exp/log round trip degrades exponentially with
  distance (below 1e-5 at 4 units from the origin, below 1 at the 0.999-radius image), so
  hyperboloid latents must stay within a few units of the origin; `HyperbolicHead.max_step` and
  `embed_scale` enforce this.
- The two models agree with each other under the isometry to 1e-9 in float64 at every swept
  curvature (`test_lorentz.py::test_isometry_to_poincare_ball`), which is the cross-geometry
  consistency check.
