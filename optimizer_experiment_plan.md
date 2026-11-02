# Optimizer comparison for VISReg, SIGReg, and projected moments

## Experiment design

Add `figures/optimizer_experiment.py`, preserving the original point-optimization setup: **N=100, D=1024, 10 projections per step, 1520 steps**. This studies optimization of the point cloud itself; there is no encoder or downstream task.

Compare three standalone objectives:

| Objective | Purpose |
|---|---|
| Full VISReg | Existing shape, scale, and center regularization |
| SIGReg | Existing sliced `EppsPulley()` implementation |
| `standartization_loss()` | Notebook’s projected mean and variance penalty |

Use identical initial clouds and projection sequences across methods. Keep the shifted X-shaped first two coordinates and the existing ten coordinate-pair plots. This makes differences attributable to the objective and optimizer.

## Optimizers and selection

Use the selected lightweight budget: **four configurations per objective–optimizer combination**, tuned on seed 0, then rerun each winner on seeds 0–2 (seed 0 reuses the tuning winner).

| Optimizer | Learning rates | Other settings |
|---|---|---|
| Adam | 0.001, 0.01, 0.1, 1.0 | decay 0 |
| AdamW | 0.001, 0.01, 0.1, 1.0 | decay 0.01 |
| Pure SGD | 0.1, 1, 10, 100 | momentum 0, decay 0 |

Use constant learning rates, fresh optimizer state per run, and no scheduler or clipping. Mark nonfinite runs failed and exclude them from selection; report failure if all four candidates fail.

Select the lowest final **sliced squared distance to standard-normal quantiles**, computed on raw points using 256 fixed evaluation directions unseen during training. Do not standardize before scoring: center and scale are part of the target.

Evaluate selected runs on 1024 new fixed directions and report mean and standard deviation across the three seeds, including tuning seed 0. Describe winners as **best among the tested settings**, not globally best.

**Why:** a common score permits comparison across differently scaled objectives. SGD’s wider learning-rate grid accounts for its dependence on gradient magnitude. AdamW’s nonzero decay preserves a meaningful difference from Adam, but confounds optimizer and decay effects in this small-budget experiment.

## Notebook loss and interfaces

Move the notebook to `figures/notebooks/random_loss.ipynb` and extract its function into `lejepa/multivariate/projected_moments.py`.

- Preserve `standartization_loss(x, n_projections=16)` and its exact formula; add optional `generator=None` for reproducible directions.
- Validate floating-point `(N, D)` input, N ≥ 2, and positive projection count.
- Sample unit directions without gradients; retain gradients through the points.
- Use 10 projections in the comparison for parity with the other objectives.
- Import the function explicitly; leave existing public exports and loss APIs unchanged.

**Why:** the notebook remains an exploratory demonstration, while the experiment imports reusable code. This loss matches moments, so low loss alone does not establish Gaussian shape.

## Diagnostics and outputs

Reuse the existing full-dimensional HZ call:

```python
report = normality_report(points, pairs=[])
```

It already passes the entire matrix to Pingouin. At N=100 and D=1024, covariance is necessarily singular; label the result unavailable for joint-normality interpretation. Pingouin 0.5.5 returns its `4N` fallback in this case. [Implementation](https://raw.githubusercontent.com/raphaelvallat/pingouin/v0.5.5/src/pingouin/multivariate.py)

Save under `output/optimizer_comparison/`:

- Configuration, versions, failures, selected settings, loss histories, and final point clouds.
- Common geometry scores, coordinate mean/variance errors, centered-data singular values, and existing HZ reports.
- Training curves and coordinate-pair figures grouped by objective, with optimizer columns.
- An aggregate results table across seeds 0, 1, 2 (including tuning seed 0).

Treat pairwise HZ as exploratory; neither pairwise non-rejection nor moment matching establishes joint N(0, I).

## Verification and limits

- Check the extracted loss against the notebook formula using identical directions, including gradients and constant-coordinate inputs.
- Smoke-test all nine objective–optimizer combinations for finite updates and reproducibility.
- Verify scoring leaves training random sequences unchanged and correctly excludes failed candidates.
- Confirm full-dimensional HZ remains explicitly labeled singular and outputs serialize successfully.

Implementation and runs are recorded under `output/optimizer_comparison/`. Pure SGD replaces the originally proposed momentum variant, following the user’s clarification. The chosen budget skips decay and momentum sweeps; full 1024D HZ remains uninformative at N=100.

The simplified runner uses seeds 0, 1, 2; existing saved full-run results use
the previous seeds 1, 2, 3. Only seeds 1 and 2 are independent of tuning.

The simplified runner omits the Gaussian reference cloud, as requested.

## Tuning criterion update (2026-11-02)

The current runner supersedes the common-score selection above: select learning
rates by each loss's own objective on 256 fixed directions (seed 10000), then
report that objective on 1024 different directions (seed 20000). Keep the user's
commented-out geometry function and learning-rate alternatives intact. Current
settings are Adam/AdamW at 0.1, SGD rates 0.1/1/10/100, and SEEDS=(0,). Seed 0
reuses its tuning result, so this is not independent-seed validation. New outputs
default to output/optimizer_comparison_objective_tuned; previous runs are retained.
