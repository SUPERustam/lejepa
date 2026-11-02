# Bound constant

To generate figure 1, simply run `python bound_constant.py`. This will generate a `bound_constant.pdf` file.

# Sobolev spheres

To generate figure 2, simply run `python 3d_sobolev.py`. This will generate `3d_sphere_0.png`, `3d_sphere_1.png` and `3d_sphere_2.png`.

# Quantile density and variance

To generate figures 1 and 2, simply run `python quantile_variance.py`. To vary `N` (the sample size), simply edit that variable at the top of the file. The execution will produce files `quantile_pdf_*.pdf` and `quantile_variance_*.pdf` over the list of `N` specified by the user.


# Nonparametric example (paper Figure 6)

From the repository root, run:

```sh
mamba activate menv
python -m pip install pingouin==0.5.5
mkdir -p output/figure6
cd output/figure6
MPLBACKEND=Agg PYTHONPATH=../.. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ../../figures/nonparametric_example.py
```

The script uses 100 samples in 1024 dimensions, 10 newly sampled unit directions per step, and Adam with learning rate 0.1 for 1520 steps per statistic. Seed 0 initializes the points; the first two coordinates form an X shifted by +1. Each statistic starts from the same points and projection sequence. All coordinates are optimized.

Outputs are `2d_slicing_dim_1024_N_100_slices_10.pdf`, a PNG preview, and a `.pt` file containing the initial points, optimized points, and training losses. Both the LeJEPA and VISReg experiment plots show 10 coordinate pairs: (1, 2), (3, 4), and eight pairs spaced across the remaining dimensions through (1023, 1024). Axis labels identify the dimensions, and the saved results include zero-based `plot_pairs`. The script requires Matplotlib, tqdm, and Rich in addition to the package dependencies. Edit the dimension and slice loops to reproduce the additional settings in Figure 16.

Both experiments run `pingouin.multivariate_normality` (Henze-Zirkler, alpha=0.05) on the initial points and each optimized result, for the full matrix and each plotted pair. Plots annotate HZ, raw p-values, and rejection decisions. The `.pt` files include `normality` reports; readable JSON reports are saved alongside the figures. Pingouin 0.5.5 is used to preserve the existing menv NumPy version.

With N=100 and D=1024, the full covariance is necessarily singular: Pingouin returns its fallback HZ=4N rather than a useful joint normality assessment. These outputs are explicitly marked singular; nonfinite p-values are stored as JSON null. Pairwise p-values are unadjusted exploratory diagnostics of optimized, dependent samples; non-rejection does not establish full joint normality, zero mean, unit variance, or isotropy. Run `python figures/normality.py` from the root for the diagnostic self-checks.

# Optimizer comparison

Run from the repository root in `menv`:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/lejepa-matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m figures.optimizer_experiment
```

The experiment compares full VISReg, SIGReg (sliced Epps–Pulley), and
`standartization_loss()` extracted from `figures/notebooks/random_loss.ipynb`.
It optimizes 100 points in 1024 dimensions, using 10 new shared directions per
step for 1520 steps. Pure SGD uses zero momentum and zero decay; Adam uses zero
decay and AdamW uses decay 0.01. Current `RATES` fixes Adam and AdamW at 0.1;
SGD tunes 0.1, 1, 10, 100. All other optimizer settings use PyTorch defaults.

Each loss/optimizer combination selects its learning rate using **that loss's
own objective**, evaluated on 256 fixed directions (seed 10000), not `geometry()`
and not the last noisy training step. Final diagnostics evaluate each native
objective on 1024 different directions (seed 20000). Lower is better within the
same objective; values from different objectives have different scales and must
not be ranked against one another.

Current `SEEDS=(0,)` reuses seed 0's winning tuning run, so the defaults require
18 training runs and provide no independent-seed validation or standard
deviation. Changing `SEEDS` adds evaluation seeds; seed 0 remains the tuning
seed. Evaluation uses held-out directions, not held-out observations.
Plots compare the initial cloud with the three optimizers; no Gaussian
reference cloud is generated.

Outputs in `output/optimizer_comparison_objective_tuned/` include all tuning clouds and loss
histories (`tuning.pt`), selected learning rates, selected-seed clouds and losses
(`results.pt`), native-objective/spectrum/HZ diagnostics, CSV/JSON summaries, training
curves, and coordinate plots for every objective and seed. The output
curves show a trailing 50-step mean, averaged across seeds with a ±1 population
standard-deviation band; raw losses remain in the `.pt` files. The output
directory must not already exist, preventing accidental overwrites. Use
`--output PATH` for another run or `--steps 3 --output PATH` for an end-to-end
smoke run. Results from shortened runs are not the full experiment.

Full HZ is still singular at N=100, D=1024. Projected moments constrain only
mean/variance. AdamW versus Adam includes the effect of decay, which directly
shrinks the point cloud. SGD's four-rate search does not establish a global best;
an endpoint winner signals that a wider grid may be useful. No momentum or
weight-decay sweep is included.

Existing results in `output/optimizer_comparison/` use the previous geometry-based
selection and describe seeds 1, 2, 3. Other existing output directories are also
historical; their saved `config.json` records their settings. Use a new `--output` directory
when running the simplified script; old results are not rewritten.
