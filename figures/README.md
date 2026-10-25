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
