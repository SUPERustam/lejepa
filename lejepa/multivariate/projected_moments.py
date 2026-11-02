"""Projected mean/variance loss from Random loss.ipynb."""

import torch


def standartization_loss(x, n_projections=16, generator=None):
    """Match population mean 0 and variance 1 on random unit directions.

    The notebook spelling is preserved. Directions use (D, K) sampling to
    match VISReg and SlicingUnivariateTest with the same generator seed.
    This constrains moments, not Gaussian shape.
    """
    if x.ndim != 2 or x.size(0) < 2 or x.size(1) < 1 or not x.is_floating_point():
        raise ValueError("Expected floating-point (N, D) with N >= 2, D >= 1")
    if not isinstance(n_projections, int) or isinstance(n_projections, bool) or n_projections < 1:
        raise ValueError("n_projections must be a positive integer")
    directions = torch.randn(
        x.size(1), n_projections, dtype=x.dtype, device=x.device, generator=generator,
    )
    directions = directions / directions.norm(dim=0)
    projected = x @ directions
    return (projected.mean(0).square() + (projected.var(0, unbiased=False) - 1).square()).mean()
