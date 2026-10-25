"""VISReg regularization, Algorithm 1 of https://arxiv.org/abs/2606.02572.

Run this file directly for numerical and gradient self-checks.
"""

import torch
from torch import nn


class VISReg(nn.Module):
    """Shape-only or full (shape + scale + center) regularization of (N, D).

    Uses population standard deviations, detached normalization, Gaussian
    quantiles i/(N+1), and mean reductions as in Algorithm 1. Equation 2's
    epsilon guards constant coordinates. No paired-view invariance is included.
    """

    def __init__(self, num_slices=10, shape_only=False, eps=1e-6):
        super().__init__()
        if num_slices < 1 or eps <= 0:
            raise ValueError("num_slices and eps must be positive")
        self.num_slices = num_slices
        self.shape_only = shape_only
        self.eps = eps

    def forward(self, z, generator=None):
        if z.ndim != 2 or z.size(0) < 2 or z.size(1) < 1 or not z.is_floating_point():
            raise ValueError("Expected a floating-point (N, D) tensor with N >= 2, D >= 1")
        mean = z.mean(dim=0)
        centered = z - mean
        std = centered.std(dim=0, unbiased=False)
        normalized = centered / (std.detach() + self.eps)
        directions = torch.randn(
            z.size(1), self.num_slices, device=z.device, dtype=z.dtype,
            generator=generator,
        )
        directions = directions / directions.norm(dim=0)
        projected = (normalized @ directions).sort(dim=0).values
        u = torch.arange(1, z.size(0) + 1, device=z.device, dtype=z.dtype) / (z.size(0) + 1)
        target = torch.distributions.Normal(0, 1).icdf(u)[:, None]
        shape = (projected - target).square().mean()
        if self.shape_only:
            return shape
        return shape + (1 - std).square().mean() + mean.square().mean()


if __name__ == "__main__":
    torch.manual_seed(7)
    z = (torch.randn(31, 5, dtype=torch.float64) * 2 + 3).requires_grad_()
    generator = torch.Generator().manual_seed(11)
    directions = torch.randn(5, 7, dtype=z.dtype, generator=generator)
    directions /= directions.norm(dim=0)
    centered = z - z.mean(0)
    std = centered.std(0, unbiased=False)
    target = torch.distributions.Normal(0, 1).icdf(torch.arange(1, 32, dtype=z.dtype) / 32)
    expected_shape = torch.stack([
        (((centered / (std.detach() + 1e-6)) @ direction).sort().values - target).square().mean()
        for direction in directions.T
    ]).mean()
    for shape_only in (True, False):
        loss_fn = VISReg(7, shape_only=shape_only)
        expected = expected_shape
        if not shape_only:
            expected = expected + (1 - std).square().mean() + z.mean(0).square().mean()
        actual = loss_fn(z, torch.Generator().manual_seed(11))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, z, retain_graph=True)[0],
            torch.autograd.grad(expected, z, retain_graph=True)[0],
        )
        constant = torch.ones(10, 3, requires_grad=True)
        loss = loss_fn(constant)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(constant.grad).all()
    shape_fn = VISReg(7, shape_only=True)
    torch.testing.assert_close(
        shape_fn(z, torch.Generator().manual_seed(11)),
        shape_fn(z + 10, torch.Generator().manual_seed(11)),
    )
    print("VISReg checks passed: values, detached gradients, translation, constant coordinates")
