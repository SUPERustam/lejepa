"""Figure 6 point optimization with VISReg shape-only and full regularization.

From the repository root, in menv:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONPATH=. python figures/visreg_experiment.py

Outputs: output/visreg_figure6/{visreg_figure6.pdf,visreg_figure6.png,results.pt}.
Loss definition: https://arxiv.org/abs/2606.02572, Algorithm 1.
Full means shape + scale + center, with unit weights; no paired views exist here.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch

from lejepa.visreg import VISReg
from normality import normality_report, normality_label


def main():
    n, dim, num_slices, steps, lr = 100, 1024, 10, 1520, 0.1
    torch.manual_seed(0)
    initial = torch.randn(n, dim)
    initial[:, 1] = initial[:, 0] * ((torch.rand(n) > 0.5).float() * 2 - 1)
    initial[:, :2] += 1
    results = {
        "config": {"n": n, "dim": dim, "num_slices": num_slices,
                   "steps": steps, "lr": lr, "seed": 0, "eps": 1e-6},
        "initial": initial, "samples": {}, "losses": {},
    }
    for name, shape_only in (("L_shape only", True), ("Full VISReg", False)):
        points = initial.clone().requires_grad_()
        objective = VISReg(num_slices=num_slices, shape_only=shape_only)
        optimizer = torch.optim.Adam([points], lr=lr)
        generator = torch.Generator()
        losses = []
        for step in range(steps):
            # Match the previous slicing implementation's seed-per-step directions.
            generator.manual_seed(step)
            optimizer.zero_grad()
            loss = objective(points, generator=generator)
            assert torch.isfinite(loss), f"Nonfinite loss: {name}, step {step}"
            loss.backward()
            assert torch.isfinite(points.grad).all(), f"Nonfinite gradients: {name}, step {step}"
            optimizer.step()
            losses.append(loss.item())
        assert torch.isfinite(points).all()
        results["samples"][name] = points.detach().clone()
        results["losses"][name] = losses
        print(f"{name}: mean loss first/last 100 steps "
              f"{sum(losses[:100]) / 100:.6f} -> {sum(losses[-100:]) / 100:.6f}")

    pair_starts = [0, 2, *torch.linspace(4, dim - 2, 8, dtype=torch.long).tolist()]
    results["plot_pairs"] = [(start, start + 1) for start in pair_starts]
    results["normality"] = {}
    fig, axes = plt.subplots(10, 3, figsize=(10, 25), sharex="row", sharey="row")
    for col, (name, points) in enumerate(
        [("Original data", initial), *results["samples"].items()]
    ):
        report = normality_report(points, results["plot_pairs"])
        results["normality"][name] = report
        print(name, "full-dimensional:", normality_label(report["results"][0]))
        for row, start in enumerate(pair_starts):
            axes[row, col].scatter(
                points[:, start], points[:, start + 1],
                c="grey" if col == 0 else "green", edgecolor="k", alpha=0.5,
            )
            axes[row, col].set_xlabel(f"dim {start + 1}")
            axes[row, 0].set_ylabel(f"dim {start + 2}")
            axes[row, col].text(
                0.03, 0.97, normality_label(report["results"][row + 1]),
                transform=axes[row, col].transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
        axes[0, col].set_title(name + "\nFull: " + normality_label(report["results"][0]), fontsize=10)
    fig.suptitle("N=100, D=1024, 10 slices/step; Adam lr=0.1, 1520 steps\n"
                 "Pingouin HZ, raw p-values, alpha=0.05; full covariance is singular.\n"
                 "Pairwise non-rejection does not establish joint normality or N(0, I).", y=0.998, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = Path(__file__).resolve().parents[1] / "output" / "visreg_figure6"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "visreg_figure6.pdf")
    fig.savefig(output / "visreg_figure6.png", dpi=200)
    plt.close(fig)
    torch.save(results, output / "results.pt")
    (output / "normality.json").write_text(json.dumps(results["normality"], indent=2, allow_nan=False))
    print(f"Saved plots, samples, and losses to {output}")


if __name__ == "__main__":
    main()
