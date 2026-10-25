import json
import matplotlib.pyplot as plt
from tqdm.rich import trange
import torch
import lejepa as ds
from normality import normality_report, normality_label


def get_X(N, dim):
    torch.manual_seed(0)
    X = torch.randn(N, dim, device=device)
    X[:, 1] = X[:, 0] * ((torch.rand(X.size(0), device=device) > 0.5).float() * 2 - 1)
    X[:, :2] = X[:, :2] + 1
    return X


N = 100
device = "cpu"
tests = [
    ds.univariate.VCReg(),
    ds.univariate.ExtendedJarqueBera(),
    ds.univariate.CramerVonMises(),
    ds.univariate.Watson(),
    ds.univariate.AndersonDarling(),
    ds.univariate.EppsPulley(),
]
for num_slices in [10]:
    for i, dim in enumerate([1024]):
        X = get_X(N, dim)
        results = {"initial": X.clone(), "samples": {}, "losses": {}}
        for j, test in enumerate(tests):
            print(test)
            torch.manual_seed(0)
            if isinstance(test, ds.univariate.UnivariateTest):
                g_loss = ds.multivariate.SlicingUnivariateTest(
                    univariate_test=test, num_slices=num_slices
                )
            else:
                g_loss = test
            Xp = X.clone().detach().requires_grad_(True)
            optim = torch.optim.Adam([Xp], lr=0.1)
            losses = []
            for step in trange(1520):
                optim.zero_grad()
                loss = g_loss(Xp)
                assert torch.isfinite(loss), f"Nonfinite loss for {test} at step {step}"
                losses.append(loss.item())
                loss.backward()
                assert torch.isfinite(Xp.grad).all(), f"Nonfinite gradients for {test}"
                optim.step()
            results["samples"][type(test).__name__] = Xp.detach().clone()
            results["losses"][type(test).__name__] = losses
        pair_starts = [0, 2, *torch.linspace(4, dim - 2, 8, dtype=torch.long).tolist()]
        results["plot_pairs"] = [(start, start + 1) for start in pair_starts]
        results["normality"] = {}
        fig, axs = plt.subplots(
            10, len(tests) + 1, sharex="row", sharey="row",
            figsize=(2.5 * (len(tests) + 1), 25),
        )
        for col, (name, points) in enumerate(
            [("Original data", X), *results["samples"].items()]
        ):
            report = normality_report(points, results["plot_pairs"])
            results["normality"][name] = report
            print(name, "full-dimensional:", normality_label(report["results"][0]))
            for row, start in enumerate(pair_starts):
                axs[row, col].scatter(
                    points[:, start].cpu(), points[:, start + 1].cpu(),
                    c="grey" if col == 0 else "green", linewidth=1,
                    edgecolor="k", alpha=0.5, zorder=1000,
                )
                axs[row, col].set_xlabel(f"dim {start + 1}")
                axs[row, 0].set_ylabel(f"dim {start + 2}")
                axs[row, col].text(
                    0.03, 0.97, normality_label(report["results"][row + 1]),
                    transform=axs[row, col].transAxes, va="top", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
                )
            axs[0, col].set_title(
                name + "\nFull: " + normality_label(report["results"][0]), fontsize=9
            )
        fig.suptitle("Pingouin HZ, raw p-values, alpha=0.05; full 1024D covariance is singular.\n"
                     "Pairwise non-rejection does not establish joint normality or N(0, I).", y=0.998)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        plt.savefig(f"2d_slicing_dim_{dim}_N_{N}_slices_{num_slices}.pdf")
        plt.savefig(f"2d_slicing_dim_{dim}_N_{N}_slices_{num_slices}.png", dpi=200)
        torch.save(results, f"2d_slicing_dim_{dim}_N_{N}_slices_{num_slices}.pt")
        with open(f"2d_slicing_dim_{dim}_N_{N}_slices_{num_slices}_normality.json", "w") as f:
            json.dump(results["normality"], f, indent=2, allow_nan=False)
        plt.close()
