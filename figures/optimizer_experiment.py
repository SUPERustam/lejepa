"""Paired point-cloud optimizer experiment; see figures/README.md for usage."""

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import torch

from lejepa.multivariate import SlicingUnivariateTest
from lejepa.multivariate.projected_moments import standartization_loss
from lejepa.univariate import EppsPulley
from lejepa.visreg import VISReg
from figures.normality import normality_label, normality_report


SEEDS = (0,)
OBJECTIVES = ("VISReg", "SIGReg", "Projected moments")
RATES = {
    "Adam": (0.001, 0.01, 0.1, 1.0),
    # "Adam": (0.1,),
    "AdamW": (0.001, 0.01, 0.1, 1.0),
    # "AdamW": (0.1,),
    "SGD": (0.1, 1.0, 10.0, 100.0),
}


def initial_points(seed, n=100, dim=1024):
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(n, dim, generator=generator)
    points[:, 1] = points[:, 0] * (2 * (torch.rand(n, generator=generator) > 0.5) - 1)
    points[:, :2] += 1
    return points


# @torch.no_grad()
# def geometry(points, n_directions=1024, seed=20000):
#     """Raw sliced quantile discrepancy, with center/scale/shape decomposition."""
#     directions = torch.randn(
#         points.size(1),
#         n_directions,
#         dtype=points.dtype,
#         generator=torch.Generator().manual_seed(seed),
#     )
#     directions /= directions.norm(dim=0)
#     projected = (points @ directions).sort(dim=0).values
#     u = torch.arange(1, points.size(0) + 1, dtype=points.dtype) / (points.size(0) + 1)
#     target = torch.distributions.Normal(0, 1).icdf(u)[:, None]
#     mean, std = projected.mean(0), projected.std(0, unbiased=False)
#     normalized = (projected - mean) / std.clamp_min(1e-6)
#     return {
#         "score": (projected - target).square().mean().item(),
#         "projected_center": mean.square().mean().item(),
#         "projected_scale": (std - 1).square().mean().item(),
#         "projected_shape": (normalized - target).square().mean().item(),
#         "coordinate_mean_mse": points.mean(0).square().mean().item(),
#         "coordinate_variance_mse": (points.var(0, unbiased=False) - 1)
#         .square()
#         .mean()
#         .item(),
#     }


def make_objective(objective_name, num_slices):
    return {
        "VISReg": lambda: VISReg(num_slices=num_slices),
        "SIGReg": lambda: SlicingUnivariateTest(EppsPulley(), num_slices=num_slices),
        "Projected moments": lambda: standartization_loss,
    }[objective_name]()


@torch.no_grad()
def objective_value(points, objective_name, num_slices=256, seed=10000):
    """Evaluate the native loss on fixed directions without touching training state."""
    objective = make_objective(objective_name, num_slices)
    generator = torch.Generator(device=points.device).manual_seed(seed)
    if objective_name == "SIGReg":
        objective = objective.to(device=points.device, dtype=points.dtype)
        objective.global_step.fill_(seed)
        return objective(points).item()
    if objective_name == "VISReg":
        return objective(points, generator=generator).item()
    return objective(points, num_slices, generator=generator).item()


def run(initial, objective_name, optimizer_name, lr, steps=1520, num_slices=10):
    points = initial.clone().requires_grad_()
    objective = make_objective(objective_name, num_slices)
    optimizer = getattr(torch.optim, optimizer_name)(
        [points],
        lr=lr,
        weight_decay=0.01 if optimizer_name == "AdamW" else 0,
    )
    generator = torch.Generator()
    losses = []
    started = perf_counter()
    failure = None
    for step in range(steps):
        generator.manual_seed(step)
        optimizer.zero_grad(set_to_none=True)
        if objective_name == "SIGReg":
            loss = objective(points)
        elif objective_name == "VISReg":
            loss = objective(points, generator=generator)
        else:
            loss = objective(points, num_slices, generator=generator)
        if not torch.isfinite(loss):
            failure = f"nonfinite loss at step {step}"
            break
        loss.backward()
        if not torch.isfinite(points.grad).all():
            failure = f"nonfinite gradient at step {step}"
            break
        optimizer.step()
        losses.append(loss.item())
        if not torch.isfinite(points).all():
            failure = f"nonfinite points after step {step}"
            break
    metrics = (
        {"objective_value": objective_value(points.detach(), objective_name)}
        if failure is None else None
    )
    if metrics is not None and not all(np.isfinite(v) for v in metrics.values()):
        failure, metrics = "nonfinite selection metrics", None
    return {
        "objective": objective_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "failure": failure,
        "losses": losses,
        "metrics": metrics,
        "points": points.detach(),
        "seconds": perf_counter() - started,
    }


def select_best(candidates):
    successful = [r for r in candidates if r["failure"] is None]
    return (
        min(successful, key=lambda r: r["metrics"]["objective_value"])
        if successful else None
    )


def diagnose(points, pairs):
    return {
        "objective_values": {
            name: objective_value(points, name, 1024, 20000) for name in OBJECTIVES
        },
        "singular_values": torch.linalg.svdvals(
            (points - points.mean(0)).double()
        ).tolist(),
        "normality": normality_report(points, pairs),
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_results(output, runs, originals, diagnostics, pairs, steps):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    window = min(50, steps)
    for axis, objective in zip(axes, OBJECTIVES):
        for color, optimizer in enumerate(RATES):
            group = [
                r
                for r in runs
                if r["objective"] == objective and r["optimizer"] == optimizer
            ]
            if not group:
                continue
            smoothed = np.array(
                [
                    np.convolve(r["losses"], np.ones(window) / window, mode="valid")
                    for r in group
                ]
            )
            mean, std = smoothed.mean(0), smoothed.std(0)
            x = np.arange(window, steps + 1)
            axis.plot(
                x, mean, color=f"C{color}", label=f"{optimizer}, lr={group[0]['lr']:g}"
            )
            axis.fill_between(
                x, np.maximum(mean - std, 0), mean + std, color=f"C{color}", alpha=0.15
            )
        axis.set(
            title=objective,
            xlabel="Step",
            ylabel=f"Objective ({window}-step mean ± seed SD)",
        )
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    fig.savefig(output / "training_curves.pdf")
    plt.close(fig)

    for seed in originals:
        for objective in OBJECTIVES:
            samples = {f"initial/{seed}": originals[seed]}
            titles = ["Initial"]
            for result in runs:
                if result["seed"] == seed and result["objective"] == objective:
                    samples[result["key"]] = result["points"]
                    titles.append(f"{result['optimizer']}, lr={result['lr']:g}")
            fig, axes = plt.subplots(
                len(pairs),
                len(samples),
                squeeze=False,
                figsize=(3 * len(samples), 2.5 * len(pairs)),
                sharex="row",
                sharey="row",
            )
            for col, (key, points) in enumerate(samples.items()):
                report = diagnostics[key]
                for row, (a, b) in enumerate(pairs):
                    axis = axes[row, col]
                    axis.scatter(points[:, a], points[:, b], s=16, alpha=0.6)
                    axis.set(xlabel=f"dim {a + 1}", ylabel=f"dim {b + 1}")
                    axis.text(
                        0.03,
                        0.97,
                        normality_label(report["normality"]["results"][row + 1]),
                        transform=axis.transAxes,
                        va="top",
                        fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
                    )
                axes[0, col].set_title(
                    f"{titles[col]}\n{objective}={report['objective_values'][objective]:.5f}",
                    fontsize=9,
                )
            fig.suptitle(
                f"{objective}; seed {seed}; N=100, D=1024; {steps} steps\n"
                "Full HZ: singular covariance. Pairwise p-values are exploratory.",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            stem = f"{objective.lower().replace(' ', '_')}_seed_{seed}"
            fig.savefig(output / f"{stem}.png", dpi=140)
            fig.savefig(output / f"{stem}.pdf")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/optimizer_comparison_objective_tuned")
    )
    parser.add_argument("--steps", type=int, default=1520)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = {
        "n": 100,
        "dim": 1024,
        "num_slices": 10,
        "steps": args.steps,
        "tuning_seed": 0,
        "evaluation_seeds": list(SEEDS),
        "rates": RATES,
        "evaluation_includes_tuning_seed": True,
        "weight_decay": {"Adam": 0, "AdamW": 0.01, "SGD": 0},
        "SGD_momentum": 0,
        "selection_directions": 256,
        "selection_metric": "native objective on fixed evaluation directions",
        "selection_seed": 10000,
        "evaluation_directions": 1024,
        "evaluation_seed": 20000,
        "dtype": "float32",
        "device": "mps",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pingouin": pg.__version__,
        "python": platform.python_version(),
    }
    save_json(output / "config.json", config)
    originals = {seed: initial_points(seed) for seed in SEEDS}
    tuning, selected, runs = [], [], []
    for objective in OBJECTIVES:
        for optimizer, rates in RATES.items():
            candidates = []
            for lr in rates:
                result = run(originals[0], objective, optimizer, lr, args.steps)
                candidates.append(result)
                tuning.append(result)
                print(
                    f"Tune {objective}/{optimizer}, lr={lr:g}: "
                    f"{result['failure'] or result['metrics']['objective_value']}",
                    flush=True,
                )
                torch.save(tuning, output / "tuning.pt")
            best = select_best(candidates)
            selected.append(
                {
                    "objective": objective,
                    "optimizer": optimizer,
                    "lr": best["lr"] if best else None,
                    "failure": None if best else "all candidates failed",
                }
            )
            if best is None:
                continue
            for seed, points in originals.items():
                # Seed 0 is already trained during tuning; reuse it exactly.
                result = (
                    dict(best)
                    if seed == 0
                    else run(points, objective, optimizer, best["lr"], args.steps)
                )
                result.update(seed=seed, key=f"{objective}/{optimizer}/{seed}")
                runs.append(result)
                torch.save(
                    {"config": config, "initial": originals, "runs": runs},
                    output / "results.pt",
                )
                print(
                    f"Evaluate {result['key']}: {result['failure'] or 'complete'}",
                    flush=True,
                )
    save_json(output / "selected.json", selected)
    save_json(
        output / "tuning.json",
        [{k: v for k, v in r.items() if k not in ("points", "losses")} for r in tuning],
    )

    starts = [0, 2, *torch.linspace(4, 1022, 8, dtype=torch.long).tolist()]
    pairs = [(start, start + 1) for start in starts]
    diagnostics = {}
    for seed in SEEDS:
        diagnostics[f"initial/{seed}"] = diagnose(originals[seed], pairs)
    successful = [r for r in runs if r["failure"] is None]
    for result in successful:
        diagnostics[result["key"]] = diagnose(result["points"], pairs)
        save_json(output / "diagnostics.json", diagnostics)
    rows = []
    for setting in selected:
        group = [
            diagnostics[r["key"]]["objective_values"][setting["objective"]]
            for r in successful
            if r["objective"] == setting["objective"]
            and r["optimizer"] == setting["optimizer"]
        ]
        row = {**setting, "successful_seeds": len(group)}
        row["objective_value_mean"] = float(np.mean(group)) if group else None
        row["objective_value_std"] = float(np.std(group, ddof=1)) if len(group) > 1 else None
        rows.append(row)
    save_json(output / "summary.json", rows)
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(output, successful, originals, diagnostics, pairs, args.steps)
    print(f"Saved experiment to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
