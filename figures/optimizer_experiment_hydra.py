"""Standalone Hydra optimizer experiment; defaults live in config/optimizer_experiment.yaml.

Run from the repository root with the experiment dependencies and hydra-core::

    python -m figures.optimizer_experiment_hydra
    python -m figures.optimizer_experiment_hydra 'seeds=[1,2]' 'optimizers=[Adam]'
    python -m figures.optimizer_experiment_hydra hydra.job.env_set.OMP_NUM_THREADS=2

Hydra records the config and overrides in each output directory's .hydra/ folder.
Use hydra.run.dir=... to choose an output directory.
"""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

from figures.normality import normality_label, normality_report
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.multivariate.projected_moments import standartization_loss
from lejepa.univariate import EppsPulley
from lejepa.visreg import VISReg


OmegaConf.register_new_resolver(
    "experiment_version",
    {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pingouin": pg.__version__,
        "python": platform.python_version(),
    }.__getitem__,
    replace=True,
)
OmegaConf.register_new_resolver(
    "contains", lambda values, value: value in values, replace=True
)


def initial_points(seed, cfg):
    n, dim = cfg.n, cfg.dim
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(n, dim, generator=generator)
    points[:, 1] = points[:, 0] * (2 * (torch.rand(n, generator=generator) > 0.5) - 1)
    points[:, :2] += 1
    return points


def make_objective(objective_name, num_slices):
    return {
        "VISReg": lambda: VISReg(num_slices=num_slices),
        "SIGReg": lambda: SlicingUnivariateTest(EppsPulley(), num_slices=num_slices),
        "Projected moments": lambda: standartization_loss,
    }[objective_name]()


@torch.no_grad()
def objective_value(points, objective_name, num_slices, seed):
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


def run(initial, objective_name, optimizer_name, lr, cfg):
    steps, num_slices = cfg.steps, cfg.num_slices
    points = initial.clone().requires_grad_()
    objective = make_objective(objective_name, num_slices)
    options = {"weight_decay": cfg.weight_decay[optimizer_name]}
    if optimizer_name == "SGD":
        options["momentum"] = cfg.SGD_momentum
    optimizer = getattr(torch.optim, optimizer_name)(
        [points],
        lr=lr,
        **options,
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
        {
            "objective_value": objective_value(
                points.detach(),
                objective_name,
                cfg.selection_directions,
                cfg.selection_seed,
            )
        }
        if failure is None
        else None
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
    return min(
        (r for r in candidates if r["failure"] is None),
        key=lambda r: r["metrics"]["objective_value"],
        default=None,
    )


def diagnose(points, pairs, cfg):
    return {
        "objective_values": {
            name: objective_value(
                points, name, cfg.evaluation_directions, cfg.evaluation_seed
            )
            for name in cfg.objectives
        },
        "singular_values": torch.linalg.svdvals(
            (points - points.mean(0)).double()
        ).tolist(),
        "normality": normality_report(points, pairs),
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_results(output, runs, originals, diagnostics, pairs, cfg):
    steps = cfg.steps
    fig, axes = plt.subplots(
        1, len(cfg.objectives), figsize=(5 * len(cfg.objectives), 4), squeeze=False
    )
    window = min(50, steps)
    for axis, objective in zip(axes[0], cfg.objectives):
        for color, optimizer in enumerate(cfg.optimizers):
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
        for objective in cfg.objectives:
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
                f"{objective}; seed {seed}; N={cfg.n}, D={cfg.dim}; {steps} steps\n"
                "Pairwise p-values are exploratory; singular covariance is marked in panels.",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            stem = f"{objective.lower().replace(' ', '_')}_seed_{seed}"
            fig.savefig(output / f"{stem}.png", dpi=140)
            fig.savefig(output / f"{stem}.pdf")
            plt.close(fig)


@hydra.main(
    version_base="1.3", config_path="config", config_name="optimizer_experiment"
)
def main(cfg: DictConfig):
    output = Path(HydraConfig.get().runtime.output_dir)
    environment = dict(HydraConfig.get().job.env_set)
    if any(
        (output / name).exists() for name in ("config.json", "results.pt", "tuning.pt")
    ):
        raise FileExistsError(f"Output already contains experiment results: {output}")
    with (
        threadpool_limits(
            limits=int(environment["OMP_NUM_THREADS"]), user_api="openmp"
        ),
        threadpool_limits(
            limits=int(environment["OPENBLAS_NUM_THREADS"]), user_api="blas"
        ),
    ):
        torch.set_num_threads(cfg.torch_num_threads)
        run_experiment(cfg, output)


def run_experiment(cfg, output):
    start = int(np.random.default_rng(cfg.evaluation_seed).integers(4, cfg.dim - 1))
    pairs = [(0, 1), (3, 2), (start + 1, start)]
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    save_json(output / "config.json", config)
    originals = {seed: initial_points(seed, cfg) for seed in cfg.seeds}
    tuning_points = initial_points(cfg.tuning_seed, cfg)
    tuning, selected, runs = [], [], []
    for objective in cfg.objectives:
        for optimizer in cfg.optimizers:
            learning_rates = cfg.rates[optimizer]
            candidates = []
            for lr in learning_rates:
                result = run(tuning_points, objective, optimizer, lr, cfg)
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
                # Reuse the tuning run when its seed is also an evaluation seed.
                result = (
                    dict(best)
                    if seed == cfg.tuning_seed
                    else run(points, objective, optimizer, best["lr"], cfg)
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

    diagnostics = {}
    for seed in cfg.seeds:
        diagnostics[f"initial/{seed}"] = diagnose(originals[seed], pairs, cfg)
    successful = [r for r in runs if r["failure"] is None]
    for result in successful:
        diagnostics[result["key"]] = diagnose(result["points"], pairs, cfg)
        save_json(output / "diagnostics.json", diagnostics)
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
        row["objective_value_std"] = (
            float(np.std(group, ddof=1)) if len(group) > 1 else None
        )
        rows.append(row)
    save_json(output / "summary.json", rows)
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(output, successful, originals, diagnostics, pairs, cfg)
    print(f"Saved experiment to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
