import pytest
import torch

from figures.optimizer_experiment import OBJECTIVES, RATES, objective_value, initial_points, run, select_best
from lejepa.multivariate.projected_moments import standartization_loss


def test_notebook_formula_and_gradient():
    x = torch.randn(31, 7, dtype=torch.float64, requires_grad=True)
    directions = torch.randn(7, 10, dtype=x.dtype, generator=torch.Generator().manual_seed(11))
    directions /= directions.norm(dim=0)
    projected = x @ directions
    expected = (projected.mean(0).pow(2) + (projected.var(0, unbiased=False) - 1).pow(2)).mean()
    actual = standartization_loss(x, 10, torch.Generator().manual_seed(11))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])
    constant = torch.ones(10, 3, requires_grad=True)
    loss = standartization_loss(constant)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(constant.grad).all()


@pytest.mark.parametrize("points,count", [(torch.ones(3), 2), (torch.ones(1, 3), 2),
                                          (torch.ones(3, 0), 2), (torch.ones(3, 2).long(), 2),
                                          (torch.ones(3, 2), 0), (torch.ones(3, 2), 1.5)])
def test_loss_validation(points, count):
    with pytest.raises(ValueError):
        standartization_loss(points, count)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("optimizer", RATES)
def test_runs_are_finite_reproducible_and_use_all_coordinates(objective, optimizer):
    initial = initial_points(3, 20, 8)
    first = run(initial, objective, optimizer, 0.1, steps=3)
    second = run(initial, objective, optimizer, 0.1, steps=3)
    assert first["failure"] is None
    assert first["losses"] == second["losses"]
    torch.testing.assert_close(first["points"], second["points"], rtol=0, atol=0)
    assert ((initial - first["points"]).abs().sum(0) > 0).all()


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_scoring_uses_native_objective_and_does_not_touch_rng(objective):
    points = torch.randn(100, 16, generator=torch.Generator().manual_seed(8))
    state = torch.random.get_rng_state().clone()
    base = objective_value(points, objective)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    generator = torch.Generator().manual_seed(10000)
    if objective == "VISReg":
        from lejepa.visreg import VISReg
        expected = VISReg(256)(points, generator=generator)
    elif objective == "SIGReg":
        from lejepa.univariate import EppsPulley
        directions = torch.randn(16, 256, generator=generator)
        directions /= directions.norm(dim=0)
        expected = EppsPulley()(points @ directions).mean()
    else:
        expected = standartization_loss(points, 256, generator=generator)
    assert base == pytest.approx(expected.item())
    assert objective_value(points, objective) == base
    assert objective_value(points + 5, objective) > base


def test_failed_run_is_excluded():
    failed = run(initial_points(0, 20, 8), "VISReg", "SGD", float("nan"), steps=3)
    assert failed["failure"] and failed["metrics"] is None
    successful = {"failure": None, "metrics": {"objective_value": 1.0, "score": 10.0}}
    assert select_best([failed, successful]) is successful
    assert select_best([failed]) is None
    worse = {"failure": None, "metrics": {"objective_value": 2.0, "score": 0.0}}
    assert select_best([worse, successful]) is successful


def test_main_uses_seeds_zero_one_two_and_reuses_tuning(monkeypatch, tmp_path):
    import json
    from figures import optimizer_experiment as experiment

    calls = []

    def fake_run(initial, objective, optimizer, lr, steps):
        calls.append((objective, optimizer, lr))
        return {"objective": objective, "optimizer": optimizer, "lr": lr,
                "failure": None, "points": initial.clone(), "losses": [lr] * steps,
                "metrics": {"objective_value": lr}}

    output = tmp_path / "experiment"
    monkeypatch.setattr("sys.argv", ["experiment", "--output", str(output), "--steps", "1"])
    monkeypatch.setattr(experiment, "run", fake_run)
    monkeypatch.setattr(experiment, "SEEDS", (0, 1, 2))
    monkeypatch.setattr(experiment, "RATES", {name: (0.001, 0.01, 0.1, 1.0) for name in RATES})
    monkeypatch.setattr(experiment, "diagnose", lambda points, pairs: {"objective_values": {name: 0.0 for name in OBJECTIVES}})
    monkeypatch.setattr(experiment, "plot_results", lambda *args: None)
    experiment.main()
    results = torch.load(output / "results.pt", weights_only=False)
    assert len(calls) == 54  # 36 tuning runs + 18 runs on seeds 1 and 2.
    assert {r["seed"] for r in results["runs"]} == {0, 1, 2}
    assert len(results["runs"]) == 27
    config = json.loads((output / "config.json").read_text())
    assert config["evaluation_seeds"] == [0, 1, 2]
    assert config["evaluation_includes_tuning_seed"]
