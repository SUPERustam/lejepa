"""Pingouin HZ diagnostics shared by the point-optimization experiments."""

import numpy as np
import pingouin as pg


def normality_report(points, pairs):
    data = points.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(data).all():
        raise ValueError("Normality diagnostics require finite samples")
    report = []
    for dims in [None, *pairs]:
        subset = data if dims is None else data[:, list(dims)]
        rank = int(np.linalg.matrix_rank(subset - subset.mean(0)))
        hz, pval, normal = pg.multivariate_normality(subset, alpha=0.05)
        report.append({
            "dimensions": "all" if dims is None else [d + 1 for d in dims],
            "hz": float(hz), "pval": float(pval) if np.isfinite(pval) else None,
            "pingouin_normal": bool(normal), "rank": rank,
            "rank_deficient": rank < subset.shape[1],
        })
    return {"test": "Henze-Zirkler", "pingouin_version": pg.__version__,
            "alpha": 0.05, "results": report}


def normality_label(result):
    p = "undefined" if result["pval"] is None else f'{result["pval"]:.3g}'
    status = "singular covariance" if result["rank_deficient"] else (
        "reject" if result["pval"] is not None and result["pval"] <= 0.05 else (
            "not rejected" if result["pval"] is not None else "undefined"
        )
    )
    return f'HZ={result["hz"]:.3g}, p={p}\n{status}'


if __name__ == "__main__":
    import torch

    x = torch.tensor(np.random.default_rng(0).normal(size=(100, 4)))
    report = normality_report(x, [(0, 1)])
    for result, data in zip(report["results"], [x.numpy(), x[:, :2].numpy()]):
        expected = pg.multivariate_normality(data)
        np.testing.assert_allclose([result["hz"], result["pval"]], expected[:2])
        assert not result["rank_deficient"]
    singular = normality_report(x[:, :2].repeat(1, 3), [])
    assert singular["results"][0]["rank_deficient"]
    assert "singular covariance" in normality_label(singular["results"][0])
    print("Normality checks passed: direct Pingouin agreement and singular labeling")
