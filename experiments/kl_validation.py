"""Validation of the KL implementation itself.

The closed form for the divergence between two Gaussians is derived in the
supplementary material. A derivation is not evidence that the code computes
it, and the closed form is doing most of the work in this project, so it is
checked here against the definition it claims to evaluate,

    KL(P || Q) = E_P[ log p(X) - log q(X) ],

by three independent routes: the textbook scalar formula, numerical
integration of the definition on a grid, and Monte Carlo estimation of the
expectation. The two approximate estimators used elsewhere, grid integration
and the k-nearest-neighbour estimator, are checked against the closed form on
a case where all three apply.

Nothing here depends on any other part of the project. If this file passes,
the instrument is sound; whether the framework's claims hold is a separate
question answered elsewhere.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import multivariate_normal as mvn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kl import gaussian_kl, grid_kl, knn_kl          # noqa: E402
from core.reporting import save                            # noqa: E402

SEED = 20261019
GRID_POINTS = 1400
MC_DRAWS = 5_000_000
DIM_DRAWS = 2_000_000


def check_scalar():
    """One dimension, against the formula as it is usually written."""
    m0, s0, m1, s1 = 0.3, 1.4, -0.8, 0.9
    textbook = np.log(s1 / s0) + (s0 ** 2 + (m0 - m1) ** 2) / (2 * s1 ** 2) - 0.5
    ours = gaussian_kl([m0], [[s0 ** 2]], [m1], [[s1 ** 2]])
    return {"textbook": float(textbook), "closed_form": ours,
            "error": float(abs(textbook - ours))}


def check_grid():
    """Two dimensions, by integrating the definition over a fine grid."""
    m0, S0 = np.array([0.4, -0.2]), np.array([[1.3, 0.5], [0.5, 0.8]])
    m1, S1 = np.array([-0.3, 0.6]), np.array([[0.9, -0.2], [-0.2, 1.6]])
    g = np.linspace(-9.0, 9.0, GRID_POINTS)
    cell = (g[1] - g[0]) ** 2
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    p, q = mvn(m0, S0).pdf(pts), mvn(m1, S1).pdf(pts)
    # grid_kl is the project's own integrator, so this checks it too.
    integrated = grid_kl(p, q, cell)
    closed = gaussian_kl(m0, S0, m1, S1)
    return {"integrated": integrated, "closed_form": closed,
            "error": float(abs(integrated - closed)), "points": GRID_POINTS}


def check_monte_carlo(rng):
    """The definition as an expectation, estimated directly."""
    m0, S0 = np.array([0.4, -0.2]), np.array([[1.3, 0.5], [0.5, 0.8]])
    m1, S1 = np.array([-0.3, 0.6]), np.array([[0.9, -0.2], [-0.2, 1.6]])
    x = rng.multivariate_normal(m0, S0, size=MC_DRAWS)
    d = mvn(m0, S0).logpdf(x) - mvn(m1, S1).logpdf(x)
    mean, half = float(np.mean(d)), 1.96 * float(np.std(d, ddof=1)) / np.sqrt(MC_DRAWS)
    closed = gaussian_kl(m0, S0, m1, S1)
    return {"estimate": mean, "half_width": half, "closed_form": closed,
            "inside_interval": bool(abs(mean - closed) <= half),
            "gap_over_interval": float(abs(mean - closed) / half), "draws": MC_DRAWS}


def check_dimensions(rng):
    """The same comparison in three, five and eight dimensions."""
    rows, worst = [], 0.0
    for k in (3, 5, 8):
        A, B = rng.normal(size=(k, k)), rng.normal(size=(k, k))
        S0, S1 = A @ A.T + k * np.eye(k), B @ B.T + k * np.eye(k)
        m0, m1 = rng.normal(size=k), rng.normal(size=k)
        x = rng.multivariate_normal(m0, S0, size=DIM_DRAWS)
        d = mvn(m0, S0).logpdf(x) - mvn(m1, S1).logpdf(x)
        mean = float(np.mean(d))
        half = 1.96 * float(np.std(d, ddof=1)) / np.sqrt(DIM_DRAWS)
        closed = gaussian_kl(m0, S0, m1, S1)
        ratio = abs(mean - closed) / half
        worst = max(worst, ratio)
        rows.append({"k": k, "estimate": mean, "closed_form": closed,
                     "gap_over_interval": float(ratio)})
    return {"rows": rows, "worst_gap_over_interval": float(worst), "draws": DIM_DRAWS}


def check_knn(rng):
    """The sampled estimator against the exact value, where both apply."""
    m0, S0 = np.array([0.4, -0.2]), np.array([[1.3, 0.5], [0.5, 0.8]])
    m1, S1 = np.array([-0.3, 0.6]), np.array([[0.9, -0.2], [-0.2, 1.6]])
    closed = gaussian_kl(m0, S0, m1, S1)
    vals = [knn_kl(rng.multivariate_normal(m0, S0, size=50_000),
                   rng.multivariate_normal(m1, S1, size=50_000)) for _ in range(20)]
    mean = float(np.mean(vals))
    return {"estimate": mean, "std": float(np.std(vals, ddof=1)),
            "closed_form": closed,
            "relative_error": float(abs(mean - closed) / closed)}


def check_properties(rng):
    """Identity, asymmetry, and refusal on a measure with no density."""
    m0, S0 = np.array([0.4, -0.2]), np.array([[1.3, 0.5], [0.5, 0.8]])
    m1, S1 = np.array([-0.3, 0.6]), np.array([[0.9, -0.2], [-0.2, 1.6]])
    forward, backward = gaussian_kl(m0, S0, m1, S1), gaussian_kl(m1, S1, m0, S0)
    try:
        gaussian_kl([0.0], [[0.0]], [0.0], [[1.0]])
        raises = False
    except ValueError:
        raises = True
    return {"self_divergence": gaussian_kl(m0, S0, m0, S0),
            "forward": forward, "backward": backward,
            "asymmetry": float(abs(forward - backward)),
            "degenerate_raises": raises}


def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED,
               "scalar": check_scalar(),
               "grid": check_grid(),
               "monte_carlo": check_monte_carlo(rng),
               "dimensions": check_dimensions(rng),
               "knn": check_knn(rng),
               "properties": check_properties(rng)}

    s, g, mc = results["scalar"], results["grid"], results["monte_carlo"]
    dim, kn, pr = results["dimensions"], results["knn"], results["properties"]
    macros = {
        "KLchkScalarErr": s["error"],
        "KLchkGridErr": g["error"],
        "KLchkGridPoints": f"{GRID_POINTS:,}".replace(",", "{,}"),
        "KLchkMCDraws": f"{MC_DRAWS:,}".replace(",", "{,}"),
        "KLchkMCGap": mc["gap_over_interval"],
        "KLchkDimWorst": dim["worst_gap_over_interval"],
        "KLchkKnnClosed": kn["closed_form"],
        "KLchkKnnEst": kn["estimate"],
        "KLchkKnnRel": f"{100.0 * kn['relative_error']:.1f}",
        "KLchkSelf": pr["self_divergence"],
        "KLchkAsym": pr["asymmetry"],
    }
    save("klcheck", results, macros)

    print(f"scalar formula          error {s['error']:.3e}")
    print(f"grid integration        error {g['error']:.3e}  ({GRID_POINTS} points)")
    print(f"monte carlo             gap {mc['gap_over_interval']:.3f} of the 95% interval")
    print(f"dimensions 3, 5, 8      worst gap {dim['worst_gap_over_interval']:.3f} of the interval")
    print(f"knn estimator           {kn['estimate']:.4f} against {kn['closed_form']:.4f}"
          f"  ({100 * kn['relative_error']:.1f}% relative)")
    print(f"KL(P||P)                {pr['self_divergence']}")
    print(f"degenerate input raises {pr['degenerate_raises']}")
    return results


if __name__ == "__main__":
    main()
