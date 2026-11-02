"""
Kullback-Leibler divergence tools.

KL divergence is the verification measure used throughout the project.
KL(P || Q) is zero exactly when P and Q are the same probability measure,
and positive otherwise. Every experiment therefore reduces to one question:
"is the KL divergence between the two routes zero?"

Three ways of computing it are provided, because different experiments
need different ones:

1. gaussian_kl   exact closed form, for two Gaussian measures.
2. knn_kl        estimate from samples, for measures with no closed form.
3. grid_kl       numerical integration of two known densities on a grid.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def gaussian_kl(m0, S0, m1, S1) -> float:
    """Exact KL( N(m0, S0) || N(m1, S1) ) for non-degenerate Gaussians.

    Formula (k is the dimension):
        0.5 * [ tr(S1^-1 S0) + (m1-m0)^T S1^-1 (m1-m0) - k + ln det S1 - ln det S0 ]
    """
    m0 = np.atleast_1d(np.asarray(m0, dtype=float))
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    S0 = np.atleast_2d(np.asarray(S0, dtype=float))
    S1 = np.atleast_2d(np.asarray(S1, dtype=float))
    k = m0.shape[0]
    sign0, logdet0 = np.linalg.slogdet(S0)
    sign1, logdet1 = np.linalg.slogdet(S1)
    if sign0 <= 0 or sign1 <= 0:
        raise ValueError("gaussian_kl needs positive definite covariances; "
                         "a degenerate Gaussian has no density and KL is not finite.")
    diff = m1 - m0
    trace_term = np.trace(np.linalg.solve(S1, S0))
    maha_term = float(diff @ np.linalg.solve(S1, diff))
    value = 0.5 * (trace_term + maha_term - k + logdet1 - logdet0)
    # Rounding can produce a tiny negative number when the true value is 0.
    return float(max(value, 0.0)) if abs(value) < 1e-12 else float(value)


def symmetric_gaussian_kl(m0, S0, m1, S1) -> float:
    """Jeffreys divergence: KL(P||Q) + KL(Q||P). Symmetric in its arguments."""
    return gaussian_kl(m0, S0, m1, S1) + gaussian_kl(m1, S1, m0, S0)


def knn_kl(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """k-nearest-neighbour estimate of KL(P || Q) from samples x ~ P, y ~ Q.

    Estimator of Wang, Kulkarni and Verdu (2009):
        D = (d/n) * sum_i ln( nu_k(i) / rho_k(i) ) + ln( m / (n-1) )
    where rho_k(i) is the distance from x_i to its k-th neighbour among the
    other x samples, and nu_k(i) the distance from x_i to its k-th neighbour
    among the y samples.

    The estimate is noisy, so it is always reported next to a null baseline
    (see null_baseline below): the value the estimator returns when P = Q.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    n, d = x.shape
    m = y.shape[0]
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    # k+1 because the nearest neighbour of x_i inside x is x_i itself.
    rho = tree_x.query(x, k=k + 1)[0][:, -1]
    nu = tree_y.query(x, k=k)[0]
    nu = nu[:, -1] if k > 1 else nu
    tiny = 1e-300
    return float(d * np.mean(np.log((nu + tiny) / (rho + tiny))) + np.log(m / (n - 1)))


def null_baseline(sampler, n: int, reps: int, rng: np.random.Generator, k: int = 5):
    """Distribution of knn_kl when both samples come from the SAME measure.

    sampler(n, rng) must return an (n, d) array. Returns (mean, std) over reps.
    A test statistic inside mean +/- 3 std is indistinguishable from zero.
    """
    vals = [knn_kl(sampler(n, rng), sampler(n, rng), k=k) for _ in range(reps)]
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def grid_kl(p: np.ndarray, q: np.ndarray, cell: float) -> float:
    """KL(P || Q) by numerical integration of densities tabulated on a grid.

    p, q  arrays of density values on the same grid (any dimension)
    cell  volume of one grid cell (dx in 1-D, dx*dy in 2-D)
    Cells where p is zero contribute zero (the convention 0 ln 0 = 0).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 1e-300
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask] + 1e-300))) * cell)
