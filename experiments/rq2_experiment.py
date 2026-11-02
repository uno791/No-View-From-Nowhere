"""
RQ2 experiment. Can causal knowledge be encoded in the category CSprob of
convex spaces of probability measures, and when does the encoding preserve
interventional consistency?

Parts
    A  the two readings of "blend two interventions" give different measures
    B  closure: where a blend of causal measures IS and IS NOT causal knowledge
    C  affineness of pushforward, with a control that is not affine
    D  interventional consistency, on a study-hours model where it holds and
       on a near-identical model where it fails
    E  the functor laws of the encoding functor

Story used in part D (synthetic, chosen for readability):
    micro model   H1 lecture study hours, H2 tutorial practice hours, Mk test mark
    macro model   T total study hours,    Mk test mark
    abstraction   T = H1 + H2, Mk = Mk
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.abstraction import LinearAbstraction, push_gaussian   # noqa: E402
from core.kl import grid_kl, knn_kl                             # noqa: E402
from core.reporting import FIGURES_DIR, save                    # noqa: E402
from core.scm import LinearGaussianSCM                          # noqa: E402

SEED = 20260918
LAM = 0.5
A1, A2 = 1.0, 4.0     # the two soft interventions on the coefficient of X -> Y


def gauss2(x, y, cov):
    """Density of a zero-mean bivariate Gaussian on a grid."""
    inv = np.linalg.inv(cov)
    quad = inv[0, 0] * x * x + 2 * inv[0, 1] * x * y + inv[1, 1] * y * y
    return np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


# ---------------------------------------------------------------------------
# Part A: blending measures is not blending mechanisms
# ---------------------------------------------------------------------------

def part_a():
    """Model X = Z_X, Y = a X + Z_Y with unit noises; interventions a = 1 and a = 4."""
    cov = lambda a: np.array([[1.0, a], [a, a * a + 1.0]])
    out = {}
    # Reading A: blend the measures (flip a coin, then use one intervention).
    var_a = LAM * cov(A1)[1, 1] + (1 - LAM) * cov(A2)[1, 1]
    # Reading B: blend the mixing maps, then push the noise through once.
    a_blend = LAM * A1 + (1 - LAM) * A2
    var_b = cov(a_blend)[1, 1]
    out["var_y_measure_blend"], out["var_y_map_blend"], out["a_blend"] = var_a, var_b, a_blend

    # The measure blend is not Gaussian: exact excess kurtosis of its Y marginal.
    v1, v2 = cov(A1)[1, 1], cov(A2)[1, 1]
    fourth = 3 * (LAM * v1 ** 2 + (1 - LAM) * v2 ** 2)
    out["excess_kurtosis"] = fourth / var_a ** 2 - 3

    # KL divergences by numerical integration on a fine grid.
    xs, ys = np.linspace(-7, 7, 701), np.linspace(-34, 34, 1701)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    mix = LAM * gauss2(X, Y, cov(A1)) + (1 - LAM) * gauss2(X, Y, cov(A2))
    out["grid_mass"] = float(mix.sum() * dx * dy)
    out["kl_mix_vs_map_blend"] = grid_kl(mix, gauss2(X, Y, cov(a_blend)), dx * dy)
    # The Gaussian closest to the mixture in KL is the one with the same mean and covariance.
    cov_mm = LAM * cov(A1) + (1 - LAM) * cov(A2)
    out["kl_mix_vs_best_gaussian"] = grid_kl(mix, gauss2(X, Y, cov_mm), dx * dy)
    figure_blend(v1, v2, var_a, var_b)
    return out


def figure_blend(v1, v2, var_a, var_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    y = np.linspace(-12, 12, 1200)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(y, LAM * norm.pdf(y, 0, np.sqrt(v1)) + (1 - LAM) * norm.pdf(y, 0, np.sqrt(v2)),
            color="black", lw=1.5, label="blend of the measures (reading A)")
    ax.plot(y, norm.pdf(y, 0, np.sqrt(var_b)), color="#2166ac", lw=1.3, ls="--",
            label="blend of the mechanisms (reading B)")
    ax.plot(y, norm.pdf(y, 0, np.sqrt(var_a)), color="#b2182b", lw=1.3, ls=":",
            label="Gaussian with the variance of reading A")
    ax.set_xlabel("$y$")
    ax.set_ylabel("density of $Y$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "rq2_blend.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part B: closure, where it holds and where it fails
# ---------------------------------------------------------------------------

def conditional_quantile_mechanism(x, z):
    """A single mechanism Y = f(X, Z_Y) whose output law is the measure blend.

    Given X = x, the blend says Y is N(A1 x, 1) with probability LAM and
    N(A2 x, 1) otherwise. Its conditional CDF is
        F(y | x) = LAM * Phi(y - A1 x) + (1 - LAM) * Phi(y - A2 x).
    Setting f(x, z) = F^-1( Phi(z) | x ) turns ONE standard normal noise z into
    a draw from that conditional law. F is increasing in y, so bisection works.
    """
    u = norm.cdf(z)
    lo = np.minimum(A1 * x, A2 * x) - 12.0
    hi = np.maximum(A1 * x, A2 * x) + 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = LAM * norm.cdf(mid - A1 * x) + (1 - LAM) * norm.cdf(mid - A2 * x) < u
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def sample_measure_blend(n, rng):
    x = rng.normal(size=n)
    coin = rng.random(n) < LAM
    y = np.where(coin, A1, A2) * x + rng.normal(size=n)
    return np.column_stack([x, y])


def part_b(rng):
    out = {}
    n, reps = 20_000, 20

    # B1. Complete DAG X -> Y, unrestricted mechanisms: the blend IS a soft intervention.
    zx, zy = rng.normal(size=n), rng.normal(size=n)
    y = conditional_quantile_mechanism(zx, zy)
    # Exactness check (probability integral transform): F(f(x, z) | x) must equal Phi(z).
    F_at_y = LAM * norm.cdf(y - A1 * zx) + (1 - LAM) * norm.cdf(y - A2 * zx)
    out["pit_error"] = float(np.max(np.abs(F_at_y - norm.cdf(zy))))

    def mechanism_samples(m, r):
        a, b = r.normal(size=m), r.normal(size=m)
        return np.column_stack([a, conditional_quantile_mechanism(a, b)])

    test = [knn_kl(mechanism_samples(n, rng), sample_measure_blend(n, rng)) for _ in range(reps)]
    null = [knn_kl(sample_measure_blend(n, rng), sample_measure_blend(n, rng)) for _ in range(reps)]
    out["mechanism_kl_mean"], out["mechanism_kl_std"] = float(np.mean(test)), float(np.std(test, ddof=1))
    out["mechanism_null_mean"], out["mechanism_null_std"] = float(np.mean(null)), float(np.std(null, ddof=1))
    out["mechanism_z"] = float((np.mean(test) - np.mean(null))
                               / np.sqrt((np.var(test, ddof=1) + np.var(null, ddof=1)) / reps))
    # Sensitivity: the map blend (a = 2.5) is a different measure and the estimator sees it.
    a_blend = LAM * A1 + (1 - LAM) * A2
    xb = rng.normal(size=n)
    map_blend = np.column_stack([xb, a_blend * xb + rng.normal(size=n)])
    out["map_blend_kl"] = knn_kl(map_blend, sample_measure_blend(n, rng))

    # B2. DAG with a missing edge (X and Y not connected): closure FAILS.
    # Interventions: S1 leaves both roots alone, S2 shifts both by `shift`.
    # Every measure the model entails is a product measure (mutual information 0).
    # The blend is not a product measure: its mutual information is positive.
    shift = 3.0
    g = np.linspace(-7, 10, 1701)
    d = g[1] - g[0]
    X, Y = np.meshgrid(g, g, indexing="ij")
    mix = LAM * norm.pdf(X) * norm.pdf(Y) + (1 - LAM) * norm.pdf(X - shift) * norm.pdf(Y - shift)
    marg = LAM * norm.pdf(g) + (1 - LAM) * norm.pdf(g - shift)
    out["missing_edge_shift"] = shift
    out["missing_edge_cov"] = LAM * (1 - LAM) * shift * shift          # exact covariance of the blend
    out["missing_edge_mutual_information"] = grid_kl(mix, np.outer(marg, marg), d * d)
    return out


# ---------------------------------------------------------------------------
# Part C: affineness of pushforward
# ---------------------------------------------------------------------------

def hist_kl(a, b, edges):
    pa = np.histogram(a, bins=edges)[0] + 0.5
    pb = np.histogram(b, bins=edges)[0] + 0.5
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))


def part_c(rng):
    out = {}
    cov = lambda a: np.array([[1.0, a], [a, a * a + 1.0]])
    S1, S2 = cov(A1), cov(A2)

    # C1. The check promised in the proposal: on covariance matrices the map
    # S -> F S F^T commutes with blending, for any F whatsoever.
    maps = {
        "sensible [1, 0.5]": np.array([[1.0, 0.5]]),
        "arbitrary [1, -1]": np.array([[1.0, -1.0]]),
        "random": rng.normal(size=(1, 2)),
        "rank deficient [0, 0]": np.array([[0.0, 0.0]]),
    }
    out["covariance_defect"] = {
        k: float(np.max(np.abs(F @ (LAM * S1 + (1 - LAM) * S2) @ F.T
                               - (LAM * F @ S1 @ F.T + (1 - LAM) * F @ S2 @ F.T))))
        for k, F in maps.items()
    }

    # C2. The same statement on full measures, including a nonlinear map.
    # Left side:  draw from the blend, then apply the map.
    # Right side: apply the map to each component, then blend the two histograms.
    n = 200_000
    funcs = dict(maps)
    funcs["nonlinear x^2 - y"] = None
    out["measure_defect"], out["measure_null"] = {}, {}
    for k, F in funcs.items():
        apply = (lambda v: v[:, 0] ** 2 - v[:, 1]) if F is None else (lambda v, F=F: (v @ F.T)[:, 0])
        comp1 = rng.multivariate_normal([0, 0], S1, size=n)
        comp2 = rng.multivariate_normal([0, 0], S2, size=n)
        left = apply(sample_measure_blend(n, rng))
        left_again = apply(sample_measure_blend(n, rng))
        lo, hi = np.quantile(left, [0.001, 0.999])
        edges = np.linspace(lo - 1e-9, hi + 1e-9, 61)
        h1 = np.histogram(apply(comp1), bins=edges)[0] + 0.5
        h2 = np.histogram(apply(comp2), bins=edges)[0] + 0.5
        right = LAM * h1 / h1.sum() + (1 - LAM) * h2 / h2.sum()
        hl = np.histogram(left, bins=edges)[0] + 0.5
        hl = hl / hl.sum()
        out["measure_defect"][k] = float(np.sum(hl * np.log(hl / right)))
        out["measure_null"][k] = hist_kl(left, left_again, edges)

    # C3. Control: a map on measures that is NOT a pushforward and NOT affine.
    # G replaces a measure by the Gaussian with the same mean and covariance.
    # G(blend) is one Gaussian; blend(G(chi1), G(chi2)) is a two-component mixture.
    xs, ys = np.linspace(-7, 7, 701), np.linspace(-34, 34, 1701)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    blend_of_G = LAM * gauss2(X, Y, S1) + (1 - LAM) * gauss2(X, Y, S2)
    G_of_blend = gauss2(X, Y, LAM * S1 + (1 - LAM) * S2)
    out["nonaffine_control_kl"] = grid_kl(blend_of_G, G_of_blend, cell)
    return out


# ---------------------------------------------------------------------------
# Part D: interventional consistency
# ---------------------------------------------------------------------------

def build_pair(b1, b2):
    """Micro model with mark coefficients (b1, b2) and the macro model fitted to it.

    The macro coefficient c and noise variance are always chosen so that the
    abstraction reproduces the OBSERVATIONAL macro law exactly. Whether it also
    reproduces the interventional laws is what the experiment measures.
    """
    v1, v2, v3 = 1.0, 0.64, 0.25
    micro = LinearGaussianSCM(C=[[0, 0, 0], [0, 0, 0], [b1, b2, 0]], noise_var=[v1, v2, v3],
                              names=["H1", "H2", "Mk"])
    F = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _, cov = push_gaussian(F, micro.law())
    c = cov[0, 1] / cov[0, 0]
    macro = LinearGaussianSCM(C=[[0, 0], [c, 0]], noise_var=[cov[0, 0], cov[1, 1] - c * c * cov[0, 0]],
                              names=["T", "Mk"])
    return LinearAbstraction(micro, macro, F), c


def part_d(rng):
    out = {}
    for label, (b1, b2) in {"consistent": (0.6, 0.6), "inconsistent": (0.9, 0.2)}.items():
        alpha, c = build_pair(b1, b2)
        micro, macro = alpha.micro, alpha.macro
        res = {"b": [b1, b2], "c": float(c), "macro_noise_var": macro.noise_var.tolist()}
        res["naturality_residual"] = alpha.naturality_residual()
        res["noise_is_preserved"] = alpha.noise_is_preserved()
        res["kl_observational"] = alpha.ic_gap(micro, macro)

        # Hard intervention do(T = 2) on the macro model. Two micro interventions map to it.
        macro_hard = macro.do_hard({0: 2.0})
        for tag, (h1, h2) in {"lift_2_0": (2.0, 0.0), "lift_0_2": (0.0, 2.0)}.items():
            res[f"kl_hard_{tag}"] = alpha.ic_gap(micro.do_hard({0: h1, 1: h2}), macro_hard, coords=[1])

        # Soft intervention on the mark mechanism: macro c -> 1.0, micro b -> (1.0 / c) b.
        c_new = 1.0
        macro_soft = macro.do_soft(1, coeffs={0: c_new})
        micro_soft = micro.do_soft(2, coeffs={0: b1 * c_new / c, 1: b2 * c_new / c})
        res["kl_soft_mark"] = alpha.ic_gap(micro_soft, macro_soft)

        # Soft intervention on study hours: every hour variable is spread out by a factor 1.5.
        macro_soft_t = macro.do_soft(0, gain=1.5)
        micro_soft_t = micro.do_soft(0, gain=1.5).do_soft(1, gain=1.5)
        res["kl_soft_hours"] = alpha.ic_gap(micro_soft_t, macro_soft_t)

        # Value level commutation alpha_X o I = I' o alpha_X for the soft mark intervention.
        x = micro.sample(10_000, rng)
        res["value_gap_soft_mark"] = alpha.value_level_gap(micro_soft, macro_soft, x)

        # Affineness of the pushforward along F, on covariances. Holds in BOTH cases.
        S1, S2 = micro.law()[1], micro_soft.law()[1]
        F = alpha.F
        res["affine_defect"] = float(np.max(np.abs(
            F @ (LAM * S1 + (1 - LAM) * S2) @ F.T - (LAM * F @ S1 @ F.T + (1 - LAM) * F @ S2 @ F.T))))
        out[label] = res
    figure_ic(out)
    return out


def figure_ic(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    keys = ["kl_observational", "kl_hard_lift_2_0", "kl_hard_lift_0_2", "kl_soft_mark", "kl_soft_hours"]
    labels = ["observational", "do($T$=2)\nvia (2, 0)", "do($T$=2)\nvia (0, 2)", "soft: mark\nmechanism", "soft: hours\nspread"]
    pos = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(pos - 0.18, [res["consistent"][k] for k in keys], width=0.36, color="#2166ac", label="consistent model")
    ax.bar(pos + 0.18, [res["inconsistent"][k] for k in keys], width=0.36, color="#b2182b", label="inconsistent model")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("KL(pushed micro law || macro law)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "rq2_consistency.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part E: functor laws of the encoding functor
# ---------------------------------------------------------------------------

def part_e():
    alpha, c = build_pair(0.6, 0.6)
    micro = alpha.micro
    # A second abstraction, macro -> outcome model with the single variable O = Mk.
    F2 = np.array([[0.0, 1.0]])
    laws = {
        "observational": micro.law(),
        "soft": micro.do_soft(2, coeffs={0: 1.0, 1: 1.0}).law(),
        "soft_hours": micro.do_soft(0, gain=1.5).do_soft(1, gain=1.5).law(),
    }
    out = {"identity_defect": 0.0, "composition_defect": 0.0}
    for law in laws.values():
        m, S = push_gaussian(np.eye(3), law)
        out["identity_defect"] = max(out["identity_defect"], float(np.max(np.abs(S - law[1]))))
        one_step = push_gaussian(F2 @ alpha.F, law)
        two_steps = push_gaussian(F2, push_gaussian(alpha.F, law))
        out["composition_defect"] = max(out["composition_defect"],
                                        float(np.max(np.abs(one_step[1] - two_steps[1]))),
                                        float(np.max(np.abs(one_step[0] - two_steps[0]))))
    return out


def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "lambda": LAM, "part_a": part_a(), "part_b": part_b(rng),
               "part_c": part_c(rng), "part_d": part_d(rng), "part_e": part_e()}
    A, B, C, D, E = (results[k] for k in ("part_a", "part_b", "part_c", "part_d", "part_e"))
    macros = {
        "RQtwoVarMeasure": A["var_y_measure_blend"], "RQtwoVarMap": A["var_y_map_blend"],
        "RQtwoKurt": A["excess_kurtosis"], "RQtwoKLMap": A["kl_mix_vs_map_blend"],
        "RQtwoKLBestGauss": A["kl_mix_vs_best_gaussian"],
        "RQtwoPIT": B["pit_error"], "RQtwoMechKL": B["mechanism_kl_mean"], "RQtwoMechKLStd": B["mechanism_kl_std"],
        "RQtwoMechNull": B["mechanism_null_mean"], "RQtwoMechNullStd": B["mechanism_null_std"],
        "RQtwoMechZ": f"{B['mechanism_z']:.2f}", "RQtwoMapBlendKL": B["map_blend_kl"],
        "RQtwoMissCov": B["missing_edge_cov"], "RQtwoMissMI": B["missing_edge_mutual_information"],
        "RQtwoCovDefect": max(C["covariance_defect"].values()),
        "RQtwoNonAffine": C["nonaffine_control_kl"],
        "RQtwoIdDefect": E["identity_defect"], "RQtwoCompDefect": E["composition_defect"],
    }
    for label, tag in (("consistent", "Con"), ("inconsistent", "Inc")):
        r = D[label]
        macros[f"RQtwo{tag}C"] = r["c"]
        macros[f"RQtwo{tag}NoiseVar"] = r["macro_noise_var"][1]
        macros[f"RQtwo{tag}Nat"] = r["naturality_residual"]
        macros[f"RQtwo{tag}Obs"] = r["kl_observational"]
        macros[f"RQtwo{tag}HardA"] = r["kl_hard_lift_2_0"]
        macros[f"RQtwo{tag}HardB"] = r["kl_hard_lift_0_2"]
        macros[f"RQtwo{tag}SoftMark"] = r["kl_soft_mark"]
        macros[f"RQtwo{tag}SoftHours"] = r["kl_soft_hours"]
        macros[f"RQtwo{tag}Value"] = r["value_gap_soft_mark"]
        macros[f"RQtwo{tag}Affine"] = r["affine_defect"]
    for i, (k, v) in enumerate(C["measure_defect"].items()):
        tag = "ABCDE"[i]
        macros[f"RQtwoMeasDefect{tag}"] = v
        macros[f"RQtwoMeasNull{tag}"] = C["measure_null"][k]
    save("rq2", results, macros)
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
