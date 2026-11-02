"""
RQ1 experiment. Does the categorical representation of an SCM lose any
interventional meaning?

Story used for the variables (synthetic, chosen for readability):
    X  weekly study hours (standardised)
    Y  test mark (standardised),   Y = a X + Z_Y

The experiment answers one question in several settings: if an intervention
is carried out the categorical way, by pushing the observational measure
through I = M_I o M^-1, is the result the same measure that classical
do-calculus predicts? "The same" is tested with KL divergence.

Parts
    A  linear Gaussian model of the proposal, exact closed forms
    B  nonlinear models from the three invertible classes, on samples
    C  a model that is NOT invertible, to show why invertibility is needed
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kl import gaussian_kl, knn_kl, null_baseline          # noqa: E402
from core.reporting import FIGURES_DIR, save                     # noqa: E402
from core.scm import GeneralSCM, LinearGaussianSCM, Mechanism    # noqa: E402

SEED = 20260917


# ---------------------------------------------------------------------------
# Part A: the linear Gaussian running example
# ---------------------------------------------------------------------------

def part_a(rng):
    a, var_x, var_y = 0.8, 1.0, 0.36
    x0, a_soft, y0 = 1.5, 0.3, 1.0
    scm = LinearGaussianSCM(C=[[0, 0], [a, 0]], noise_var=[var_x, var_y], names=["X", "Y"])
    out = {"a": a, "var_x": var_x, "var_y": var_y, "x0": x0, "a_soft": a_soft, "y0": y0}

    # A1. The functor data agrees with the hand derivation.
    M = scm.mixing_matrix
    _, cov = scm.law()
    cov_hand = np.array([[var_x, a * var_x], [a * var_x, a * a * var_x + var_y]])
    out["mixing_matrix"] = M.tolist()
    out["cov_matches_hand"] = float(np.max(np.abs(cov - cov_hand)))

    # A2. Invertibility: M^-1 = I - C, and the round trip returns the noise.
    z = scm.sample_noise(100_000, rng)
    out["inverse_is_I_minus_C"] = float(np.max(np.abs(np.linalg.inv(M) - (np.eye(2) - scm.C))))
    out["round_trip_error"] = float(np.max(np.abs(scm.inverse_mixing(scm.mixing(z)) - z)))

    # A3. Hard intervention do(X = x0) as a natural transformation <Id, H>.
    hard = scm.do_hard({0: x0})
    A, b = scm.intervention_map(hard)                 # H(x) = A x + b
    out["H_matrix"], out["H_shift"] = A.tolist(), b.tolist()
    # Naturality square: H o M = M_H o Id, checked on every noise sample.
    out["hard_naturality_error"] = float(np.max(np.abs((scm.mixing(z) @ A.T + b) - hard.mixing(z))))
    # Pushforward of the observational law along H, Y coordinate, closed form.
    mean_obs, cov_obs = scm.law()
    push_mean, push_cov = A @ mean_obs + b, A @ cov_obs @ A.T
    out["hard_pushed_mean_y"], out["hard_pushed_var_y"] = float(push_mean[1]), float(push_cov[1, 1])
    out["hard_classical_mean_y"], out["hard_classical_var_y"] = a * x0, var_y
    out["hard_kl_exact"] = gaussian_kl([push_mean[1]], [[push_cov[1, 1]]], [a * x0], [[var_y]])
    # Same comparison on samples: transported observational data against fresh do-samples.
    n = 100_000
    y_categorical = (scm.sample(n, rng) @ A.T + b)[:, 1]
    y_classical = a * x0 + rng.normal(size=n) * np.sqrt(var_y)
    out["hard_kl_samples"] = knn_kl(y_categorical, y_classical)
    base = null_baseline(lambda m, r: a * x0 + r.normal(size=(m, 1)) * np.sqrt(var_y), n, 20, rng)
    out["hard_kl_null_mean"], out["hard_kl_null_std"] = base

    # A4. Soft intervention a -> a_soft as a natural transformation <Id, S>.
    soft = scm.do_soft(1, coeffs={0: a_soft})
    S, s_shift = scm.intervention_map(soft)
    out["S_matrix"] = S.tolist()
    out["soft_naturality_error"] = float(np.max(np.abs(S @ M - soft.mixing_matrix)))
    soft_mean, soft_cov = soft.law()
    out["soft_kl_exact"] = gaussian_kl(S @ mean_obs + s_shift, S @ cov_obs @ S.T, soft_mean, soft_cov)

    # A5. Composition of intervention morphisms: a -> a_soft -> a_third equals a -> a_third.
    third = scm.do_soft(1, coeffs={0: -1.1})
    S2, _ = soft.intervention_map(third)
    S_direct, _ = scm.intervention_map(third)
    out["composition_error"] = float(np.max(np.abs(S2 @ S - S_direct)))

    # A6. Seeing is not doing. Intervene on the EFFECT: do(Y = y0).
    # Classical answer: X is untouched, X ~ N(0, var_x).
    # Conditioning answer: X | Y = y0 is a different Gaussian.
    hard_y = scm.do_hard({1: y0})
    Ay, by = scm.intervention_map(hard_y)
    pm, pc = Ay @ mean_obs + by, Ay @ cov_obs @ Ay.T
    var_y_total = a * a * var_x + var_y
    cond_mean = a * var_x * y0 / var_y_total
    cond_var = var_x - (a * var_x) ** 2 / var_y_total
    out["doY_kl_categorical_vs_do"] = gaussian_kl([pm[0]], [[pc[0, 0]]], [0.0], [[var_x]])
    out["doY_kl_conditioning_vs_do"] = gaussian_kl([cond_mean], [[cond_var]], [0.0], [[var_x]])
    out["cond_mean"], out["cond_var"] = cond_mean, cond_var

    figure_rq1(y_categorical, a * x0, var_y, scm, Ay, by, var_x, cond_mean, cond_var, rng)
    return out


def figure_rq1(y_cat, mean_y, var_y, scm, Ay, by, var_x, cond_mean, cond_var, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))

    grid = np.linspace(mean_y - 4 * np.sqrt(var_y), mean_y + 4 * np.sqrt(var_y), 400)
    axes[0].hist(y_cat[:60_000], bins=80, density=True, color="#9db7d5", label="categorical route: samples pushed through $H$")
    axes[0].plot(grid, norm.pdf(grid, mean_y, np.sqrt(var_y)), color="black", lw=1.4, label="classical do-calculus density")
    axes[0].set_title(r"(a) $\mathrm{do}(X = x_0)$: law of $Y$")
    axes[0].set_xlabel("$y$")
    axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    axes[0].set_ylim(0, 1.0)

    x_cat = (scm.sample(60_000, rng) @ Ay.T + by)[:, 0]
    grid = np.linspace(-4, 4, 400)
    axes[1].hist(x_cat, bins=80, density=True, color="#9db7d5", label="categorical route")
    axes[1].plot(grid, norm.pdf(grid, 0, np.sqrt(var_x)), color="black", lw=1.4, label="classical do-calculus")
    axes[1].plot(grid, norm.pdf(grid, cond_mean, np.sqrt(cond_var)), color="#b2182b", lw=1.4, ls="--",
                 label="conditioning on $Y = y_0$ (seeing)")
    axes[1].set_title(r"(b) $\mathrm{do}(Y = y_0)$: law of $X$")
    axes[1].set_xlabel("$x$")
    axes[1].legend(fontsize=7, loc="upper left", frameon=False)
    axes[1].set_ylim(0, 0.75)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("density")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "rq1_interventions.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part B: the three invertible model classes, nonlinear, on samples
# ---------------------------------------------------------------------------

def build_models():
    root = Mechanism(kind="anm", parents=[])
    anm = GeneralSCM([
        root,
        Mechanism(kind="anm", parents=[0], f=lambda p: 2.0 * np.tanh(p[:, 0])),
        Mechanism(kind="anm", parents=[1], f=lambda p: 0.5 * p[:, 0] ** 2),
    ], ["X1", "X2", "X3"])
    anm_soft = Mechanism(kind="anm", parents=[0], f=lambda p: np.sin(2.0 * p[:, 0]))

    pnl = GeneralSCM([
        root,
        Mechanism(kind="pnl", parents=[0], f=lambda p: 1.5 * p[:, 0], g=np.arcsinh, g_inv=np.sinh),
        Mechanism(kind="pnl", parents=[1], f=lambda p: 2.0 * p[:, 0], g=np.arcsinh, g_inv=np.sinh),
    ], ["X1", "X2", "X3"])
    pnl_soft = Mechanism(kind="pnl", parents=[0], f=lambda p: -p[:, 0] ** 2, g=np.arcsinh, g_inv=np.sinh)

    lsnm = GeneralSCM([
        root,
        Mechanism(kind="lsnm", parents=[0], f=lambda p: np.sin(p[:, 0]), h=lambda p: np.sqrt(0.5 + p[:, 0] ** 2)),
        Mechanism(kind="lsnm", parents=[1], f=lambda p: 0.7 * p[:, 0], h=lambda p: 0.5 + 1.0 / (1.0 + p[:, 0] ** 2)),
    ], ["X1", "X2", "X3"])
    lsnm_soft = Mechanism(kind="lsnm", parents=[0], f=lambda p: 0.5 * p[:, 0], h=lambda p: 1.0 + 0.5 * np.cos(p[:, 0]))

    return {"anm": (anm, anm_soft), "pnl": (pnl, pnl_soft), "lsnm": (lsnm, lsnm_soft)}


def part_b(rng):
    n, reps = 20_000, 20
    out = {}
    for label, (scm, soft_mech) in build_models().items():
        res = {}
        z = scm.sample_noise(n, rng)
        x = scm.mixing(z)
        res["round_trip_error"] = float(np.max(np.abs(scm.inverse_mixing(x) - z)))

        for kind, intervened, coords in (
            ("hard", scm.do_hard({1: 1.0}), [0, 2]),      # compare on the coordinates that keep a density
            ("soft", scm.do_soft(1, soft_mech), [0, 1, 2]),
        ):
            transport = scm.intervention_map(intervened)
            # Value level: transporting x must equal re-running the intervened model on the SAME noise.
            res[f"{kind}_value_error"] = float(np.max(np.abs(transport(x) - intervened.mixing(z))))
            # Measure level: transported observational data against independent interventional data.
            # The estimator is noisy, so the comparison is repeated `reps` times and set against a
            # null distribution (both samples drawn from the classical intervened model).
            test = [knn_kl(transport(scm.sample(n, rng))[:, coords], intervened.sample(n, rng)[:, coords])
                    for _ in range(reps)]
            null = [knn_kl(intervened.sample(n, rng)[:, coords], intervened.sample(n, rng)[:, coords])
                    for _ in range(reps)]
            res[f"{kind}_kl"], res[f"{kind}_kl_std"] = float(np.mean(test)), float(np.std(test, ddof=1))
            res[f"{kind}_null_mean"], res[f"{kind}_null_std"] = float(np.mean(null)), float(np.std(null, ddof=1))
            pooled = np.sqrt((res[f"{kind}_kl_std"] ** 2 + res[f"{kind}_null_std"] ** 2) / reps)
            res[f"{kind}_z_score"] = (res[f"{kind}_kl"] - res[f"{kind}_null_mean"]) / pooled
            # Sensitivity check: the estimator must react when the measures really differ.
            res[f"{kind}_kl_vs_observational"] = knn_kl(scm.sample(n, rng)[:, coords],
                                                        intervened.sample(n, rng)[:, coords])
        out[label] = res
    return out


# ---------------------------------------------------------------------------
# Part C: why invertibility is needed
# ---------------------------------------------------------------------------

def part_c():
    """X2 = (X1 + Z2)^2 cannot be solved for Z2: two noise values give the same data.

    The would-be transport I = M_I o M^-1 then has two candidate outputs for a
    single observed point, so it is not a function. Numbers below use the soft
    intervention X2 = (2 X1 + Z2)^2 at the observed point (x1, x2) = (1, 4).
    """
    x1, x2 = 1.0, 4.0
    noise_candidates = [-x1 + np.sqrt(x2), -x1 - np.sqrt(x2)]          # both reproduce (x1, x2)
    reproduced = [(x1 + zc) ** 2 for zc in noise_candidates]
    transported = [(2 * x1 + zc) ** 2 for zc in noise_candidates]
    return {"noise_candidates": noise_candidates, "reproduced_x2": reproduced, "transported_x2": transported}


def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "part_a": part_a(rng), "part_b": part_b(rng), "part_c": part_c()}
    A, B, C = results["part_a"], results["part_b"], results["part_c"]
    macros = {
        "RQoneRoundTrip": A["round_trip_error"],
        "RQoneHardNat": A["hard_naturality_error"],
        "RQoneHardKLExact": A["hard_kl_exact"],
        "RQoneHardKLSamples": A["hard_kl_samples"],
        "RQoneHardNullMean": A["hard_kl_null_mean"],
        "RQoneHardNullStd": A["hard_kl_null_std"],
        "RQoneSoftNat": A["soft_naturality_error"],
        "RQoneSoftKLExact": A["soft_kl_exact"],
        "RQoneCompErr": A["composition_error"],
        "RQoneDoYCat": A["doY_kl_categorical_vs_do"],
        "RQoneDoYCond": A["doY_kl_conditioning_vs_do"],
        "RQoneCondMean": A["cond_mean"],
        "RQoneCondVar": A["cond_var"],
        "RQoneNonInvA": C["transported_x2"][0],
        "RQoneNonInvB": C["transported_x2"][1],
    }
    for label, tag in (("anm", "Anm"), ("pnl", "Pnl"), ("lsnm", "Lsnm")):
        r = B[label]
        macros[f"RQone{tag}RoundTrip"] = r["round_trip_error"]
        for kind, ktag in (("hard", "Hard"), ("soft", "Soft")):
            macros[f"RQone{tag}{ktag}Value"] = r[f"{kind}_value_error"]
            macros[f"RQone{tag}{ktag}KL"] = r[f"{kind}_kl"]
            macros[f"RQone{tag}{ktag}KLStd"] = r[f"{kind}_kl_std"]
            macros[f"RQone{tag}{ktag}Null"] = r[f"{kind}_null_mean"]
            macros[f"RQone{tag}{ktag}NullStd"] = r[f"{kind}_null_std"]
            macros[f"RQone{tag}{ktag}Z"] = f"{r[f'{kind}_z_score']:.2f}"
            macros[f"RQone{tag}{ktag}Sens"] = r[f"{kind}_kl_vs_observational"]
    save("rq1", results, macros)
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
