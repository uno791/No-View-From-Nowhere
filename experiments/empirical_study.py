"""
Empirical study: four quantitative experiments that turn the theory of
Chapters 6 and 7 into measured curves rather than single table entries.

E1  Goodness of fit against interventional validity.
    500 random micro models, each with a macro model fitted so that the two
    agree exactly on observational data. Does that fit predict whether the
    abstraction is interventionally consistent? It does not. What does
    predict it is a geometric quantity, and the prediction is exact.

E2  The blend gap as a function of the mixing weight.
    Sweep lambda from 0 to 1 and measure both the variance gap between the
    two readings of "blend", and the distance from the blend to the nearest
    member of the linear Gaussian causal knowledge.

E3  Knowledge decay along a chain of subjects.
    Two distinguishable messages leave subject 1. How distinguishable are
    they after k hops? Proposition 7.7 predicts monotone decay.

E4  What a subject can actually reach.
    The set of backbone variance pairs subject B can produce, under two
    different families of soft intervention, against the consensus target.

Every number is written to results/emp_numbers.tex and read by the report.
Run:  python experiments/empirical_study.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from core.abstraction import LinearAbstraction, push_gaussian        # noqa: E402
from core.kl import gaussian_kl, grid_kl                            # noqa: E402
from core.reporting import FIGURES_DIR, save                         # noqa: E402
from core.scm import LinearGaussianSCM                               # noqa: E402
from core.sheaf import (CausalNetwork,                               # noqa: E402
                       soft_intervention_matching_edge_variances)

SEED = 20261008
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
BLUE, RED, GREY = "#2166ac", "#b2182b", "#4d4d4d"


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ===========================================================================
# E1. Observational fit against interventional consistency
# ===========================================================================

def build_fitted_pair(b, w, noise_var):
    """Micro: H1 = Z1, H2 = Z2, Mk = b1 H1 + b2 H2 + Z3.
    Macro: T = Z_T, Mk = c T + Z_M, with the aggregation T = w1 H1 + w2 H2.

    The macro coefficient c and noise variance are always chosen by regression,
    so the macro model reproduces the OBSERVATIONAL macro law exactly. Whether
    it also reproduces the INTERVENTIONAL laws is what the experiment measures.
    """
    v1, v2, v3 = noise_var
    micro = LinearGaussianSCM(C=[[0, 0, 0], [0, 0, 0], [b[0], b[1], 0]],
                              noise_var=[v1, v2, v3], names=["H1", "H2", "Mk"])
    F = np.array([[w[0], w[1], 0.0], [0.0, 0.0, 1.0]])
    _, cov = push_gaussian(F, micro.law())
    c = cov[0, 1] / cov[0, 0]
    macro = LinearGaussianSCM(C=[[0, 0], [c, 0]],
                              noise_var=[cov[0, 0], cov[1, 1] - c * c * cov[0, 0]],
                              names=["T", "Mk"])
    return LinearAbstraction(micro, macro, F), c


def e1(rng, n_models=500):
    """For each random model: observational KL, IC gap, and the misalignment.

    Theory. With independent roots of variance v1, v2 the fitted coefficient is
        c = <b, w>_V / <w, w>_V ,     <x, y>_V = x1 y1 v1 + x2 y2 v2,
    so the residual r = b - c w is the V-orthogonal part of the effect vector b
    relative to the aggregation direction w. Writing rho = <r, r>_V, the exact
    IC gap for the macro intervention do(T = t) lifted to do(H = h) is
        KL = 0.5 [ (v3 + (r . h)^2) / (v3 + rho) - 1 + ln(1 + rho / v3) ] .
    Interventional consistency is r = 0, the consistency criterion.
    """
    rows = []
    for _ in range(n_models):
        b = rng.normal(size=2)
        # An aggregate variable has positive weights on its parts, so w is drawn
        # positive. Allowing near-zero weights would only inflate the gaps.
        w = rng.uniform(0.5, 1.5, size=2)
        noise_var = rng.uniform(0.3, 1.5, size=3)
        alpha, c = build_fitted_pair(b, w, noise_var)
        v1, v2, v3 = noise_var

        obs = alpha.ic_gap(alpha.micro, alpha.macro)
        # Two different micro lifts of the same macro intervention do(T = 2).
        t = 2.0
        lifts = [np.array([t / w[0], 0.0]), np.array([0.0, t / w[1]])]
        macro_hard = alpha.macro.do_hard({0: t})
        gaps = [alpha.ic_gap(alpha.micro.do_hard({0: h[0], 1: h[1]}), macro_hard, coords=[1])
                for h in lifts]

        r = b - c * w
        rho = float(r[0] ** 2 * v1 + r[1] ** 2 * v2)
        predicted = [0.5 * ((v3 + float(r @ h) ** 2) / (v3 + rho) - 1 + np.log1p(rho / v3))
                     for h in lifts]
        # The smallest gap any lift of do(T = t) can have is the one with r . h = 0.
        floor = 0.5 * (np.log1p(rho / v3) - rho / (v3 + rho))
        rows.append(dict(obs=obs, gap_a=gaps[0], gap_b=gaps[1],
                         pred_a=predicted[0], pred_b=predicted[1],
                         rho=rho, rho_rel=rho / v3, floor=floor,
                         disagreement=abs(gaps[0] - gaps[1])))

    obs = np.array([r["obs"] for r in rows])
    gap = np.array([max(r["gap_a"], r["gap_b"]) for r in rows])
    pred = np.array([max(r["pred_a"], r["pred_b"]) for r in rows])
    rho_rel = np.array([r["rho_rel"] for r in rows])
    floor = np.array([r["floor"] for r in rows])
    disagreement = np.array([r["disagreement"] for r in rows])

    out = {
        "n_models": n_models,
        "obs_max": float(obs.max()),
        "obs_median": float(np.median(obs)),
        "gap_median": float(np.median(gap)),
        "gap_max": float(gap.max()),
        "gap_q90": float(np.quantile(gap, 0.9)),
        "prediction_max_abs_error": float(np.max(np.abs(gap - pred))),
        "prediction_max_rel_error": float(np.max(np.abs(gap - pred) / np.maximum(gap, 1e-12))),
        "frac_fit_perfect_but_inconsistent": float(np.mean((obs < 1e-12) & (gap > 1e-2))),
        "frac_gap_above_tenth": float(np.mean(gap > 0.1)),
        "corr_obs_gap": float(np.corrcoef(obs, gap)[0, 1]),
        "corr_rho_gap": float(np.corrcoef(rho_rel, gap)[0, 1]),
        "two_lifts_disagree_median": float(np.median(disagreement)),
        "floor_violations": int(np.sum(gap < floor - 1e-12)),
    }
    figure_e1(obs, gap, rho_rel, floor)
    return out


def figure_e1(obs, gap, rho_rel, floor):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))

    eps = 1e-18
    axes[0].scatter(np.maximum(obs, eps), gap, s=7, color=BLUE, alpha=0.45, linewidths=0)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-19, 1e-2)
    axes[0].set_xlabel("observational KL (micro pushed vs macro)")
    axes[0].set_ylabel("IC gap under do($T$=2)")
    axes[0].set_title("(a) perfect fit, any amount of inconsistency")
    axes[0].axvline(1e-12, color=GREY, lw=0.8, ls=":")
    axes[0].text(1.6e-12, gap.max() * 0.93, "machine precision", fontsize=7, color=GREY)

    order = np.argsort(rho_rel)
    axes[1].scatter(rho_rel, gap, s=7, color=BLUE, alpha=0.45, linewidths=0, label="500 random models")
    axes[1].plot(rho_rel[order], floor[order], color=RED, lw=1.4,
                 label="predicted floor")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"misalignment $\rho/v_3$,  $\rho=\|b-cw\|_V^2$")
    axes[1].set_ylabel("IC gap under do($T$=2)")
    axes[1].set_title("(b) misalignment sets a floor it never breaks")
    axes[1].legend(frameon=False, loc="upper left")

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "emp_fit_vs_ic.pdf"))
    plt.close(fig)


# ===========================================================================
# E2. The blend gap as a function of lambda
# ===========================================================================

def gauss2(x, y, cov):
    inv = np.linalg.inv(cov)
    quad = inv[0, 0] * x * x + 2 * inv[0, 1] * x * y + inv[1, 1] * y * y
    return np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def kl_blend_to_nearest_gaussian(lam, a1, a2):
    """KL( blend || the closest Gaussian to it ), by grid integration.

    The Gaussian closest in KL to any distribution is the one with the same
    mean and covariance, so this is a certificate: the blend is at least this
    far from EVERY member of the linear Gaussian causal knowledge.
    """
    cov = lambda a: np.array([[1.0, a], [a, a * a + 1.0]])
    S1, S2 = cov(a1), cov(a2)
    if lam <= 0:
        return 0.0
    if lam >= 1:
        return 0.0
    span_y = 6.0 * np.sqrt(max(S1[1, 1], S2[1, 1]))
    xs = np.linspace(-7, 7, 601)
    ys = np.linspace(-span_y, span_y, 1401)
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    mix = lam * gauss2(X, Y, S1) + (1 - lam) * gauss2(X, Y, S2)
    moment_matched = gauss2(X, Y, lam * S1 + (1 - lam) * S2)
    return grid_kl(mix, moment_matched, cell)


def e2():
    lams = np.linspace(0, 1, 41)
    pairs = [(1.0, 2.0), (1.0, 3.0), (1.0, 4.0), (1.0, 6.0)]
    out = {"lambdas": lams.tolist(), "pairs": pairs, "variance_gap": {}, "kl": {}, "theory": {}}
    for a1, a2 in pairs:
        key = f"{a1}-{a2}"
        measured, theory, kls = [], [], []
        for lam in lams:
            var_measure = lam * (a1 ** 2 + 1) + (1 - lam) * (a2 ** 2 + 1)
            var_map = (lam * a1 + (1 - lam) * a2) ** 2 + 1
            measured.append(var_measure - var_map)
            theory.append(lam * (1 - lam) * (a1 - a2) ** 2)
            kls.append(kl_blend_to_nearest_gaussian(lam, a1, a2))
        out["variance_gap"][key] = measured
        out["theory"][key] = theory
        out["kl"][key] = kls
    out["max_variance_formula_error"] = float(max(
        max(abs(m - t) for m, t in zip(out["variance_gap"][k], out["theory"][k]))
        for k in out["variance_gap"]))
    # Consistency with the value quoted in Chapter 6 (lambda = 1/2, a = 1 and 4).
    idx = int(np.argmin(np.abs(lams - 0.5)))
    out["kl_at_half_a14"] = out["kl"]["1.0-4.0"][idx]
    out["kl_peak_a14"] = float(max(out["kl"]["1.0-4.0"]))
    out["lambda_at_peak_a14"] = float(lams[int(np.argmax(out["kl"]["1.0-4.0"]))])
    figure_e2(lams, pairs, out)
    return out


def figure_e2(lams, pairs, data):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    shades = ["#9db7d5", "#5d8cbd", "#2166ac", "#0d3b66"]

    for (a1, a2), colour in zip(pairs, shades):
        key = f"{a1}-{a2}"
        axes[0].plot(lams, data["variance_gap"][key], color=colour, lw=1.5,
                     label=rf"$a_1={a1:.0f},\ a_2={a2:.0f}$")
        axes[1].plot(lams, data["kl"][key], color=colour, lw=1.5)
    axes[0].plot(lams, data["theory"]["1.0-6.0"], color=RED, lw=1.0, ls="--",
                 label=r"theory $\lambda\bar\lambda(a_1-a_2)^2$")
    axes[0].set_xlabel(r"mixing weight $\lambda$")
    axes[0].set_ylabel(r"$\mathrm{Var}(Y)$ gap between the two blends")
    axes[0].set_title("(a) the two readings of blend")
    axes[0].legend(frameon=False, loc="upper center", ncol=1)

    axes[1].axhline(0, color=GREY, lw=0.6)
    axes[1].set_xlabel(r"mixing weight $\lambda$")
    axes[1].set_ylabel(r"KL(blend $\|$ nearest Gaussian)")
    axes[1].set_title("(b) distance from the causal knowledge")
    idx = int(np.argmin(np.abs(lams - 0.5)))
    axes[1].plot([0.5], [data["kl"]["1.0-4.0"][idx]], "o", ms=4, color=RED, zorder=5)
    axes[1].annotate(f"reported value\n{data['kl']['1.0-4.0'][idx]:.4f}",
                     xy=(0.5, data["kl"]["1.0-4.0"][idx]), xytext=(0.54, data["kl"]["1.0-4.0"][idx] * 0.42),
                     fontsize=7, color=RED,
                     arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    for ax in axes:
        despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "emp_blend_curve.pdf"))
    plt.close(fig)


# ===========================================================================
# E3. Knowledge decay along a chain of subjects
# ===========================================================================

def random_subject(dim, rng, tag):
    C = np.tril(rng.normal(scale=0.5, size=(dim, dim)), k=-1)
    return LinearGaussianSCM(C, rng.uniform(0.4, 1.4, size=dim), [f"{tag}{i}" for i in range(dim)])


def unit_rows(weights, scm):
    """Scale each restriction row so the subject's observational law projects to unit
    variance on every backbone coordinate."""
    W = np.atleast_2d(np.asarray(weights, dtype=float))
    _, S = scm.law()
    return W / np.sqrt(np.einsum("ij,jk,ik->i", W, S, W))[:, None]


def build_chain(length, width, rng):
    """A chain of `length` subjects joined by backbones of `width` variables."""
    net = CausalNetwork()
    subjects = []
    for k in range(length):
        dim = int(rng.integers(max(4, width + 1), 7))
        scm = random_subject(dim, rng, f"S{k}_")
        net.add_node(f"N{k}", scm)
        subjects.append(scm)
    for k in range(length - 1):
        backbone = LinearGaussianSCM(C=np.zeros((width, width)), noise_var=np.ones(width),
                                     names=[f"E{k}_{j}" for j in range(width)])
        net.add_edge(f"E{k}", backbone,
                     f"N{k}", unit_rows(rng.normal(size=(width, subjects[k].n)), subjects[k]),
                     f"N{k+1}", unit_rows(rng.normal(size=(width, subjects[k + 1].n)), subjects[k + 1]))
    return net


def e3(rng, length=6, n_chains=30, widths=(1, 2, 3)):
    """Two distinguishable messages leave subject 0. How distinguishable are
    they after each hop, and how much does the width of the shared backbone
    change the answer?

    Proposition 7.7: extension preserves KL exactly, restriction can only
    shrink it. So every curve must be monotone non-increasing, and all of the
    loss happens at the projection onto the backbone.
    """
    out = {"length": length, "n_chains": n_chains, "widths": list(widths), "curves": {}}
    monotone, preserve_err = True, 0.0
    for width in widths:
        rows = []
        for _ in range(n_chains):
            net = build_chain(length, width, rng)
            source = net.nodes["N0"]
            msg1 = source.law()
            msg2 = source.do_soft(0, gain=1.8).do_soft(source.n - 1, gain=0.6).law()
            kl = [gaussian_kl(*msg1, *msg2)]
            backbone_kl = []
            for hop in range(length - 1):
                src, edge, dst = f"N{hop}", f"E{hop}", f"N{hop+1}"
                r1, r2 = net.restrict(src, edge, msg1), net.restrict(src, edge, msg2)
                backbone_kl.append(gaussian_kl(*r1, *r2))
                msg1 = net.extend_kernel(dst, edge, r1)
                msg2 = net.extend_kernel(dst, edge, r2)
                kl.append(gaussian_kl(*msg1, *msg2))
            kl = np.array(kl)
            monotone = monotone and bool(np.all(np.diff(kl) <= 1e-9))
            preserve_err = max(preserve_err, float(np.max(np.abs(np.array(backbone_kl) - kl[1:]))))
            rows.append(kl / kl[0])
        rows = np.array(rows)
        out["curves"][str(width)] = {
            "median": np.median(rows, axis=0).tolist(),
            "q25": np.quantile(rows, 0.25, axis=0).tolist(),
            "q75": np.quantile(rows, 0.75, axis=0).tolist(),
            "per_hop_ratio_median": float(np.median(rows[:, 1:] / np.maximum(rows[:, :-1], 1e-300))),
            "retained_hop1": float(np.median(rows[:, 1])),
            "retained_final": float(np.median(rows[:, -1])),
        }
    out["monotone_everywhere"] = monotone
    out["extension_preserves_kl_max_error"] = preserve_err
    figure_e3(out)
    return out


def figure_e3(data):
    hops = np.arange(data["length"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    shades = {"1": "#b2182b", "2": "#2166ac", "3": "#1b7837"}

    for width in data["widths"]:
        key = str(width)
        med = np.array(data["curves"][key]["median"])
        colour = shades[key]
        axes[0].plot(hops, med, color=colour, lw=1.8, marker="o", ms=3.5,
                     label=f"backbone width {width}")
        axes[0].fill_between(hops, data["curves"][key]["q25"], data["curves"][key]["q75"],
                             color=colour, alpha=0.15, linewidth=0)
        axes[1].plot(hops, np.maximum(med, 1e-12), color=colour, lw=1.8, marker="o", ms=3.5,
                     label=f"width {width}")

    axes[0].set_xlabel("hops from the source subject")
    axes[0].set_ylabel("fraction of original KL retained")
    axes[0].set_title("(a) two messages blur together with distance")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(frameon=False)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("hops from the source subject")
    axes[1].set_ylabel("fraction retained (log scale)")
    axes[1].set_title("(b) geometric decay, never upward")
    axes[1].legend(frameon=False)

    for ax in axes:
        despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "emp_chain_decay.pdf"))
    plt.close(fig)


# ===========================================================================
# E4. What a subject can actually reach
# ===========================================================================

def build_middle_subject():
    """Subject B of Chapter 7, with its two restriction rows."""
    B = LinearGaussianSCM(C=[[0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0],
                             [0, 0, 0.6, 0, 0], [0, 0, 0.3, 0.5, 0]],
                          noise_var=[1.0, 0.8, 0.6, 0.7, 0.5],
                          names=["B1", "B2", "B3", "B4", "B5"])
    _, S = B.law()
    fx = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    fy = np.array([0.0, 0.0, 0.4, 0.3, 0.3])
    fx = fx / np.sqrt(fx @ S @ fx)
    fy = fy / np.sqrt(fy @ S @ fy)
    return B, np.vstack([fx, fy])


def e4(rng, n_samples=4000):
    """Which pairs (Var on X, Var on Y) can subject B produce?

    Family 1, noise gains only. The projected variance is linear in the squared
    gains u >= 0, so the reachable set is the convex cone spanned by the columns
    of A, where A[k, i] = (rows[k] M0)_i^2 * noise_var_i.

    Family 2, the full soft-intervention family: every existing coefficient may
    change and every gain may change, but no edge may be added or removed.
    Sampled.
    """
    B, rows = build_middle_subject()
    M0 = np.linalg.inv(np.eye(B.n) - B.C)
    A = (rows @ M0) ** 2 * B.noise_var                 # 2 x n, columns are the cone generators

    angles = np.arctan2(A[1], A[0])
    cone_lo, cone_hi = float(angles.min()), float(angles.max())

    edges = [(i, j) for i in range(B.n) for j in range(B.n) if B.C[i, j] != 0]
    pts = []
    for _ in range(n_samples):
        C = B.C.copy()
        for i, j in edges:
            C[i, j] = B.C[i, j] * np.exp(rng.normal(scale=0.9))     # sign and parent set preserved
        D = np.exp(rng.normal(scale=0.7, size=B.n))
        model = LinearGaussianSCM(C, B.noise_var, B.names, D=D)
        _, S = model.law()
        pts.append([float(rows[0] @ S @ rows[0]), float(rows[1] @ S @ rows[1])])
    pts = np.array(pts)

    target = np.array([1.6926315789473687, 0.4572192513368985])      # the Chapter 7 consensus target
    target_angle = float(np.arctan2(target[1], target[0]))
    in_cone = bool(cone_lo - 1e-12 <= target_angle <= cone_hi + 1e-12)

    # The exact solution found by the Chapter 7 search, to show the full family
    # does not merely come close but reaches the target.
    solved = soft_intervention_matching_edge_variances(B, rows, target)
    _, Ssol = solved.law()
    solved_point = np.array([float(rows[0] @ Ssol @ rows[0]), float(rows[1] @ Ssol @ rows[1])])

    dist = np.linalg.norm(pts - target, axis=1)
    out = {
        "cone_generators": A.T.tolist(),
        "cone_angle_range_deg": [np.degrees(cone_lo), np.degrees(cone_hi)],
        "target": target.tolist(),
        "target_angle_deg": float(np.degrees(target_angle)),
        "target_in_gain_cone": in_cone,
        "nearest_sampled_distance": float(dist.min()),
        "n_samples": n_samples,
        "frac_samples_outside_cone": float(np.mean(
            (np.arctan2(pts[:, 1], pts[:, 0]) < cone_lo - 1e-12) |
            (np.arctan2(pts[:, 1], pts[:, 0]) > cone_hi + 1e-12))),
        "observational_point": [float(rows[0] @ B.law()[1] @ rows[0]),
                                float(rows[1] @ B.law()[1] @ rows[1])],
        "exact_solution_point": solved_point.tolist(),
        "exact_solution_error": float(np.linalg.norm(solved_point - target)),
    }
    figure_e4(A, pts, target, solved_point, cone_lo, cone_hi)
    return out


def figure_e4(A, pts, target, solved, cone_lo, cone_hi):
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    xlim, ylim = 3.0, 2.0

    t = np.linspace(cone_lo, cone_hi, 60)
    far = 8.0
    ax.fill(np.concatenate([[0], far * np.cos(t), [0]]),
            np.concatenate([[0], far * np.sin(t), [0]]),
            color=BLUE, alpha=0.16, linewidth=0, zorder=1)
    ax.plot([0, far * np.cos(cone_lo)], [0, far * np.sin(cone_lo)], color=BLUE, lw=1.2, zorder=3)
    ax.plot([0, far * np.cos(cone_hi)], [0, far * np.sin(cone_hi)], color=BLUE, lw=1.2, zorder=3)

    ax.scatter(pts[:, 0], pts[:, 1], s=3.5, color="#3f3f3f", alpha=0.35, linewidths=0, zorder=2)
    ax.plot(*target, "*", ms=15, color=RED, zorder=6, mec="white", mew=0.6)
    ax.plot(*solved, "o", ms=9, mfc="none", mec=RED, mew=1.4, zorder=6)

    ax.annotate("consensus target:\nno choice of gains\nreaches it",
                xy=(target[0] - 0.06, target[1] - 0.04), xytext=(0.72, 0.14),
                fontsize=7.5, color=RED, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.text(0.55, 1.62, "reachable with\nnoise gains only", fontsize=8, color="#13457a")
    ax.text(2.12, 0.26, "reachable with the\nfull soft family", fontsize=8, color="#3f3f3f")
    ax.text(2.12, 0.08, "(open circle: exact solution)", fontsize=6.8, color=RED)

    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("variance projected onto backbone $X$")
    ax.set_ylabel("variance projected onto backbone $Y$")
    ax.set_title("What subject $B$ can actually reach", fontsize=9.5)
    despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "emp_reachability.pdf"))
    plt.close(fig)


# ===========================================================================

def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED}
    results["e1"] = e1(rng)
    results["e2"] = e2()
    results["e3"] = e3(rng)
    results["e4"] = e4(rng)
    a, b, c, d = results["e1"], results["e2"], results["e3"], results["e4"]

    macros = {
        "EmpNModels": str(a["n_models"]),
        "EmpObsMax": a["obs_max"],
        "EmpGapMedian": a["gap_median"],
        "EmpGapMax": a["gap_max"],
        "EmpGapQninety": a["gap_q90"],
        "EmpPredErr": a["prediction_max_abs_error"],
        "EmpFracInconsistent": f"{100 * a['frac_fit_perfect_but_inconsistent']:.1f}",
        "EmpFracAboveTenth": f"{100 * a['frac_gap_above_tenth']:.1f}",
        "EmpCorrRho": f"{a['corr_rho_gap']:.3f}",
        "EmpCorrObsGap": f"{a['corr_obs_gap']:.3f}",
        "EmpLiftDisagree": a["two_lifts_disagree_median"],
        "EmpFloorViolations": str(a["floor_violations"]),

        "EmpVarFormulaErr": b["max_variance_formula_error"],
        "EmpKLatHalf": b["kl_at_half_a14"],
        "EmpKLPeak": b["kl_peak_a14"],
        "EmpLambdaPeak": f"{b['lambda_at_peak_a14']:.3f}",

        "EmpChains": str(c["n_chains"]),
        "EmpChainLen": str(c["length"]),
        "EmpRetainOneA": f"{100 * c['curves']['1']['retained_hop1']:.1f}",
        "EmpRetainOneB": f"{100 * c['curves']['2']['retained_hop1']:.1f}",
        "EmpRetainOneC": f"{100 * c['curves']['3']['retained_hop1']:.1f}",
        "EmpRetainFinalA": f"{100 * c['curves']['1']['retained_final']:.3f}",
        "EmpRetainFinalB": f"{100 * c['curves']['2']['retained_final']:.2f}",
        "EmpRetainFinalC": f"{100 * c['curves']['3']['retained_final']:.1f}",
        "EmpExtPreserve": c["extension_preserves_kl_max_error"],
        "EmpMonotone": "yes" if c["monotone_everywhere"] else "no",

        "EmpConeLo": f"{d['cone_angle_range_deg'][0]:.1f}",
        "EmpConeHi": f"{d['cone_angle_range_deg'][1]:.1f}",
        "EmpTargetAngle": f"{d['target_angle_deg']:.1f}",
        "EmpTargetInCone": "yes" if d["target_in_gain_cone"] else "no",
        "EmpNearestSample": d["nearest_sampled_distance"],
        "EmpExactSolErr": d["exact_solution_error"],
        "EmpSamples": str(d["n_samples"]),
    }
    save("emp", results, macros)
    return results


if __name__ == "__main__":
    r = main()
    a, b, c, d = r["e1"], r["e2"], r["e3"], r["e4"]
    print("E1  observational KL max      ", f"{a['obs_max']:.2e}")
    print("E1  IC gap median / max       ", f"{a['gap_median']:.4f} / {a['gap_max']:.4f}")
    print("E1  perfect fit, inconsistent ", f"{100*a['frac_fit_perfect_but_inconsistent']:.1f}%")
    print("E1  corr(obs KL, gap)         ", f"{a['corr_obs_gap']:.3f}")
    print("E1  corr(misalignment, gap)   ", f"{a['corr_rho_gap']:.3f}")
    print("E1  closed form max error     ", f"{a['prediction_max_abs_error']:.2e}")
    print("E1  floor violations          ", a["floor_violations"])
    print("E2  variance formula max err  ", f"{b['max_variance_formula_error']:.2e}")
    print("E2  KL at lambda=0.5, a=1,4   ", f"{b['kl_at_half_a14']:.4f}  (Ch.6 reports 0.2453)")
    print("E2  peak KL / at lambda       ", f"{b['kl_peak_a14']:.4f} / {b['lambda_at_peak_a14']:.3f}")
    print("E3  monotone in every chain   ", c["monotone_everywhere"])
    print("E3  extension preserves KL    ", f"{c['extension_preserves_kl_max_error']:.2e}")
    for wd in c["widths"]:
        cu = c["curves"][str(wd)]
        print(f"E3  width {wd}: hop1 / final      ",
              f"{100*cu['retained_hop1']:.1f}% / {100*cu['retained_final']:.3f}%")
    print("E4  gain cone angles          ", f"{d['cone_angle_range_deg'][0]:.1f} to "
          f"{d['cone_angle_range_deg'][1]:.1f} deg")
    print("E4  target angle / in cone    ", f"{d['target_angle_deg']:.1f} deg / {d['target_in_gain_cone']}")
    print("E4  nearest sampled to target ", f"{d['nearest_sampled_distance']:.4f}")
    print("E4  exact solution error      ", f"{d['exact_solution_error']:.2e}")
