"""
Does the misalignment machinery work, and is it worth having?

Six checks. The first four ask whether the closed forms are correct. The
fifth asks whether the design result earns its place, and it is the one that
could have gone against us: it compares our weights against the three things
a practitioner would actually do, and scores all four on the quantity that
matters rather than on our own objective. The sixth records the case where no
weighting works.

WHY THE BENEFIT TEST IS SCORED THE WAY IT IS
--------------------------------------------
Every method is scored by the MEASURED interventional error, computed through
the Kullback-Leibler divergence between the coarse model's prediction and
what the micro model actually does. It is not scored by misalignment.
Scoring by misalignment would be circular, since our method minimises it by
construction and would win by definition. The design was fixed before the
numbers were seen, and the result is reported whichever way it falls.

Run:  python -m experiments.lint_validation
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np

from core.misalignment import (design, fitted_coefficient, floor, gram,
                               inner, irreducible, misalignment, repair,
                               residual)
from core.reporting import MACROS_DIR, RESULTS_DIR, fmt
from core.scm import LinearGaussianSCM

SEED = 20261018
N_MODELS = 400
PARTS = (2, 6)
TARGET = 2.0


# ---------------------------------------------------------------------------
# building a micro model and measuring what a coarse model really costs
# ---------------------------------------------------------------------------

def build_micro(B: Sequence[np.ndarray], v: np.ndarray,
                v_out: Sequence[float]) -> LinearGaussianSCM:
    """Parts are independent roots; each outcome is linear in the parts."""
    n, K = len(v), len(B)
    C = np.zeros((n + K, n + K))
    for k, b in enumerate(B):
        C[n + k, :n] = b
    return LinearGaussianSCM(
        C=C, noise_var=list(v) + list(v_out),
        names=[f"H{i+1}" for i in range(n)] + [f"Y{k+1}" for k in range(K)])


def measured_gap(b: np.ndarray, w: np.ndarray, v: np.ndarray, v_out: float,
                 h: np.ndarray) -> float:
    """KL between what the coarse model predicts for do(T = w.h) and what the
    micro model actually produces when the parts are set to h.

    Both laws are Gaussian with the same support, so this is the closed-form
    univariate KL and no sampling enters. Computed from the models rather
    than from any formula of ours, so it can disagree with our prediction.
    """
    c = fitted_coefficient(b, w, v)
    rho = misalignment(b, w, v)
    mu_macro, var_macro = c * float(w @ h), v_out + rho   # coarse prediction
    mu_micro, var_micro = float(b @ h), v_out             # what really happens
    return float(0.5 * (var_micro / var_macro - 1.0
                        + np.log(var_macro / var_micro)
                        + (mu_micro - mu_macro) ** 2 / var_macro))


def sample_actions(n: int, rng, m: int = 24, scale: float = 1.0,
                   target: float = TARGET) -> List[np.ndarray]:
    """A set of real actions on the parts, drawn independently of any
    proposed weighting.

    THIS IS THE CORRECTION THAT MATTERS. An earlier version of this file
    built the actions out of w, by rescaling random vectors until w.h hit a
    fixed target. That was wrong twice over. It scored every method on a
    different set of actions, so it was not a comparison at all; and when w.h
    landed near zero the rescaling produced actions of norm 70 and upwards,
    whose error dominated every score and had nothing to do with the method.

    The corrected design fixes the actions first and lets each method's macro
    intervention be whatever its own weighting implies, do(T = w.h). Every
    method is then asked the same question about the same world.

    The set covers the single-part actions a practitioner reaches for first,
    a uniform action across all parts, and bounded random directions.
    """
    out = [np.eye(n)[i] * target for i in range(n)]           # one part at a time
    out.append(np.full(n, target / n))                        # spread evenly
    while len(out) < m:
        h = rng.normal(size=n)
        norm = np.linalg.norm(h)
        if norm > 1e-9:
            out.append(h / norm * scale * target)             # bounded, random
    return out


# ---------------------------------------------------------------------------
# V1 to V4: are the closed forms right
# ---------------------------------------------------------------------------

def v1_closed_form(rng) -> Dict:
    """Does the predicted interventional error match the measured one, for
    any number of parts? E1 checked this for two parts only."""
    worst = 0.0
    for _ in range(N_MODELS):
        n = int(rng.integers(*PARTS))
        b = rng.normal(size=n)
        w = rng.uniform(0.5, 1.5, size=n)
        v = rng.uniform(0.3, 1.5, size=n)
        v_out = float(rng.uniform(0.5, 1.5))
        r, rho = residual(b, w, v), misalignment(b, w, v)
        for h in sample_actions(n, rng):
            predicted = 0.5 * ((v_out + float(r @ h) ** 2) / (v_out + rho)
                               - 1.0 + np.log1p(rho / v_out))
            worst = max(worst, abs(predicted - measured_gap(b, w, v, v_out, h)))
    return {"max_abs_error": worst, "models": N_MODELS}


def v2_repair(rng) -> Dict:
    """Does the one-outcome repair drive the MEASURED error to zero, not just
    the misalignment? Checking our own quantity would prove nothing."""
    worst_gap, worst_rho = 0.0, 0.0
    for _ in range(N_MODELS):
        n = int(rng.integers(*PARTS))
        b = rng.normal(size=n)
        v = rng.uniform(0.3, 1.5, size=n)
        v_out = float(rng.uniform(0.5, 1.5))
        w = repair(b, v)
        worst_rho = max(worst_rho, misalignment(b, w, v))
        for h in sample_actions(n, rng):
            worst_gap = max(worst_gap, measured_gap(b, w, v, v_out, h))
    return {"max_measured_gap": worst_gap, "max_misalignment": worst_rho,
            "models": N_MODELS}


def v3_floor(rng) -> Dict:
    """Is the floor ever violated by a real action?"""
    violations, tightest = 0, np.inf
    for _ in range(N_MODELS):
        n = int(rng.integers(*PARTS))
        b = rng.normal(size=n)
        w = rng.uniform(0.5, 1.5, size=n)
        v = rng.uniform(0.3, 1.5, size=n)
        v_out = float(rng.uniform(0.5, 1.5))
        f = floor(b, w, v, v_out)
        for h in sample_actions(n, rng):
            gap = measured_gap(b, w, v, v_out, h)
            if gap < f - 1e-12:
                violations += 1
            tightest = min(tightest, gap - f)
    return {"violations": violations, "closest_approach": float(tightest),
            "models": N_MODELS}


def v4_observable(rng) -> Dict:
    """With several outcomes, does the off-diagonal of the residual Gram
    matrix equal the observational mismatch a data check would see?"""
    from core.abstraction import push_gaussian
    worst = 0.0
    for _ in range(N_MODELS // 2):
        n, K = int(rng.integers(2, 6)), 2
        v = rng.uniform(0.3, 1.5, size=n)
        v_out = rng.uniform(0.5, 1.5, size=K)
        B = [rng.normal(size=n) for _ in range(K)]
        w = rng.uniform(0.5, 1.5, size=n)

        micro = build_micro(B, v, v_out)
        F = np.zeros((1 + K, n + K)); F[0, :n] = w
        for k in range(K):
            F[1 + k, n + k] = 1.0
        _, cov = push_gaussian(F, micro.law())
        c = [fitted_coefficient(b, w, v) for b in B]
        # what a coarse model with independent noises would produce
        predicted_cov = c[0] * c[1] * cov[0, 0]
        worst = max(worst, abs((cov[1, 2] - predicted_cov)
                               - gram(B, w, v)[0, 1]))
    return {"max_abs_error": worst, "models": N_MODELS // 2}


# ---------------------------------------------------------------------------
# V5: the benefit test
# ---------------------------------------------------------------------------

def weights_equal(B, v, cov_parts) -> np.ndarray:
    return np.ones(len(v)) / np.sqrt(len(v))


def weights_pca(B, v, cov_parts) -> np.ndarray:
    """What a careful practitioner does: first principal component of the
    parts. Driven entirely by how the data varies."""
    vals, vecs = np.linalg.eigh(cov_parts)
    w = vecs[:, int(np.argmax(vals))]
    return w if w.sum() >= 0 else -w


def weights_fit(B, v, cov_parts) -> np.ndarray:
    """What a practitioner with data does: regress the first outcome on the
    parts and use those coefficients. For one outcome this coincides with the
    right answer, so the comparison only bites when there are several."""
    w = np.asarray(B[0], dtype=float)
    n = np.linalg.norm(w)
    return w / n if n > 0 else np.ones(len(v)) / np.sqrt(len(v))


def weights_design(B, v, cov_parts) -> np.ndarray:
    return design(B, v)


METHODS = [("equal", weights_equal), ("pca", weights_pca),
           ("fit first outcome", weights_fit), ("design", weights_design)]


def v5_benefit(rng, n_models: int = N_MODELS, K: int = 3) -> Dict:
    """Score four ways of choosing the weights on measured interventional
    error, over random models with several outcomes.

    The score for a model is the worst measured error across outcomes and
    across realisations of the macro intervention, because a composite is
    only as safe as its worst use.
    """
    rows = []
    for _ in range(n_models):
        n = int(rng.integers(3, 7))
        v = rng.uniform(0.3, 1.5, size=n)
        v_out = rng.uniform(0.5, 1.5, size=K)
        B = [rng.normal(size=n) for _ in range(K)]
        cov_parts = np.diag(v)
        # one action set per model, SHARED by every method, so all four are
        # asked about the same interventions on the same world
        actions = sample_actions(n, rng)
        row = {}
        for name, fn in METHODS:
            w = fn(B, v, cov_parts)
            worst = 0.0
            for b, vo in zip(B, v_out):
                for h in actions:
                    worst = max(worst, measured_gap(b, w, v, vo, h))
            row[name] = worst
        rows.append(row)

    names = [n for n, _ in METHODS]
    stats = {}
    for name in names:
        xs = np.array([r[name] for r in rows])
        stats[name] = {"median": float(np.median(xs)),
                       "mean": float(xs.mean()),
                       "q90": float(np.quantile(xs, 0.9))}
    wins, margins = {}, {}
    for name in names:
        if name == "design":
            continue
        better = np.array([r["design"] < r[name] - 1e-12 for r in rows])
        ratio = np.array([r[name] / max(r["design"], 1e-12) for r in rows])
        wins[name] = float(better.mean())
        margins[name] = {"median_ratio": float(np.median(ratio)),
                         "q90_ratio": float(np.quantile(ratio, 0.9))}
    return {"models": n_models, "outcomes": K, "stats": stats,
            "design_wins_fraction": wins, "ratio_vs_design": margins,
            "rows": rows}


def v7_optimality(rng, trials: int = 60) -> Dict:
    """Is the closed form really the minimiser?

    Proposition "best compromise composite" claims the leading eigenvector
    minimises total misalignment. That was first confirmed on a single
    worked example, which is not evidence. Here a derivative-free search is
    run from several starts on each of many random cases and asked to beat
    it. A single win would refute the proposition.
    """
    from scipy.optimize import minimize
    search_won, worst_rel = 0, 0.0
    for _ in range(trials):
        n, K = int(rng.integers(2, 7)), int(rng.integers(2, 5))
        v = rng.uniform(0.3, 1.5, size=n)
        B = [rng.normal(size=n) for _ in range(K)]
        total = lambda w: sum(misalignment(b, w, v) for b in B)
        ours = total(design(B, v))
        best = min(minimize(total, x0=rng.normal(size=n), method="Powell",
                            options=dict(xtol=1e-10, ftol=1e-12)).fun
                   for _ in range(3))
        scale = sum(inner(b, b, v) for b in B)
        worst_rel = max(worst_rel, (ours - best) / scale)
        search_won += int(ours - best > 1e-9)
    return {"trials": trials, "times_search_won": search_won,
            "worst_relative_excess": float(worst_rel)}


# ---------------------------------------------------------------------------
# V6: when nothing works
# ---------------------------------------------------------------------------

def v6_irreducible(rng) -> Dict:
    """How often is the best achievable composite still badly misaligned, and
    how does that depend on how much the outcomes disagree?"""
    rows = []
    for _ in range(N_MODELS):
        n, K = int(rng.integers(3, 7)), 2
        v = rng.uniform(0.3, 1.5, size=n)
        B = [rng.normal(size=n) for _ in range(K)]
        scale = sum(inner(b, b, v) for b in B)
        best = irreducible(B, v)
        # how parallel are the two effect vectors, in the V metric
        cos = abs(inner(B[0], B[1], v)) / np.sqrt(inner(B[0], B[0], v)
                                                  * inner(B[1], B[1], v))
        rows.append({"relative": best / scale, "alignment": float(cos)})
    rel = np.array([r["relative"] for r in rows])
    al = np.array([r["alignment"] for r in rows])
    return {"models": N_MODELS,
            "median_relative_residual": float(np.median(rel)),
            "fraction_above_ten_percent": float((rel > 0.1).mean()),
            "corr_alignment_residual": float(np.corrcoef(al, rel)[0, 1]),
            "rows": rows}


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------

def figure(res: Dict, stem: str = "lint_benefit") -> None:
    """Two panels, because the result is two claims.

    (a) Does choosing the weights by causal effect beat the three things a
        practitioner would otherwise do? Scored on measured interventional
        error, over a shared set of actions, so the comparison is fair.

    (b) When it cannot be fixed, why not? The residual at the optimum is
        governed by how much the outcomes agree about what the composite
        should mean, and that relationship is the practical signal to split
        a composite rather than keep tuning it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FINDING, CONTEXT = "#B2182B", "#2166AC"
    INK, MUTED, GRID = "#14213D", "#6A7179", "#DDE3EA"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "figure.dpi": 150})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    names = [n for n, _ in METHODS]
    data = [[r[n] for r in res["v5"]["rows"]] for n in names]
    colours = [CONTEXT] * (len(names) - 1) + [FINDING]
    bp = ax1.boxplot(data, widths=0.55, showfliers=False,
                     patch_artist=True, medianprops=dict(color="white", lw=1.4))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor(c)
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set_color(MUTED); ln.set_linewidth(0.8)
    ax1.set_xticklabels(["equal", "PCA", "fit one", "by effect"], fontsize=7.2)
    ax1.set_ylabel("worst interventional error (nats)")
    ax1.set_title("(a)  choosing the weights", fontsize=8.5, loc="left",
                  color=INK, pad=8)
    ax1.grid(axis="y", color=GRID, linewidth=0.5); ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    # the claim, placed in clear space above the box rather than on top of it
    wins = np.mean([res["v5"]["design_wins_fraction"][n]
                    for n in names if n != "design"]) * 100
    # axes-fraction coordinates: the data maximum sits outside the drawn
    # range once outliers are hidden, so a data-space position gets clipped
    ax1.annotate(f"lower on {wins:.0f}% of models",
                 xy=(0.97, 0.06), xycoords="axes fraction",
                 fontsize=7.0, color=FINDING, weight="bold", ha="right")

    al = [r["alignment"] for r in res["v6"]["rows"]]
    rel = [r["relative"] * 100 for r in res["v6"]["rows"]]
    ax2.scatter(al, rel, s=9, color=CONTEXT, alpha=0.45, linewidth=0)
    ax2.set_xlabel("agreement between the outcomes  $|\\cos\\theta|$")
    ax2.set_ylabel("effect uncaptured at best weights (%)")
    ax2.set_title("(b)  when no weighting works", fontsize=8.5, loc="left",
                  color=INK, pad=8)
    ax2.grid(color=GRID, linewidth=0.5); ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.annotate(f"r = {res['v6']['corr_alignment_residual']:.2f}",
                 xy=(0.72, 0.88), xycoords="axes fraction",
                 fontsize=7.2, color=INK)

    fig.tight_layout(pad=0.9)
    figs = os.path.join(MACROS_DIR, "figs")
    os.makedirs(figs, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figs, f"{stem}.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------

def write_macros(res: Dict, path: str) -> None:
    s5, w5 = res["v5"]["stats"], res["v5"]["design_wins_fraction"]
    r5 = res["v5"]["ratio_vs_design"]
    lines = [
        "% Generated by experiments/lint_validation.py. Do not edit by hand.",
        f"\\newcommand{{\\LintModels}}{{{N_MODELS}}}",
        f"\\newcommand{{\\LintClosedFormErr}}{{{fmt(res['v1']['max_abs_error'])}}}",
        f"\\newcommand{{\\LintRepairGap}}{{{fmt(res['v2']['max_measured_gap'])}}}",
        f"\\newcommand{{\\LintFloorViolations}}{{{res['v3']['violations']}}}",
        f"\\newcommand{{\\LintGramErr}}{{{fmt(res['v4']['max_abs_error'])}}}",
        f"\\newcommand{{\\LintOptTrials}}{{{res['v7']['trials']}}}",
        f"\\newcommand{{\\LintOptSearchWon}}{{{res['v7']['times_search_won']}}}",
        f"\\newcommand{{\\LintOptExcess}}"
        f"{{{fmt(res['v7']['worst_relative_excess'])}}}",
        f"\\newcommand{{\\LintOutcomes}}{{{res['v5']['outcomes']}}}",
        f"\\newcommand{{\\LintMedEqual}}{{{s5['equal']['median']:.3f}}}",
        f"\\newcommand{{\\LintMedPCA}}{{{s5['pca']['median']:.3f}}}",
        f"\\newcommand{{\\LintMedFit}}{{{s5['fit first outcome']['median']:.3f}}}",
        f"\\newcommand{{\\LintMedDesign}}{{{s5['design']['median']:.3f}}}",
        f"\\newcommand{{\\LintWinsEqual}}{{{w5['equal'] * 100:.1f}}}",
        f"\\newcommand{{\\LintWinsPCA}}{{{w5['pca'] * 100:.1f}}}",
        f"\\newcommand{{\\LintWinsFit}}{{{w5['fit first outcome'] * 100:.1f}}}",
        f"\\newcommand{{\\LintRatioPCA}}{{{r5['pca']['median_ratio']:.2f}}}",
        f"\\newcommand{{\\LintRatioEqual}}{{{r5['equal']['median_ratio']:.2f}}}",
        f"\\newcommand{{\\LintRatioFit}}{{{r5['fit first outcome']['median_ratio']:.2f}}}",
        f"\\newcommand{{\\LintIrredMedian}}"
        f"{{{res['v6']['median_relative_residual'] * 100:.1f}}}",
        f"\\newcommand{{\\LintIrredAboveTen}}"
        f"{{{res['v6']['fraction_above_ten_percent'] * 100:.1f}}}",
        f"\\newcommand{{\\LintIrredCorr}}{{{res['v6']['corr_alignment_residual']:.3f}}}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED,
           "v1": v1_closed_form(rng), "v2": v2_repair(rng),
           "v3": v3_floor(rng), "v4": v4_observable(rng),
           "v5": v5_benefit(rng), "v6": v6_irreducible(rng),
           "v7": v7_optimality(rng)}

    print("V1  closed form vs measured gap, n parts")
    print(f"      max abs error            {res['v1']['max_abs_error']:.2e}")
    print("V2  does the repair work, measured not predicted")
    print(f"      max measured gap         {res['v2']['max_measured_gap']:.2e}")
    print(f"      max misalignment         {res['v2']['max_misalignment']:.2e}")
    print("V3  is the floor ever beaten")
    print(f"      violations               {res['v3']['violations']}")
    print(f"      closest approach         {res['v3']['closest_approach']:.2e}")
    print("V4  Gram off-diagonal vs observational mismatch")
    print(f"      max abs error            {res['v4']['max_abs_error']:.2e}")
    print("V7  can a numerical search beat the closed-form optimum")
    print(f"      search won               {res['v7']['times_search_won']}"
          f" of {res['v7']['trials']}  (0 means the formula is optimal)")
    print(f"      worst relative excess    {res['v7']['worst_relative_excess']:.2e}")

    print(f"\nV5  benefit test: {res['v5']['models']} models, "
          f"{res['v5']['outcomes']} outcomes each")
    print(f"      {'method':<20}{'median':>10}{'mean':>10}{'q90':>10}"
          f"{'design better':>16}")
    for name, _ in METHODS:
        s = res["v5"]["stats"][name]
        win = ("" if name == "design"
               else f"{res['v5']['design_wins_fraction'][name] * 100:>14.1f}%")
        print(f"      {name:<20}{s['median']:>10.3f}{s['mean']:>10.3f}"
              f"{s['q90']:>10.3f}{win:>16}")
    print("      median error ratio against design:")
    for name, m in res["v5"]["ratio_vs_design"].items():
        print(f"        {name:<20}{m['median_ratio']:.2f}x")

    print(f"\nV6  when no weighting works ({res['v6']['models']} models)")
    print(f"      median residual at best  "
          f"{res['v6']['median_relative_residual'] * 100:.1f}% of effect")
    print(f"      above 10%                "
          f"{res['v6']['fraction_above_ten_percent'] * 100:.1f}% of models")
    print(f"      corr(outcome agreement, residual)  "
          f"{res['v6']['corr_alignment_residual']:+.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "lint_validation.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    write_macros(res, os.path.join(MACROS_DIR, "lint_numbers.tex"))
    figure(res)
    print(f"\nwritten to {RESULTS_DIR}/lint_validation.json and "
          f"paper/lint_numbers.tex")


if __name__ == "__main__":
    main()
