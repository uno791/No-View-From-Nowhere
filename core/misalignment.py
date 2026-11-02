"""
Is a composite variable safe to act on?

THE QUESTION
------------
You have a fine-grained model in which several parts each cause an outcome.
You decide the detail is excessive and define a composite, a weighted sum of
the parts, and you fit a coarse model on top of it. The coarse model will
reproduce your observed data exactly, because regression fits it. The
question this module answers is the other one: will the coarse model predict
correctly when you ACT?

Experiment E1 established that the first question tells you nothing about the
second. Over 500 random aggregations the observational error never exceeded
5e-16 while the interventional error ran to 9.48, and the correlation between
them was -0.069. So the check has to be structural. It cannot be done by
looking at data, and Section "why not just fit it" below says why.

THE ANSWER
----------
Write b for the vector of causal effects of the parts on the outcome, w for
the weights you chose, and <x,y> = sum_i x_i y_i v_i for the inner product
weighted by how much each part actually varies. The coefficient regression
finds is the projection of b onto w,

    c = <b,w> / <w,w>,

so the part of the true effect your composite cannot see is the residual

    r = b - c w,        misalignment = <r,r>.

Interventional consistency holds exactly when r = 0, that is when b and w are
parallel: *a composite is safe to act on exactly when its parts are weighted
by how much they actually cause the outcome.*

A useful identity falls out. Since <r,r> = <b,b> - <b,w>^2/<w,w>,

    misalignment / <b,b> = 1 - cos^2(theta) = sin^2(theta)

where theta is the angle between b and w in this metric. So the scale-free
form of the quantity is the fraction of causal effect the composite fails to
capture, which lives in [0,1] and can carry a threshold. `captured` below
reports its complement, which is the number to quote to a person.

WHY NOT JUST FIT THE WEIGHTS
----------------------------
Because the observational loss surface is exactly flat. Every choice of w
reproduces the observed joint law of (composite, outcome) to machine
precision, so there is no gradient and nothing to descend. What breaks the
tie is b, which lives in the model and is never revealed by the data.

ONE OUTCOME, OR SEVERAL
-----------------------
With one outcome the repair is immediate: set w proportional to b and the
misalignment is zero. `repair` does that.

With several outcomes there is in general no w that satisfies all of them,
because the b_k are not parallel. Minimising the total misalignment is then a
real optimisation, and it has a closed form: maximising

    sum_k <b_k, w>^2 / <w,w>

is a Rayleigh quotient, so the optimum is the leading eigenvector of
V^{1/2} (sum_k b_k b_k^T) V^{1/2}, pulled back through V^{-1/2}. `design`
does that. The residual at the optimum is irreducible, and reporting it is
the point: it is the unavoidable cost of serving several purposes with one
composite, and when it is large the honest advice is to split the composite
rather than to reweight it.

That last construction is standard linear algebra, of the kind that appears
in reduced-rank regression. What is specific here is reading it as the
solution to the interventional consistency problem for a causal abstraction,
and the impossibility statement that comes with it.

NOT PCA
-------
The leading eigenvector here is of a matrix built from causal EFFECTS, not
from the data covariance. Ordinary dimensionality reduction finds the
direction in which the parts vary most; this finds the direction that best
preserves causal effect. They are different directions, and by E1 the two
criteria are uncorrelated, so a composite built by PCA, by averaging, or by
convenient units is optimising for the wrong thing if the intention is to act
on it.

SCOPE
-----
Linear Gaussian, parts that are mutually independent roots, outcomes that are
linear in the parts. The projection argument is linear throughout and does
not generalise to nonlinear mechanisms as it stands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Below this fraction of effect left uncaptured, a composite is called safe.
# It is a reporting threshold, not a mathematical boundary: the underlying
# quantity is continuous and the floor it implies is reported alongside.
SAFE_FRACTION = 1e-6


# ---------------------------------------------------------------------------
# the weighted inner product everything is built from
# ---------------------------------------------------------------------------

def inner(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> float:
    """<x,y> = sum_i x_i y_i v_i.

    The variances appear because what matters is how much each part actually
    moves. A part with a large coefficient and no variation contributes
    nothing to the outcome and should not pull the composite toward itself.
    """
    return float(np.sum(np.asarray(x) * np.asarray(y) * np.asarray(v)))


def fitted_coefficient(b: np.ndarray, w: np.ndarray, v: np.ndarray) -> float:
    """The macro coefficient regression will find, in closed form.

    Computing it rather than regressing is deliberate: it makes explicit that
    the coefficient is determined by the projection and not by the data.
    """
    return inner(b, w, v) / inner(w, w, v)


def residual(b: np.ndarray, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The part of the effect vector the composite cannot see.

    Returned as a vector rather than a scalar so a caller can attribute the
    failure to particular parts. A single number says a composite is unsafe;
    this says which variable is making it unsafe.
    """
    b = np.asarray(b, dtype=float)
    return b - fitted_coefficient(b, w, v) * np.asarray(w, dtype=float)


def misalignment(b: np.ndarray, w: np.ndarray, v: np.ndarray) -> float:
    """<r,r>, the absolute misalignment. Feeds the floor formula."""
    r = residual(b, w, v)
    return inner(r, r, v)


def captured(b: np.ndarray, w: np.ndarray, v: np.ndarray) -> float:
    """Fraction of causal effect the composite does capture, in [0,1].

    Equals cos^2 of the angle between b and w. This is the scale-free form
    and the one to quote: "the composite captures 76% of the causal effect"
    is a sentence a reader can act on, where "misalignment 0.405" is not.
    """
    bb = inner(b, b, v)
    if bb <= 0.0:
        return 1.0                      # no effect to capture; vacuously fine
    return float(np.clip(1.0 - misalignment(b, w, v) / bb, 0.0, 1.0))


def angle_degrees(b: np.ndarray, w: np.ndarray, v: np.ndarray) -> float:
    """The misalignment as an angle, which is the most readable form."""
    return float(np.degrees(np.arccos(np.sqrt(captured(b, w, v)))))


# ---------------------------------------------------------------------------
# what the misalignment costs, in interventional error
# ---------------------------------------------------------------------------

def floor(b: np.ndarray, w: np.ndarray, v: np.ndarray, v_out: float) -> float:
    """Smallest interventional error any realisation of a macro intervention
    can have, in nats of KL divergence.

    A macro intervention do(T = t) does not say how the parts were set, and
    every split along w.h = t is a different real action with a different
    error. This is the minimum over all of them, attained when the residual
    is orthogonal to the chosen split. It is a guarantee about the actions
    nobody checked, which is what makes it more useful than measuring the
    error of the ones they did.
    """
    rho = misalignment(b, w, v)
    return 0.5 * (np.log1p(rho / v_out) - rho / (v_out + rho))


def worst_case(b: np.ndarray, w: np.ndarray, v: np.ndarray, v_out: float,
               target: float) -> float:
    """Largest error among the single-part realisations of do(T = target).

    Putting the whole intervention through one part is the extreme a
    practitioner is most likely to reach for ("just add hours to maths"), so
    it brackets the realistic range rather than an abstract supremum.
    """
    r, w_arr = residual(b, w, v), np.asarray(w, dtype=float)
    rho = misalignment(b, w, v)
    worst = 0.0
    for i in range(len(w_arr)):
        if abs(w_arr[i]) < 1e-12:
            continue                     # this part cannot realise the target
        h = np.zeros_like(w_arr)
        h[i] = target / w_arr[i]
        gap = 0.5 * ((v_out + float(r @ h) ** 2) / (v_out + rho)
                     - 1.0 + np.log1p(rho / v_out))
        worst = max(worst, gap)
    return float(worst)


# ---------------------------------------------------------------------------
# choosing the weights
# ---------------------------------------------------------------------------

def repair(b: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Weights that make a one-outcome composite exactly consistent.

    The condition is an equality, not a minimum to approach, so this is one
    step and not a search. Returned normalised, since only the direction
    matters: scaling w rescales the composite and the fitted coefficient
    compensates exactly.
    """
    b = np.asarray(b, dtype=float)
    n = np.linalg.norm(b)
    return b / n if n > 0 else np.ones_like(b) / np.sqrt(len(b))


def design(B: Sequence[np.ndarray], v: np.ndarray) -> np.ndarray:
    """Weights minimising total misalignment across several outcomes.

    Minimising sum_k <r_k,r_k> is maximising sum_k <b_k,w>^2/<w,w>, a
    Rayleigh quotient in the V metric. Substituting u = V^{1/2} w turns it
    into an ordinary eigenvalue problem, so the optimum is the leading
    eigenvector of V^{1/2} (sum_k b_k b_k^T) V^{1/2}, mapped back.

    With one outcome this reduces to `repair`, which is a useful check on the
    implementation rather than a special case to code separately.
    """
    v = np.asarray(v, dtype=float)
    root, inv_root = np.diag(np.sqrt(v)), np.diag(1.0 / np.sqrt(v))
    scatter = sum(np.outer(np.asarray(b, dtype=float),
                           np.asarray(b, dtype=float)) for b in B)
    vals, vecs = np.linalg.eigh(root @ scatter @ root)
    w = inv_root @ vecs[:, int(np.argmax(vals))]
    n = np.linalg.norm(w)
    w = w / n if n > 0 else w
    # fix the sign so the weights read naturally against the first outcome
    return w if float(np.asarray(B[0]) @ w) >= 0 else -w


def gram(B: Sequence[np.ndarray], w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The matrix G_jk = <r_j, r_k> of residual overlaps between outcomes.

    This matrix splits the damage in two, and the split is sharp.

    Its DIAGONAL is the misalignment of each outcome, and it is invisible in
    observational data: the coarse model reproduces each outcome's marginal
    law exactly however wrong the weights are. That is experiment E1.

    Its OFF-DIAGONAL is the covariance between outcome residuals, and it is
    exactly the amount by which a coarse model with independent noises fails
    to reproduce the observed covariance between two outcomes. So with more
    than one outcome, part of the problem does leave a trace in the data.

    The practical reading: the quantity you care about stays invisible, but a
    correlate of it becomes visible as soon as the composite feeds two things
    at once. Checking it means looking at the correlation between the
    residuals of two separate regressions, which is not a check anyone runs by
    habit, and a non-zero value tells you something is wrong without telling
    you what or how much.
    """
    R = [residual(b, w, v) for b in B]
    K = len(R)
    return np.array([[inner(R[j], R[k], v) for k in range(K)]
                     for j in range(K)], dtype=float)


def observable_mismatch(B: Sequence[np.ndarray], w: np.ndarray,
                        v: np.ndarray) -> float:
    """Largest absolute off-diagonal of the Gram matrix: the biggest
    discrepancy an observational check could in principle notice. Zero for a
    single outcome, which is why E1's finding is sharp there."""
    G = gram(B, w, v)
    K = G.shape[0]
    if K < 2:
        return 0.0
    off = [abs(G[j, k]) for j in range(K) for k in range(K) if j != k]
    return float(max(off))


def irreducible(B: Sequence[np.ndarray], v: np.ndarray) -> float:
    """Total misalignment at the optimum. Zero exactly when the effect
    vectors are parallel, that is when all outcomes agree on what the
    composite should mean. Otherwise it is a cost no reweighting removes,
    and the signal that the composite is serving two purposes at once."""
    w = design(B, v)
    return float(sum(misalignment(b, w, v) for b in B))


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

@dataclass
class OutcomeReport:
    name: str
    misalignment: float
    captured: float
    angle: float
    floor: float
    worst_case: float
    residual: np.ndarray


@dataclass
class AuditReport:
    """Everything the linter prints, as data rather than as text."""
    composite: str
    part_names: List[str]
    weights: np.ndarray
    outcomes: List[OutcomeReport]
    verdict: str                       # "safe" | "repairable" | "irreducible"
    total_misalignment: float
    observable: float = 0.0            # what a data check could notice
    recommended: Optional[np.ndarray] = None
    recommended_total: Optional[float] = None
    irreducible_total: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def worst_outcome(self) -> OutcomeReport:
        return max(self.outcomes, key=lambda o: o.misalignment)


def audit(B: Sequence[np.ndarray],
          w: np.ndarray,
          v: np.ndarray,
          v_out: Sequence[float],
          part_names: Optional[Sequence[str]] = None,
          outcome_names: Optional[Sequence[str]] = None,
          composite: str = "composite",
          target: float = 2.0,
          safe_fraction: float = SAFE_FRACTION) -> AuditReport:
    """Full verdict on a proposed composite.

    The three verdicts are deliberately distinct, because they call for
    different actions. "safe" means act on it. "repairable" means the weights
    are wrong but fixable, and the fix is reported. "irreducible" means the
    composite is being asked to serve outcomes that disagree about what it
    should mean, no weighting fixes that, and the composite should be split.
    Collapsing the last two into one verdict would tell a practitioner to keep
    tuning something that cannot be tuned.
    """
    B = [np.asarray(b, dtype=float) for b in B]
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    part_names = list(part_names or [f"x{i + 1}" for i in range(len(v))])
    outcome_names = list(outcome_names or [f"outcome{k + 1}" for k in range(len(B))])

    reports = [
        OutcomeReport(
            name=name,
            misalignment=misalignment(b, w, v),
            captured=captured(b, w, v),
            angle=angle_degrees(b, w, v),
            floor=floor(b, w, v, vo),
            worst_case=worst_case(b, w, v, vo, target),
            residual=residual(b, w, v),
        )
        for b, vo, name in zip(B, v_out, outcome_names)
    ]
    total = float(sum(r.misalignment for r in reports))
    scale = float(sum(inner(b, b, v) for b in B))
    rel = total / scale if scale > 0 else 0.0

    best = design(B, v)
    best_total = float(sum(misalignment(b, best, v) for b in B))
    best_rel = best_total / scale if scale > 0 else 0.0

    notes: List[str] = []
    if rel <= safe_fraction:
        verdict = "safe"
    elif best_rel <= safe_fraction:
        verdict = "repairable"
        notes.append("Reweighting makes this composite exactly consistent.")
    else:
        verdict = "irreducible"
        notes.append("No weighting satisfies every outcome: the effect vectors "
                     "are not parallel. Consider splitting the composite.")
    if len(B) > 1 and verdict != "safe":
        notes.append(f"Best achievable leaves {best_rel * 100:.1f}% of the "
                     f"causal effect uncaptured.")

    obs = observable_mismatch(B, w, v)
    if obs > 1e-9:
        notes.append(f"Partly visible in data: the residuals of two outcome "
                     f"regressions should be uncorrelated and are off by "
                     f"{obs:.4f}. That check detects a problem without sizing it.")
    elif len(B) == 1 and verdict != "safe":
        notes.append("Invisible in observational data: the coarse model "
                     "reproduces the observed law exactly regardless.")

    return AuditReport(
        composite=composite, part_names=part_names, weights=w,
        outcomes=reports, verdict=verdict, total_misalignment=total,
        observable=obs,
        recommended=best, recommended_total=best_total,
        irreducible_total=best_total if verdict == "irreducible" else 0.0,
        notes=notes,
    )
