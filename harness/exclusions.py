"""
Membership exclusion tests: necessary conditions for lying inside CK(M).

THE PROBLEM THIS FILE SOLVES
----------------------------
Theorem 1 of the framework says that a convex combination of two measures in
CK(M) is again in CK(M). To falsify that, a harness has to decide whether a
given measure is in CK(M). It cannot. CK(M) is the set of measures reachable
by *some* intervention, and there is no algorithm that enumerates every soft
intervention on a model.

So membership is never tested directly. What is tested instead is a set of
properties that EVERY member of CK(M) must have. If a candidate measure
violates one, it is certainly not a member, and closure is falsified on that
instance. If it violates none, nothing is concluded: the exclusion tests are
one-sided, and this file never reports a measure as being in CK(M).

That one-sidedness is what makes the harness sound. A false "not a member"
would invent a counterexample; a missed one only costs sensitivity.

WHERE EACH INVARIANT COMES FROM
-------------------------------
Each is a one-line consequence of the framework's own Definition 1, derived
without reference to any known counterexample.

1. ANCESTRAL INDEPENDENCE.  In a Markovian SCM the exogenous variables are
   mutually independent. A variable is a measurable function of the noises of
   its ancestors and itself. So if two variables have disjoint ancestor sets
   they are functions of disjoint, independent noise collections, hence
   independent. A hard intervention replaces a mechanism by a constant, which
   can only shrink an ancestor set; a soft intervention preserves parent sets,
   so leaves them unchanged. Ancestral disjointness in M therefore survives
   every intervention, and the independence holds in every member of CK(M).

2. ROOT INDEPENDENCE.  The special case of (1) where both variables have no
   parents. Kept separate because it is the easiest to state, and because a
   model needs only two roots for it to bite.

3. GAUSSIAN MARGINALS.  Within the linear Gaussian family, every mechanism is
   linear and every noise is Gaussian, so every member of the causal knowledge
   has marginals that are Gaussian or (after a hard intervention) point
   masses. A Gaussian has excess kurtosis zero and a point mass has zero
   variance. Any marginal with positive variance and nonzero excess kurtosis
   therefore belongs to no member of the linear Gaussian causal knowledge.

VALIDATING THE EXCLUSION TESTS THEMSELVES
---------------------------------
An exclusion test that fires on a genuine member of CK(M) would manufacture false
counterexamples. `validate` runs every exclusion test against measures that are known
members by construction, namely observational and interventional laws of the
model itself, and asserts that none fire. That control runs as part of the
scoreboard, before any claim is tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.scm import LinearGaussianSCM

Law = Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    """The result of one exclusion test on one candidate measure.

    excluded  True means: this measure is certainly NOT in CK(M).
              False means: this exclusion test found nothing. It does NOT mean the
              measure is a member.
    witness   the quantity measured, so a failure can be reported with a size
    """

    test: str
    excluded: bool
    witness: float
    tolerance: float
    detail: str

    def __str__(self) -> str:
        verb = "EXCLUDES" if self.excluded else "no finding"
        return f"{self.test}: {verb} ({self.witness:.6g} vs tol {self.tolerance:.1g}) {self.detail}"


# ---------------------------------------------------------------------------
# mixture moments, exact
# ---------------------------------------------------------------------------

def mixture_moments(lam: float, law1: Law, law2: Law) -> Law:
    """Mean and covariance of lam * law1 + (1 - lam) * law2, exactly.

    For a two-component mixture the law of total covariance gives
        mean = lam m1 + lam_bar m2
        cov  = lam (S1 + d1 d1^T) + lam_bar (S2 + d2 d2^T),  d_i = m_i - mean
    The outer-product terms are the between-component spread, and they are
    what a blended-mechanism construction discards.
    """
    m1, S1 = np.atleast_1d(law1[0]), np.atleast_2d(law1[1])
    m2, S2 = np.atleast_1d(law2[0]), np.atleast_2d(law2[1])
    lb = 1.0 - lam
    mean = lam * m1 + lb * m2
    d1, d2 = m1 - mean, m2 - mean
    cov = lam * (S1 + np.outer(d1, d1)) + lb * (S2 + np.outer(d2, d2))
    return mean, cov


def mixture_excess_kurtosis(lam: float, law1: Law, law2: Law) -> np.ndarray:
    """Per-coordinate excess kurtosis of a two-component Gaussian mixture, exactly.

    For one coordinate with component means mu_i, variances s_i and offsets
    d_i = mu_i - mean:
        m2 = sum_i w_i (s_i + d_i^2)
        m4 = sum_i w_i (3 s_i^2 + 6 s_i d_i^2 + d_i^4)
        excess kurtosis = m4 / m2^2 - 3
    Coordinates with m2 at machine zero are returned as NaN: a point mass has
    no kurtosis, and the framework permits point-mass marginals after a hard
    intervention.
    """
    m1, S1 = np.atleast_1d(law1[0]), np.atleast_2d(law1[1])
    m2_, S2 = np.atleast_1d(law2[0]), np.atleast_2d(law2[1])
    lb = 1.0 - lam
    mean = lam * m1 + lb * m2_
    d1, d2 = m1 - mean, m2_ - mean
    s1, s2 = np.diag(S1), np.diag(S2)

    m2 = lam * (s1 + d1 ** 2) + lb * (s2 + d2 ** 2)
    m4 = (lam * (3 * s1 ** 2 + 6 * s1 * d1 ** 2 + d1 ** 4)
          + lb * (3 * s2 ** 2 + 6 * s2 * d2 ** 2 + d2 ** 4))

    out = np.full_like(m2, np.nan, dtype=float)
    ok = m2 > 1e-14
    out[ok] = m4[ok] / m2[ok] ** 2 - 3.0
    return out


# ---------------------------------------------------------------------------
# graph structure
# ---------------------------------------------------------------------------

def ancestor_sets(scm: LinearGaussianSCM) -> List[set]:
    """anc(i) = {i} together with every variable from which i is reachable.

    Variables are in topological order, so one forward pass suffices: a
    variable's ancestors are itself plus the ancestors of its parents.
    """
    C = np.asarray(scm.C, dtype=float)
    n = C.shape[0]
    anc: List[set] = []
    for i in range(n):
        s = {i}
        for j in range(i):
            if C[i, j] != 0.0:
                s |= anc[j]
        anc.append(s)
    return anc


def ancestrally_disjoint_pairs(scm: LinearGaussianSCM) -> List[Tuple[int, int]]:
    """Pairs (i, j) sharing no ancestor. These must be independent in every
    member of CK(M). Note that a directed path from j to i puts j in anc(i),
    so connected pairs are correctly excluded."""
    anc = ancestor_sets(scm)
    n = len(anc)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if not (anc[i] & anc[j])]


def root_pairs(scm: LinearGaussianSCM) -> List[Tuple[int, int]]:
    """Pairs of variables that both have no parents."""
    C = np.asarray(scm.C, dtype=float)
    roots = [i for i in range(C.shape[0]) if not np.any(C[i, :] != 0.0)]
    return [(roots[a], roots[b]) for a in range(len(roots)) for b in range(a + 1, len(roots))]


# ---------------------------------------------------------------------------
# the exclusion tests
# ---------------------------------------------------------------------------

CORRELATION_TOL = 1e-8
KURTOSIS_TOL = 1e-8


def _max_abs_correlation(cov: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Largest |correlation| over the given index pairs, with the pair attaining it.

    Correlation rather than covariance, so the tolerance is scale free: a
    covariance of 1e-6 means something different for variables of variance
    1e-6 and of variance 1e6.
    """
    worst, where = 0.0, None
    for i, j in pairs:
        vi, vj = cov[i, i], cov[j, j]
        if vi <= 1e-14 or vj <= 1e-14:
            continue                       # a point mass is independent of everything
        r = abs(cov[i, j]) / np.sqrt(vi * vj)
        if r > worst:
            worst, where = r, (i, j)
    return worst, where


def exclusion_ancestral_independence(scm: LinearGaussianSCM, law: Law,
                                  tol: float = CORRELATION_TOL) -> Verdict:
    """Variables with disjoint ancestor sets are independent in every member."""
    pairs = ancestrally_disjoint_pairs(scm)
    if not pairs:
        return Verdict("ancestral_independence", False, 0.0, tol,
                       "no ancestrally disjoint pair in this model; exclusion test inapplicable")
    cov = np.atleast_2d(law[1])
    worst, where = _max_abs_correlation(cov, pairs)
    detail = (f"pair {where} correlated" if where is not None and worst > tol
              else f"{len(pairs)} disjoint pair(s) all uncorrelated")
    return Verdict("ancestral_independence", worst > tol, worst, tol, detail)


def exclusion_root_independence(scm: LinearGaussianSCM, law: Law,
                             tol: float = CORRELATION_TOL) -> Verdict:
    """Root variables are mutually independent in every member."""
    pairs = root_pairs(scm)
    if not pairs:
        return Verdict("root_independence", False, 0.0, tol,
                       "fewer than two root variables; exclusion test inapplicable")
    cov = np.atleast_2d(law[1])
    worst, where = _max_abs_correlation(cov, pairs)
    detail = (f"roots {where} correlated" if where is not None and worst > tol
              else f"{len(pairs)} root pair(s) all uncorrelated")
    return Verdict("root_independence", worst > tol, worst, tol, detail)


def exclusion_gaussian_marginals(lam: float, law1: Law, law2: Law,
                              tol: float = KURTOSIS_TOL) -> Verdict:
    """Linear Gaussian members have Gaussian or point-mass marginals.

    Takes the mixture rather than a single law, because excess kurtosis is a
    property of the mixture and is not recoverable from its first two moments.
    """
    ek = mixture_excess_kurtosis(lam, law1, law2)
    finite = ek[~np.isnan(ek)]
    if finite.size == 0:
        return Verdict("gaussian_marginals", False, 0.0, tol,
                       "every marginal is a point mass; exclusion test inapplicable")
    worst = float(np.max(np.abs(finite)))
    idx = int(np.nanargmax(np.abs(ek)))
    detail = (f"coordinate {idx} has excess kurtosis {ek[idx]:.6g}" if worst > tol
              else "all marginals have zero excess kurtosis")
    return Verdict("gaussian_marginals", worst > tol, worst, tol, detail)


# ---------------------------------------------------------------------------
# applying the full panel
# ---------------------------------------------------------------------------

def screen_mixture(scm: LinearGaussianSCM, lam: float, law1: Law, law2: Law) -> List[Verdict]:
    """Run every exclusion test against the mixture lam*law1 + (1-lam)*law2."""
    mixed = mixture_moments(lam, law1, law2)
    return [
        exclusion_root_independence(scm, mixed),
        exclusion_ancestral_independence(scm, mixed),
        exclusion_gaussian_marginals(lam, law1, law2),
    ]


def screen_law(scm: LinearGaussianSCM, law: Law) -> List[Verdict]:
    """Run the exclusion tests that apply to a single measure rather than a mixture."""
    return [
        exclusion_root_independence(scm, law),
        exclusion_ancestral_independence(scm, law),
    ]


# ---------------------------------------------------------------------------
# the exclusion tests' own control
# ---------------------------------------------------------------------------

def validate(scm: LinearGaussianSCM, rng: np.random.Generator,
             n_interventions: int = 12) -> List[Verdict]:
    """Run the exclusion tests against measures that ARE in CK(M) by construction.

    The observational law is a member. So is the law of any hard or soft
    intervention. None of these may be excluded by any exclusion test: an exclusion test that
    fires here is unsound and would manufacture counterexamples.

    Returns the verdicts that wrongly excluded, so an empty list means the
    panel passed its control on this model.
    """
    members: List[Law] = [scm.law()]
    n = scm.n
    for _ in range(n_interventions):
        target = int(rng.integers(0, n))
        if rng.random() < 0.5:
            members.append(scm.do_hard({target: float(rng.normal(0, 1.5))}).law())
        else:
            members.append(scm.do_soft(target,
                                       gain=float(rng.uniform(0.3, 2.5)),
                                       shift=float(rng.normal(0, 1.0))).law())

    wrong: List[Verdict] = []
    for law in members:
        for v in screen_law(scm, law):
            if v.excluded:
                wrong.append(v)
        # a "mixture" of a member with itself is that member, so no exclusion test may
        # exclude it at any weight; this also exercises the kurtosis exclusion test
        for lam in (0.25, 0.5, 0.75):
            v = exclusion_gaussian_marginals(lam, law, law)
            if v.excluded:
                wrong.append(v)
    return wrong
