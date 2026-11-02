"""
Shrinking: reduce a random counterexample to a minimal one.

WHY THIS FILE EXISTS
--------------------
The harness finds its first counterexample on a model with six variables,
five edges, a three-element intervention target set and a mixing weight of
0.6758. Almost none of that is why the theorem fails. A counterexample is only
persuasive once the incidental parts have been stripped away.

Shrinking does that automatically. Starting from a failing case it repeatedly
proposes a simpler case, keeps the proposal when the failure survives, and
stops when nothing simpler fails. What comes out is minimal with respect to
the moves available.

WHY IT IS EVIDENCE AND NOT JUST TIDYING
---------------------------------------
The shrinker sees one bit about each case: did it fail. It is given no access
to the invariant that fired, no notion of a root variable, no notion of
independence or kurtosis, and no knowledge of any counterexample derived by
hand. Read `MOVES` below: every move is a generic structural simplification
(remove a variable, remove an edge, round a number, drop a target, pull the
weight toward one half) of the kind you would write for any model at all.

So if a random six-variable failure reduces to two variables, both roots, one
intervention on both and a weight of one half, and that is the shape of the
counterexample derived by hand in the written work, then the harness arrived
there independently. The reduction was carried out by a loop that could not
have been steering toward it.

A NOTE ON PROVENANCE
--------------------
Hypothesis performs shrinking automatically and would be the stronger claim,
since its reducer is third-party code. It is not used here because the case
under test is a structured object, an SCM together with a pair of
interventions, which does not map onto Hypothesis strategies without
rewriting the generators to be strategy-driven. The honest version of the
claim is therefore: the moves are generic and auditable, and they are listed
in one place so that a reader can confirm none of them encodes the expected
answer. Moving the generators onto Hypothesis remains the better long-term
option and is recorded as future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from core.scm import LinearGaussianSCM
from harness import exclusions as exc


# ---------------------------------------------------------------------------
# a case: everything needed to reproduce one draw
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """One closure test, fully determined and reproducible.

    C          strictly lower triangular causal coefficients
    noise_var  exogenous variances
    targets    the intervention target set
    kind       "hard" or "soft"
    a, b       the two interventions' parameters, one row per target
               hard: a value; soft: a (gain, shift) pair
    lam        the mixing weight
    """

    C: Tuple[Tuple[float, ...], ...]
    noise_var: Tuple[float, ...]
    kind: str
    targets: Tuple[int, ...]
    a: Tuple[Tuple[float, ...], ...]
    b: Tuple[Tuple[float, ...], ...]
    lam: float

    # ----- rebuild the objects under test ---------------------------------
    @property
    def n(self) -> int:
        return len(self.noise_var)

    def base(self) -> LinearGaussianSCM:
        return LinearGaussianSCM(np.array(self.C, dtype=float),
                                 np.array(self.noise_var, dtype=float),
                                 [f"X{i}" for i in range(self.n)])

    def _apply(self, scm: LinearGaussianSCM, params) -> LinearGaussianSCM:
        if self.kind == "hard":
            return scm.do_hard({t: float(p[0]) for t, p in zip(self.targets, params)})
        out = scm
        for t, p in zip(self.targets, params):
            out = out.do_soft(t, gain=float(p[0]), shift=float(p[1]))
        return out

    def models(self) -> Tuple[LinearGaussianSCM, LinearGaussianSCM, LinearGaussianSCM]:
        base = self.base()
        return base, self._apply(base, self.a), self._apply(base, self.b)

    # ----- how big is this case -------------------------------------------
    def size(self) -> Tuple:
        """Lexicographic size. Earlier entries dominate, so the shrinker
        prefers fewer variables over rounder numbers."""
        C = np.array(self.C, dtype=float)
        n_edges = int(np.count_nonzero(C))
        roughness = sum(_roughness(v) for v in np.array(C).ravel())
        roughness += sum(_roughness(v) for v in self.noise_var)
        roughness += sum(_roughness(v) for row in (self.a + self.b) for v in row)
        roughness += _roughness(self.lam)
        return (self.n, n_edges, len(self.targets), round(roughness, 6))

    def describe(self) -> str:
        C = np.array(self.C, dtype=float)
        n_roots = int((np.abs(C).sum(axis=1) == 0).sum())
        return (f"{self.n} variables, {int(np.count_nonzero(C))} edges, "
                f"{n_roots} roots, {self.kind} intervention on {list(self.targets)}, "
                f"lambda = {self.lam:g}")

    def signature(self) -> str:
        """A one-line form that shows every number, so a trace of accepted
        moves records what actually changed. `describe` reports shape only,
        which makes a rounding step look like no step at all."""
        C = np.array(self.C, dtype=float)
        noise = ",".join(f"{v:g}" for v in self.noise_var)
        va = "/".join(",".join(f"{v:g}" for v in row) for row in self.a)
        vb = "/".join(",".join(f"{v:g}" for v in row) for row in self.b)
        return (f"n={self.n}, edges={int(np.count_nonzero(C))}, "
                f"noise=[{noise}], {self.kind} on {list(self.targets)}, "
                f"{va} vs {vb}, lambda={self.lam:g}")


def _roughness(x: float) -> float:
    """How far a number is from a simple one. Zero for 0, +-1 and +-0.5."""
    x = float(x)
    for simple in (0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0):
        if abs(x - simple) < 1e-12:
            return 0.0
    return 1.0 + min(abs(x - round(x, 1)), 1.0)


# ---------------------------------------------------------------------------
# the predicate: does this case still falsify closure
# ---------------------------------------------------------------------------

def still_fails(case: Case, which: Sequence[str]) -> bool:
    """True when some applicable invariant excludes the blend.

    This is the only thing the shrinker learns about a case.
    """
    try:
        base, ia, ib = case.models()
        verdicts = exc.screen_mixture(base, case.lam, ia.law(), ib.law())
    except Exception:
        return False                       # an invalid case is not a counterexample
    return any(v.excluded for v in verdicts if v.test in which)


# ---------------------------------------------------------------------------
# the moves
# ---------------------------------------------------------------------------
# Every move below is a generic structural simplification. None of them
# references a root variable, an independence relation, a kurtosis, or any
# property of a counterexample known in advance. This list is the auditable
# core of the integrity claim.

def _drop_variable(case: Case) -> List[Case]:
    """Remove one variable, reindexing the graph and the target set."""
    out = []
    for k in range(case.n):
        if case.n <= 2:
            break
        keep = [i for i in range(case.n) if i != k]
        C = np.array(case.C, dtype=float)[np.ix_(keep, keep)]
        nv = [case.noise_var[i] for i in keep]
        new_targets, new_a, new_b = [], [], []
        for t, pa, pb in zip(case.targets, case.a, case.b):
            if t == k:
                continue
            new_targets.append(t - 1 if t > k else t)
            new_a.append(pa)
            new_b.append(pb)
        if not new_targets:
            continue
        out.append(replace(case, C=_tup2(C), noise_var=tuple(nv),
                           targets=tuple(new_targets),
                           a=tuple(new_a), b=tuple(new_b)))
    return out


def _remove_edge(case: Case) -> List[Case]:
    """Set one present coefficient to zero."""
    C = np.array(case.C, dtype=float)
    out = []
    for i, j in zip(*np.nonzero(C)):
        D = C.copy()
        D[i, j] = 0.0
        out.append(replace(case, C=_tup2(D)))
    return out


def _simplify_coefficient(case: Case) -> List[Case]:
    """Round one coefficient toward a simpler value."""
    C = np.array(case.C, dtype=float)
    out = []
    for i, j in zip(*np.nonzero(C)):
        for target in (1.0, -1.0, round(float(C[i, j]), 1)):
            if abs(C[i, j] - target) < 1e-12 or target == 0.0:
                continue
            D = C.copy()
            D[i, j] = target
            out.append(replace(case, C=_tup2(D)))
    return out


def _simplify_noise(case: Case) -> List[Case]:
    """Round one noise variance toward one."""
    out = []
    for k, v in enumerate(case.noise_var):
        for target in (1.0, round(float(v), 1)):
            if abs(v - target) < 1e-12 or target <= 0:
                continue
            nv = list(case.noise_var)
            nv[k] = target
            out.append(replace(case, noise_var=tuple(nv)))
    return out


def _simplify_intervention(case: Case) -> List[Case]:
    """Round one intervention parameter toward a simpler value."""
    out = []
    for side in ("a", "b"):
        rows = [list(r) for r in getattr(case, side)]
        for ri, row in enumerate(rows):
            for ci, v in enumerate(row):
                simple = (1.0, 0.0, -1.0) if not (case.kind == "soft" and ci == 0) else (1.0, 0.5, 2.0)
                for target in simple + (round(float(v), 1),):
                    if abs(v - target) < 1e-12:
                        continue
                    if case.kind == "soft" and ci == 0 and target == 0.0:
                        continue                       # a zero gain is not a soft intervention
                    new = [list(r) for r in rows]
                    new[ri][ci] = float(target)
                    out.append(replace(case, **{side: tuple(tuple(r) for r in new)}))
    return out


def _drop_target(case: Case) -> List[Case]:
    """Remove one variable from the intervention target set."""
    if len(case.targets) <= 1:
        return []
    out = []
    for k in range(len(case.targets)):
        keep = [i for i in range(len(case.targets)) if i != k]
        out.append(replace(case,
                           targets=tuple(case.targets[i] for i in keep),
                           a=tuple(case.a[i] for i in keep),
                           b=tuple(case.b[i] for i in keep)))
    return out


def _simplify_weight(case: Case) -> List[Case]:
    """Pull the mixing weight toward one half."""
    if abs(case.lam - 0.5) < 1e-12:
        return []
    return [replace(case, lam=0.5), replace(case, lam=round(case.lam, 1))]


MOVES: List[Callable[[Case], List[Case]]] = [
    _drop_variable,
    _drop_target,
    _remove_edge,
    _simplify_weight,
    _simplify_coefficient,
    _simplify_noise,
    _simplify_intervention,
]


def _tup2(a: np.ndarray) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in np.atleast_2d(a))


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

@dataclass
class ShrinkReport:
    start: Case
    minimal: Case
    steps: int
    candidates_tried: int
    trace: List[str] = field(default_factory=list)


def shrink(case: Case, which: Sequence[str], max_steps: int = 500) -> ShrinkReport:
    """Greedily reduce `case` while it still falsifies.

    At each step every move proposes candidates; the first candidate that is
    strictly smaller and still fails is adopted. The loop ends when no move
    produces one, so the result is minimal with respect to MOVES.
    """
    if not still_fails(case, which):
        raise ValueError("the starting case does not falsify; nothing to shrink")

    current = case
    steps = tried = 0
    trace: List[str] = []

    while steps < max_steps:
        improved = False
        for move in MOVES:
            for candidate in move(current):
                tried += 1
                if candidate.size() >= current.size():
                    continue
                if still_fails(candidate, which):
                    trace.append(f"{move.__name__}: {current.signature()}"
                                 f" -> {candidate.signature()}")
                    current = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        steps += 1

    return ShrinkReport(case, current, steps, tried, trace)


# ---------------------------------------------------------------------------
# building a case from a generator draw
# ---------------------------------------------------------------------------

def case_from_draw(scm: LinearGaussianSCM, record: Dict, lam: float) -> Case:
    """Package a draw from `generators.sample_intervention_pair` as a Case."""
    targets = tuple(record["targets"])
    if record["kind"] == "hard":
        a = tuple((float(v),) for v in record["values_a"])
        b = tuple((float(v),) for v in record["values_b"])
    else:
        a = tuple((float(g), float(s)) for g, s in zip(record["gains_a"], record["shifts_a"]))
        b = tuple((float(g), float(s)) for g, s in zip(record["gains_b"], record["shifts_b"]))
    return Case(C=_tup2(np.asarray(scm.C, dtype=float)),
                noise_var=tuple(float(v) for v in scm.noise_var),
                kind=record["kind"], targets=targets, a=a, b=b, lam=float(lam))
