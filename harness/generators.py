"""
Samplers for the model space the framework is defined over.

WHY THIS FILE IS THE ARTIFACT
-----------------------------
Every claim in the framework is universally quantified: "for every SCM M,
for every intervention, ...". Falsifying such a claim means searching the
space of models it quantifies over. The assertions that check each claim are
one line each. All of the difficulty, and all of the scientific content, is
in sampling that space faithfully.

The v1 suite (archive/v1_suite.py) is the cautionary case. It built causal
coefficient matrices with

    C = np.tril(rng.normal(size=(n, n)) * 0.6, k=-1)

which fills in every edge below the diagonal, so every variable except the
first has every predecessor as a parent. Sampled 2000 times it produced a
complete DAG 2000 times, and therefore exactly one root variable every time.
Closure under convex combination holds on complete DAGs with unrestricted
mechanisms, so that suite was drawing exclusively from the sub-class where
the theorem it never tested happens to be true.

Definition 1 of the framework says "a DAG", not "a complete DAG". This file
therefore treats edge density as an explicit parameter rather than a silent
default. Nothing here is special-cased toward any known counterexample: the
samplers are written from the definitions, and the structural statistics they
produce are reported by `describe` so that a reader can check the sample is
not aimed.

CONVENTIONS
-----------
Variables are always in topological order, so parents precede children and
the coefficient matrix is strictly lower triangular. Every sampler takes an
explicit numpy Generator, so every result is reproducible from its seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from core.scm import LinearGaussianSCM


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SCMSpace:
    """The space of SCMs to sample from.

    Every field is a knob the paper can sweep. Defaults are deliberately
    permissive rather than convenient: they describe the models Definition 1
    allows, not the models that are pleasant to work with.

    n_vars        range of variable counts, inclusive
    edge_prob     probability that an admissible edge is present. 1.0 gives a
                  complete DAG, which is what the v1 suite produced; lower
                  values give the sparse graphs Definition 1 also permits
    coeff_scale   standard deviation of a present causal coefficient
    coeff_min     coefficients are resampled until |c| >= this, so that a
                  "present" edge is not silently near-absent
    noise_var     range of exogenous variances, inclusive
    """

    n_vars: Tuple[int, int] = (2, 6)
    edge_prob: float = 0.5
    coeff_scale: float = 0.8
    coeff_min: float = 0.15
    noise_var: Tuple[float, float] = (0.25, 2.0)

    def __post_init__(self) -> None:
        lo, hi = self.n_vars
        if lo < 1 or hi < lo:
            raise ValueError("n_vars must be a non-empty range of positive ints")
        if not 0.0 <= self.edge_prob <= 1.0:
            raise ValueError("edge_prob must lie in [0, 1]")
        if self.noise_var[0] <= 0 or self.noise_var[1] < self.noise_var[0]:
            raise ValueError("noise_var must be a non-empty range of positive floats")


# ---------------------------------------------------------------------------
# structural description, used to correlate failures with model shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Structure:
    """Structural facts about a sampled SCM, independent of its numbers."""

    n_vars: int
    n_edges: int
    n_roots: int
    density: float          # edges present / edges admissible
    is_complete: bool
    missing_edges: Tuple[Tuple[int, int], ...]   # (child, parent) pairs absent

    def as_row(self) -> Dict[str, float]:
        return {
            "n_vars": self.n_vars,
            "n_edges": self.n_edges,
            "n_roots": self.n_roots,
            "density": round(self.density, 4),
            "is_complete": int(self.is_complete),
        }


def describe(scm: LinearGaussianSCM) -> Structure:
    """Report the structure of an SCM from its coefficient matrix alone."""
    C = np.asarray(scm.C, dtype=float)
    n = C.shape[0]
    present = np.abs(C) > 0
    admissible = np.tril(np.ones((n, n), dtype=bool), k=-1)
    n_edges = int(present.sum())
    n_admissible = int(admissible.sum())
    missing = tuple(
        (int(i), int(j))
        for i in range(n)
        for j in range(i)
        if not present[i, j]
    )
    return Structure(
        n_vars=n,
        n_edges=n_edges,
        n_roots=int((present.sum(axis=1) == 0).sum()),
        density=(n_edges / n_admissible) if n_admissible else 1.0,
        is_complete=(n_edges == n_admissible),
        missing_edges=missing,
    )


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------

def sample_scm(rng: np.random.Generator, space: SCMSpace = SCMSpace()) -> LinearGaussianSCM:
    """Draw one linear Gaussian SCM from `space`.

    The graph is drawn first and the numbers second, so that edge density is
    controlled directly rather than emerging from coefficients that happen to
    be small. A present edge is guaranteed to carry a coefficient of at least
    `coeff_min` in absolute value, which keeps "present" and "absent" from
    blurring into each other and keeps `describe` honest.
    """
    lo, hi = space.n_vars
    n = int(rng.integers(lo, hi + 1))

    # 1. the graph: each admissible edge (child i, parent j<i) present or not
    admissible = np.tril(np.ones((n, n), dtype=bool), k=-1)
    present = admissible & (rng.random((n, n)) < space.edge_prob)

    # 2. the numbers: a present edge gets a coefficient bounded away from zero
    C = np.zeros((n, n))
    idx = np.argwhere(present)
    for i, j in idx:
        c = 0.0
        while abs(c) < space.coeff_min:
            c = rng.normal(0.0, space.coeff_scale)
        C[i, j] = c

    noise_var = rng.uniform(space.noise_var[0], space.noise_var[1], size=n)
    names = [f"X{i + 1}" for i in range(n)]
    return LinearGaussianSCM(C=C, noise_var=noise_var, names=names)


def sample_intervention_pair(
    rng: np.random.Generator,
    scm: LinearGaussianSCM,
    kind: str = "any",
    max_targets: Optional[int] = None,
) -> Tuple[LinearGaussianSCM, LinearGaussianSCM, Dict[str, object]]:
    """Draw two interventions on the same SCM, as the framework defines them.

    kind    "hard"  two hard interventions on the same target set
            "soft"  two soft interventions on the same target set
            "any"   either, chosen at random

    ON THE NUMBER OF TARGETS. The framework writes a hard intervention as
    do({X_i = x_i}_{X_i in Xtilde}) over a SET of variables, and a soft
    intervention likewise replaces a collection of mechanisms. An earlier
    version of this sampler drew a single target, and under it the claim
    "closure holds with unrestricted mechanisms" survived two thousand draws
    that it should not have survived: with one target, only one variable's
    mean moves between the two components of a blend, and a blend can only
    correlate two variables if both of their means move. The single-target
    restriction was not in the definition, it was in the sampler, and it hid
    a real failure. The target count is now drawn, with the maximum taken
    from the model size.

    Both interventions act on the SAME target set, because the claim under
    test is about blending two states of one model.

    Returns the two intervened models and a record of what was done, so a
    failing case can be reported exactly.
    """
    if kind == "any":
        kind = "hard" if rng.random() < 0.5 else "soft"

    n = len(scm.names)
    cap = n if max_targets is None else min(max_targets, n)
    k = int(rng.integers(1, cap + 1))
    targets = sorted(rng.choice(n, size=k, replace=False).tolist())

    if kind == "hard":
        v1 = rng.normal(0.0, 1.5, size=k)
        v2 = rng.normal(0.0, 1.5, size=k)
        while np.max(np.abs(v1 - v2)) < 0.3:           # the two must differ somewhere
            v2 = rng.normal(0.0, 1.5, size=k)
        a = scm.do_hard({t: float(v) for t, v in zip(targets, v1)})
        b = scm.do_hard({t: float(v) for t, v in zip(targets, v2)})
        return a, b, {"kind": "hard", "targets": targets,
                      "values_a": v1.round(4).tolist(), "values_b": v2.round(4).tolist()}

    # soft: rescale the noise gain and shift, leaving every parent set intact
    g1 = rng.uniform(0.3, 2.5, size=k)
    g2 = rng.uniform(0.3, 2.5, size=k)
    s1 = rng.normal(0.0, 1.0, size=k)
    s2 = rng.normal(0.0, 1.0, size=k)
    while np.max(np.abs(g1 - g2)) < 0.2 and np.max(np.abs(s1 - s2)) < 0.2:
        g2, s2 = rng.uniform(0.3, 2.5, size=k), rng.normal(0.0, 1.0, size=k)

    a, b = scm, scm
    for idx, t in enumerate(targets):
        a = a.do_soft(t, gain=float(g1[idx]), shift=float(s1[idx]))
        b = b.do_soft(t, gain=float(g2[idx]), shift=float(s2[idx]))
    return a, b, {"kind": "soft", "targets": targets,
                  "gains_a": g1.round(4).tolist(), "gains_b": g2.round(4).tolist(),
                  "shifts_a": s1.round(4).tolist(), "shifts_b": s2.round(4).tolist()}


def sample_weight(rng: np.random.Generator, interior: bool = True) -> float:
    """Draw a mixing weight. `interior` excludes 0 and 1, where no blending
    occurs and every closure claim is trivially true."""
    return float(rng.uniform(0.05, 0.95)) if interior else float(rng.random())


# ---------------------------------------------------------------------------
# a self-report, so a reader can check the sample is not aimed
# ---------------------------------------------------------------------------

def structural_report(
    rng: np.random.Generator,
    space: SCMSpace = SCMSpace(),
    draws: int = 2000,
) -> Dict[str, float]:
    """Summarise what `space` actually produces.

    This is printed in the paper beside any falsification rate, so that a
    claim of the form "the harness found X" can be read against the shape of
    the sample it searched.
    """
    rows = [describe(sample_scm(rng, space)) for _ in range(draws)]
    return {
        "draws": draws,
        "edge_prob": space.edge_prob,
        "mean_vars": float(np.mean([r.n_vars for r in rows])),
        "mean_density": float(np.mean([r.density for r in rows])),
        "frac_complete": float(np.mean([r.is_complete for r in rows])),
        "frac_two_or_more_roots": float(np.mean([r.n_roots >= 2 for r in rows])),
        "mean_roots": float(np.mean([r.n_roots for r in rows])),
        "frac_with_missing_edge": float(np.mean([len(r.missing_edges) > 0 for r in rows])),
    }


# ---------------------------------------------------------------------------
# abstractions, for the claims about alpha_X
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AbstractionSpace:
    """The space of micro/macro model pairs to sample from.

    The shape is the framework's own running example generalised: a micro
    model with `n_inputs` root variables driving one outcome, and a macro
    model in which the inputs are replaced by a single weighted aggregate.

    aggregation weights are drawn positive, because an aggregate with a
    near-zero weight on one of its parts is not an aggregate in any useful
    sense, and such instances otherwise dominate the tail of every statistic.
    """

    n_inputs: Tuple[int, int] = (2, 4)
    effect_scale: float = 1.0
    weight_range: Tuple[float, float] = (0.5, 1.5)
    noise_var: Tuple[float, float] = (0.25, 2.0)
    force_consistent: bool = False     # if True, set b = c w so the abstraction IS IC


def sample_abstraction(rng: np.random.Generator, space: AbstractionSpace = AbstractionSpace()):
    """Draw a micro model, an aggregation, and the macro model fitted to it.

    The macro model is always fitted by regression, so it reproduces the micro
    model's observable joint law exactly by construction. Whether it is
    interventionally consistent is a separate question, and is the point.

    Returns (LinearAbstraction, record) where record carries the effect vector
    b and the weights w, so a failing instance can be reported exactly.
    """
    from core.abstraction import LinearAbstraction, push_gaussian

    lo, hi = space.n_inputs
    p = int(rng.integers(lo, hi + 1))
    n = p + 1                                  # inputs plus the outcome

    w = rng.uniform(space.weight_range[0], space.weight_range[1], size=p)
    b = (w * float(rng.uniform(0.4, 1.6))
         if space.force_consistent
         else rng.normal(0.0, space.effect_scale, size=p))

    C = np.zeros((n, n))
    C[p, :p] = b                                # outcome driven by the inputs
    noise_var = rng.uniform(space.noise_var[0], space.noise_var[1], size=n)
    micro = LinearGaussianSCM(C, noise_var, [f"H{i + 1}" for i in range(p)] + ["Mk"])

    F = np.zeros((2, n))
    F[0, :p] = w                                # the aggregate
    F[1, p] = 1.0                               # the outcome, carried through

    _, cov = push_gaussian(F, micro.law())
    c_fit = cov[0, 1] / cov[0, 0]               # regression coefficient
    macro = LinearGaussianSCM(
        C=[[0.0, 0.0], [c_fit, 0.0]],
        noise_var=[cov[0, 0], max(cov[1, 1] - c_fit * c_fit * cov[0, 0], 1e-12)],
        names=["T", "Mk"],
    )
    record = {"b": b.tolist(), "w": w.tolist(), "c": float(c_fit),
              "rho_mis": float(np.sum((b - c_fit * w) ** 2))}
    return LinearAbstraction(micro, macro, F), record


# ---------------------------------------------------------------------------
# networks, for the sheaf and cosheaf claims
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkSpace:
    """The space of networks to sample from.

    n_nodes        how many subjects
    node_vars      range of variable counts per subject
    backbone_width range of backbone dimensions on an edge
    topology       "chain" or "star"
    """

    n_nodes: Tuple[int, int] = (2, 4)
    node_vars: Tuple[int, int] = (3, 6)
    backbone_width: Tuple[int, int] = (1, 2)
    topology: str = "chain"
    scm_space: SCMSpace = SCMSpace()


def sample_network(rng: np.random.Generator, space: NetworkSpace = NetworkSpace()):
    """Draw a network of subjects joined by shared backbone models.

    Restriction matrices are drawn with full row rank, because a rank
    deficient restriction is a degenerate case the framework treats
    separately and it would make every divergence on that edge infinite.
    """
    from core.sheaf import CausalNetwork

    k = int(rng.integers(space.n_nodes[0], space.n_nodes[1] + 1))
    net = CausalNetwork()
    names = [chr(ord("A") + i) for i in range(k)]
    for name in names:
        lo, hi = space.node_vars
        node_space = SCMSpace(
            n_vars=(lo, hi),
            edge_prob=space.scm_space.edge_prob,
            coeff_scale=space.scm_space.coeff_scale,
            coeff_min=space.scm_space.coeff_min,
            noise_var=space.scm_space.noise_var,
        )
        net.add_node(name, sample_scm(rng, node_space))

    pairs = ([(names[i], names[i + 1]) for i in range(k - 1)] if space.topology == "chain"
             else [(names[0], names[i]) for i in range(1, k)])

    for idx, (a, b) in enumerate(pairs):
        d = int(rng.integers(space.backbone_width[0], space.backbone_width[1] + 1))
        d = min(d, net.nodes[a].n, net.nodes[b].n)
        backbone = LinearGaussianSCM(np.zeros((d, d)), np.ones(d), [f"E{idx}_{j}" for j in range(d)])
        Fa = _full_rank_matrix(rng, d, net.nodes[a].n)
        Fb = _full_rank_matrix(rng, d, net.nodes[b].n)
        net.add_edge(f"E{idx}", backbone, a, Fa, b, Fb)
    return net


def _full_rank_matrix(rng: np.random.Generator, rows: int, cols: int, tries: int = 50) -> np.ndarray:
    """A rows-by-cols matrix of full row rank. Redraws rather than perturbing,
    so the distribution stays what it says it is."""
    for _ in range(tries):
        F = rng.normal(size=(rows, cols))
        if np.linalg.matrix_rank(F) == rows:
            return F
    raise RuntimeError(f"could not draw a full row rank {rows}x{cols} matrix")
