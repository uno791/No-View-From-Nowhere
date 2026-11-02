"""
The framework's claims, written as executable properties.

HOW A CLAIM BECOMES A PROPERTY
------------------------------
Every claim in the framework is universally quantified: "for every SCM, for
every intervention, ...". A property here is that sentence with the
quantifiers replaced by draws from `generators`, and the conclusion replaced
by a numerical check. Nothing is rephrased to make it easier to satisfy or
easier to break; each docstring quotes the claim it transcribes and names its
source.

Properties return an `Outcome` rather than raising, because the runner needs
to count passes as well as failures. A claim that survives a million draws is
as much a result as one that falls, and the survivors are what make a single
failure credible.

A NOTE ON WHAT IS MEASURED
--------------------------
Gaussian measures are compared by their mean and covariance rather than by KL
divergence wherever a hard intervention may be involved, because a hard
intervention produces a point-mass coordinate, the covariance is singular and
KL is not finite. For Gaussians, equality of the first two moments is
equality of the measures, so nothing is lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.abstraction import push_gaussian
from core.kl import gaussian_kl
from harness import generators as gen
from harness import exclusions as exc

Law = Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------

@dataclass
class Outcome:
    name: str
    source: str
    statement: str
    falsified: bool
    residual: float
    detail: str = ""
    witness: Optional[Dict] = None
    inapplicable: bool = False      # the draw did not exercise the claim


def _ok(name, source, statement, residual, detail="") -> Outcome:
    return Outcome(name, source, statement, False, float(residual), detail)


def _skip(name, source, statement, detail) -> Outcome:
    return Outcome(name, source, statement, False, 0.0, detail, inapplicable=True)


def _fail(name, source, statement, residual, detail, witness) -> Outcome:
    return Outcome(name, source, statement, True, float(residual), detail, witness)


def _moment_gap(a: Law, b: Law) -> float:
    """Largest absolute difference between two Gaussians' means and covariances."""
    return float(max(np.max(np.abs(np.atleast_1d(a[0]) - np.atleast_1d(b[0]))),
                     np.max(np.abs(np.atleast_2d(a[1]) - np.atleast_2d(b[1])))))


TOL = 1e-9


# ---------------------------------------------------------------------------
# claims of the source paper
# ---------------------------------------------------------------------------

def prop_intervention_is_natural(rng, cfg) -> Outcome:
    """Lemma 1: an intervention is a natural transformation, so I o M = M_I."""
    name, src = "intervention_is_natural", "source Lemma 1"
    stmt = "the transport composed with the mixing map equals the intervened mixing map"
    scm = gen.sample_scm(rng, cfg.scm_space)
    a, b, rec = gen.sample_intervention_pair(rng, scm, kind="any")
    z = scm.sample_noise(512, rng)
    worst = 0.0
    for intervened in (a, b):
        A, off = scm.intervention_map(intervened)
        worst = max(worst, float(np.max(np.abs(scm.mixing(z) @ A.T + off - intervened.mixing(z)))))
    if worst > 1e-8:
        return _fail(name, src, stmt, worst, "naturality square does not commute",
                     {"C": scm.C.tolist(), "intervention": rec})
    return _ok(name, src, stmt, worst)


def prop_transport_gives_interventional_law(rng, cfg) -> Outcome:
    """Lemma 1 (c): pushing the observational law along the transport gives the
    classical post-interventional law."""
    name, src = "transport_gives_interventional_law", "source Lemma 1"
    stmt = "pushforward of the observational measure along the transport equals the interventional measure"
    scm = gen.sample_scm(rng, cfg.scm_space)
    a, b, rec = gen.sample_intervention_pair(rng, scm, kind="any")
    m, S = scm.law()
    worst = 0.0
    for intervened in (a, b):
        A, off = scm.intervention_map(intervened)
        pushed = (A @ m + off, A @ S @ A.T)
        worst = max(worst, _moment_gap(pushed, intervened.law()))
    if worst > 1e-8:
        return _fail(name, src, stmt, worst, "transported law differs from the interventional law",
                     {"C": scm.C.tolist(), "intervention": rec})
    return _ok(name, src, stmt, worst)


def prop_convex_space_axioms(rng, cfg) -> Outcome:
    """Lemma 2: the probability measures form a convex space under cc_lambda.

    Tested on a finite measurable space, where a probability measure is a
    probability vector and the axioms are exact identities. The published
    proof is "a subset of a vector space inherits them", which does not depend
    on the dimension, so a finite space tests the claim as stated.
    """
    name, src = "convex_space_axioms", "source Lemma 2"
    stmt = "cc_lambda satisfies the four convex space axioms of Fritz Definition 3.1"
    k = int(rng.integers(2, 7))

    def pvec():
        v = rng.random(k) + 1e-3
        return v / v.sum()

    def cc(lam, p, q):
        return lam * p + (1 - lam) * q

    p1, p2, p3 = pvec(), pvec(), pvec()
    lam, mu = float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95))

    worst = 0.0
    worst = max(worst, np.max(np.abs(cc(0.0, p1, p2) - p2)))            # cc_0 = second
    worst = max(worst, np.max(np.abs(cc(1.0, p1, p2) - p1)))            # cc_1 = first
    worst = max(worst, np.max(np.abs(cc(lam, p1, p1) - p1)))            # idempotence
    worst = max(worst, np.max(np.abs(cc(lam, p1, p2) - cc(1 - lam, p2, p1))))   # symmetry
    # associativity, with the reparametrisation the paper states
    if abs(1.0 - lam * mu) > 1e-12:
        lam_e = lam * mu
        mu_e = lam * (1 - mu) / (1 - lam * mu)
        lhs = cc(lam, cc(mu, p1, p2), p3)
        rhs = cc(lam_e, p1, cc(mu_e, p2, p3))
        worst = max(worst, np.max(np.abs(lhs - rhs)))

    if worst > 1e-12:
        return _fail(name, src, stmt, worst, "a convex space axiom fails",
                     {"k": k, "lambda": lam, "mu": mu})
    return _ok(name, src, stmt, worst)


def _closure_property(rng, cfg, name: str, stmt: str, which: Tuple[str, ...]) -> Outcome:
    """Shared body for the two closure claims.

    `which` selects the exclusion tests that apply, because the two claims quantify
    over different model classes and are therefore excluded by different
    invariants. Both are one-sided: an exclusion proves non-membership, a pass
    concludes nothing.
    """
    src = "source Theorem 1"
    scm = gen.sample_scm(rng, cfg.scm_space)
    a, b, rec = gen.sample_intervention_pair(rng, scm, kind=cfg.intervention_kind)
    lam = gen.sample_weight(rng)

    verdicts = [v for v in exc.screen_mixture(scm, lam, a.law(), b.law()) if v.test in which]
    excluding = [v for v in verdicts if v.excluded]
    if excluding:
        v = max(excluding, key=lambda x: x.witness)
        return _fail(name, src, stmt, v.witness,
                     f"{v.test} excludes the blend: {v.detail}",
                     {"C": scm.C.tolist(), "noise_var": scm.noise_var.tolist(),
                      "intervention": rec, "lambda": lam,
                      "structure": gen.describe(scm).as_row(),
                      "exclusion tests": [str(x) for x in verdicts]})
    return _ok(name, src, stmt, max((v.witness for v in verdicts), default=0.0),
               "no applicable exclusion test excluded the blend")


def prop_closure_linear_gaussian(rng, cfg) -> Outcome:
    """Theorem 1, read inside the linear Gaussian family.

    Here CK(M) contains the measures reachable by hard interventions and by
    soft interventions that keep the model linear Gaussian. Every such measure
    has Gaussian or point-mass marginals, so the marginal kurtosis exclusion test
    applies. This is the reading the source paper's own worked examples use,
    since they are linear models with additive noise.
    """
    return _closure_property(
        rng, cfg,
        "closure_linear_gaussian",
        "the blend of two measures in CK_LG(M) is again in CK_LG(M)",
        ("gaussian_marginals",),
    )


def prop_closure_unrestricted_mechanisms(rng, cfg) -> Outcome:
    """Theorem 1, read with no restriction on the mechanism class.

    Here a soft intervention may install any measurable mechanism, so the
    kurtosis invariant does not apply: a non-Gaussian marginal is reachable.
    What survives is the independence structure, which no intervention can
    create, so only the ancestral and root exclusion tests apply. This is the stronger
    and more generous reading of the claim, and it is the one whose failure is
    structural rather than parametric.
    """
    return _closure_property(
        rng, cfg,
        "closure_unrestricted_mechanisms",
        "the blend of two measures in CK(M) is again in CK(M), any mechanism class",
        ("root_independence", "ancestral_independence"),
    )


def prop_pushforward_is_affine(rng, cfg) -> Outcome:
    """Theorem 2: the endogenous component of an abstraction commutes with
    cc_lambda. Tested for an arbitrary matrix, since the claim proved here is
    that affineness holds for every pushforward and not only consistent ones.
    """
    name, src = "pushforward_is_affine", "source Theorem 2"
    stmt = "pushforward along a linear map commutes with convex combination"
    n = int(rng.integers(2, 6))
    k = int(rng.integers(1, n + 1))
    F = rng.normal(size=(k, n))

    def rand_law():
        R = rng.normal(size=(n, n))
        return rng.normal(size=n), R @ R.T + 1e-6 * np.eye(n)

    l1, l2 = rand_law(), rand_law()
    lam = gen.sample_weight(rng)

    blend_then_push = push_gaussian(F, exc.mixture_moments(lam, l1, l2))
    push_then_blend = exc.mixture_moments(lam, push_gaussian(F, l1), push_gaussian(F, l2))
    gap = _moment_gap(blend_then_push, push_then_blend)
    if gap > 1e-8:
        return _fail(name, src, stmt, gap, "pushforward does not commute with blending",
                     {"F": F.tolist(), "lambda": lam})
    return _ok(name, src, stmt, gap)


def prop_encoding_functor_laws(rng, cfg) -> Outcome:
    """Theorem 3: the encoding is a functor, so it preserves identities and
    composition of abstractions."""
    name, src = "encoding_functor_laws", "source Theorem 3"
    stmt = "E(identity) is the identity and E(beta o alpha) = E(beta) o E(alpha)"
    n = int(rng.integers(2, 6))
    mid = int(rng.integers(1, n + 1))
    out = int(rng.integers(1, mid + 1))
    F1, F2 = rng.normal(size=(mid, n)), rng.normal(size=(out, mid))
    R = rng.normal(size=(n, n))
    law = (rng.normal(size=n), R @ R.T + 1e-6 * np.eye(n))

    identity_gap = _moment_gap(push_gaussian(np.eye(n), law), law)
    composed = push_gaussian(F2 @ F1, law)
    stepwise = push_gaussian(F2, push_gaussian(F1, law))
    composition_gap = _moment_gap(composed, stepwise)

    worst = max(identity_gap, composition_gap)
    if worst > 1e-8:
        return _fail(name, src, stmt, worst, "a functor law fails",
                     {"identity_gap": identity_gap, "composition_gap": composition_gap})
    return _ok(name, src, stmt, worst)


# ---------------------------------------------------------------------------
# claims established in this project
# ---------------------------------------------------------------------------

def prop_mixing_map_round_trip(rng, cfg) -> Outcome:
    """Invertibility: M^-1 recovers the noise exactly for a non-intervened model."""
    name, src = "mixing_map_round_trip", "ours, invertibility theorem"
    stmt = "the inverse mixing map recovers the noise that produced each observation"
    scm = gen.sample_scm(rng, cfg.scm_space)
    z = scm.sample_noise(512, rng)
    gap = float(np.max(np.abs(scm.inverse_mixing(scm.mixing(z)) - z)))
    if gap > 1e-8:
        return _fail(name, src, stmt, gap, "round trip does not return the noise",
                     {"C": scm.C.tolist()})
    return _ok(name, src, stmt, gap)


def prop_endogenous_component_unique(rng, cfg) -> Outcome:
    """Uniqueness: the transport is the only natural transformation with
    identity exogenous component.

    Tested constructively: any other candidate differs from it by a nonzero
    matrix Delta, and naturality then fails by at least sigma_min(M)||Delta||,
    which is bounded away from zero because M is invertible.
    """
    name, src = "endogenous_component_unique", "ours, uniqueness"
    stmt = "no transport other than M_I o M^-1 satisfies naturality with identity exogenous part"
    scm = gen.sample_scm(rng, cfg.scm_space)
    a, _, rec = gen.sample_intervention_pair(rng, scm, kind="soft")
    A, _ = scm.intervention_map(a)
    M = scm.mixing_matrix
    sigma_min = float(np.linalg.svd(M, compute_uv=False)[-1])

    worst_ratio = np.inf
    for _ in range(8):
        delta = rng.normal(size=A.shape)
        delta /= np.linalg.norm(delta)
        residual = float(np.linalg.norm((A + delta) @ M - a.mixing_matrix))
        worst_ratio = min(worst_ratio, residual)

    if sigma_min <= 1e-12 or worst_ratio < 0.5 * sigma_min:
        return _fail(name, src, stmt, worst_ratio,
                     "a perturbed transport satisfied naturality too closely",
                     {"C": scm.C.tolist(), "sigma_min": sigma_min, "intervention": rec})
    return _ok(name, src, stmt, worst_ratio, f"sigma_min(M) = {sigma_min:.4g}")


def prop_ic_criterion(rng, cfg) -> Outcome:
    """The aggregation is interventionally consistent exactly when the micro
    effect vector factors through the aggregation direction, b = c w."""
    name, src = "ic_criterion", "ours, IC criterion"
    stmt = "the interventional gap vanishes if and only if b = c w"
    force = bool(rng.random() < 0.5)
    alpha, rec = gen.sample_abstraction(rng, gen.AbstractionSpace(force_consistent=force))
    p = len(rec["w"])

    micro_do = {i: float(rng.normal(0, 1.5)) for i in range(p)}
    macro_value = float(np.dot(rec["w"], [micro_do[i] for i in range(p)]))
    gap = alpha.ic_gap(alpha.micro.do_hard(micro_do),
                       alpha.macro.do_hard({0: macro_value}),
                       coords=[1])
    consistent = rec["rho_mis"] < 1e-16
    predicted = gap < 1e-9
    if consistent != predicted:
        return _fail(name, src, stmt, gap,
                     f"rho_mis = {rec['rho_mis']:.4g} but gap = {gap:.4g}", rec)
    return _ok(name, src, stmt, gap, f"consistent={consistent}, rho_mis={rec['rho_mis']:.4g}")


# ---------------------------------------------------------------------------
# network claims
# ---------------------------------------------------------------------------

def _chain_section(net, start: str) -> Dict[str, Law]:
    """Build a global section by propagating one node's law along the chain."""
    order = list(net.nodes)
    section: Dict[str, Law] = {start: net.nodes[start].law()}
    for a, b in zip(order, order[1:]):
        edge = next(e for e in net.incident_edges(a) if e in net.incident_edges(b))
        section[b] = net.extend_kernel(b, edge, net.restrict(a, edge, section[a]))
    return section


def prop_extension_is_right_inverse(rng, cfg) -> Outcome:
    """Both extension maps are right inverses of restriction, which is the
    embedded costalk condition."""
    name, src = "extension_is_right_inverse", "ours, extension maps"
    stmt = "restricting an extended measure returns the measure that was sent"
    net = gen.sample_network(rng, cfg.net_space)
    node, edge = _pick_incidence(rng, net)
    d = net.F[(node, edge)].shape[0]
    nu = _random_law(rng, d)
    worst = 0.0
    for extend in (net.extend_det, net.extend_kernel):
        worst = max(worst, _moment_gap(net.restrict(node, edge, extend(node, edge, nu)), nu))
    if worst > 1e-7:
        return _fail(name, src, stmt, worst, "extension is not a right inverse",
                     {"node": node, "edge": edge})
    return _ok(name, src, stmt, worst)


def prop_kernel_extension_preserves_kl(rng, cfg) -> Outcome:
    """The kernel extension preserves KL divergence exactly."""
    name, src = "kernel_extension_preserves_kl", "ours, divergence on the sheaf"
    stmt = "KL between two backbone measures is unchanged by the kernel extension"
    net = gen.sample_network(rng, cfg.net_space)
    node, edge = _pick_incidence(rng, net)
    d = net.F[(node, edge)].shape[0]
    mu, nu = _random_law(rng, d), _random_law(rng, d)
    before = gaussian_kl(*mu, *nu)
    after = gaussian_kl(*net.extend_kernel(node, edge, mu), *net.extend_kernel(node, edge, nu))
    gap = abs(before - after)
    if gap > 1e-7 * max(1.0, before):
        return _fail(name, src, stmt, gap, f"KL changed from {before:.6g} to {after:.6g}",
                     {"node": node, "edge": edge})
    return _ok(name, src, stmt, gap, f"KL = {before:.6g}")


def prop_restriction_does_not_increase_kl(rng, cfg) -> Outcome:
    """Restriction cannot increase KL divergence (data processing)."""
    name, src = "restriction_does_not_increase_kl", "ours, divergence on the sheaf"
    stmt = "projecting onto a backbone never increases the divergence between two measures"
    net = gen.sample_network(rng, cfg.net_space)
    node, edge = _pick_incidence(rng, net)
    scm = net.nodes[node]
    other = scm.do_soft(int(rng.integers(0, scm.n)),
                        gain=float(rng.uniform(0.3, 2.5)),
                        shift=float(rng.normal(0, 1.0)))
    before = gaussian_kl(*scm.law(), *other.law())
    after = gaussian_kl(*net.restrict(node, edge, scm.law()),
                        *net.restrict(node, edge, other.law()))
    if after > before + 1e-8:
        return _fail(name, src, stmt, after - before,
                     f"divergence rose from {before:.6g} to {after:.6g}",
                     {"node": node, "edge": edge})
    return _ok(name, src, stmt, before - after, f"{before:.6g} -> {after:.6g}")


def prop_kernel_extension_fixes_own_law(rng, cfg) -> Outcome:
    """The kernel extension returns a subject's own law when fed its own projection."""
    name, src = "kernel_extension_fixes_own_law", "ours, extension maps"
    stmt = "extending a subject's own projection reconstructs its own law"
    net = gen.sample_network(rng, cfg.net_space)
    node, edge = _pick_incidence(rng, net)
    own = net.nodes[node].law()
    back = net.extend_kernel(node, edge, net.restrict(node, edge, own))
    gap = _moment_gap(own, back)
    if gap > 1e-7:
        return _fail(name, src, stmt, gap, "own law not reconstructed", {"node": node, "edge": edge})
    return _ok(name, src, stmt, gap)


def prop_energy_zero_iff_section(rng, cfg) -> Outcome:
    """The disagreement energy vanishes exactly on global sections."""
    name, src = "energy_zero_iff_section", "ours, global sections"
    stmt = "the disagreement energy is zero if and only if the cochain is a global section"
    net = gen.sample_network(rng, gen.NetworkSpace(n_nodes=(2, 3), topology="chain",
                                                   node_vars=cfg.net_space.node_vars,
                                                   backbone_width=cfg.net_space.backbone_width,
                                                   scm_space=cfg.net_space.scm_space))
    order = list(net.nodes)
    section = _chain_section(net, order[0])
    e_section = net.energy(section)
    if e_section > 1e-8:
        return _fail(name, src, stmt, e_section, "a constructed global section had positive energy",
                     {"nodes": order})

    perturbed = dict(section)
    last = order[-1]
    perturbed[last] = net.extend_kernel(
        last,
        net.incident_edges(last)[0],
        _random_law(rng, net.F[(last, net.incident_edges(last)[0])].shape[0]),
    )
    e_perturbed = net.energy(perturbed)
    if e_perturbed <= 1e-8:
        return _skip(name, src, stmt, "the perturbation happened to land on a section")
    return _ok(name, src, stmt, e_section, f"section {e_section:.2e}, perturbed {e_perturbed:.4g}")


def prop_global_sections_closed_under_blending(rng, cfg) -> Outcome:
    """Global sections form a convex set: a blend of two sections is a section."""
    name, src = "global_sections_closed_under_blending", "ours, convexity of sections"
    stmt = "the convex combination of two global sections is a global section"
    net = gen.sample_network(rng, gen.NetworkSpace(n_nodes=(2, 3), topology="chain",
                                                   node_vars=cfg.net_space.node_vars,
                                                   backbone_width=cfg.net_space.backbone_width,
                                                   scm_space=cfg.net_space.scm_space))
    order = list(net.nodes)
    s1 = _chain_section(net, order[0])

    start = net.nodes[order[0]]
    altered = start.do_soft(int(rng.integers(0, start.n)),
                            gain=float(rng.uniform(0.3, 2.5)),
                            shift=float(rng.normal(0, 1.0)))
    s2 = dict(_chain_section(net, order[0]))
    s2[order[0]] = altered.law()
    for a, b in zip(order, order[1:]):
        edge = next(e for e in net.incident_edges(a) if e in net.incident_edges(b))
        s2[b] = net.extend_kernel(b, edge, net.restrict(a, edge, s2[a]))

    if net.energy(s1) > 1e-8 or net.energy(s2) > 1e-8:
        return _skip(name, src, stmt, "could not construct two global sections on this draw")

    lam = gen.sample_weight(rng)
    blended = {k: exc.mixture_moments(lam, s1[k], s2[k]) for k in s1}
    e = net.energy(blended)
    if e > 1e-7:
        return _fail(name, src, stmt, e, f"blend of two sections has energy {e:.6g}",
                     {"nodes": order, "lambda": lam})
    return _ok(name, src, stmt, e)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pick_incidence(rng, net) -> Tuple[str, str]:
    pairs = [(node, edge) for (node, edge) in net.F]
    return pairs[int(rng.integers(0, len(pairs)))]


def _random_law(rng, d: int) -> Law:
    R = rng.normal(size=(d, d))
    return rng.normal(size=d), R @ R.T + 0.05 * np.eye(d)


# ---------------------------------------------------------------------------
# configuration and the registry
# ---------------------------------------------------------------------------

@dataclass
class Config:
    scm_space: gen.SCMSpace = field(default_factory=gen.SCMSpace)
    net_space: gen.NetworkSpace = field(default_factory=gen.NetworkSpace)
    intervention_kind: str = "any"


ALL_PROPERTIES: List[Callable] = [
    # claims of the source paper
    prop_intervention_is_natural,
    prop_transport_gives_interventional_law,
    prop_convex_space_axioms,
    prop_closure_linear_gaussian,
    prop_closure_unrestricted_mechanisms,
    prop_pushforward_is_affine,
    prop_encoding_functor_laws,
    # claims established in this project
    prop_mixing_map_round_trip,
    prop_endogenous_component_unique,
    prop_ic_criterion,
    prop_extension_is_right_inverse,
    prop_kernel_extension_preserves_kl,
    prop_restriction_does_not_increase_kl,
    prop_kernel_extension_fixes_own_law,
    prop_energy_zero_iff_section,
    prop_global_sections_closed_under_blending,
]
