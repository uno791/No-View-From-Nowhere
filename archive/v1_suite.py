"""Unit tests. Run with:  python -m pytest -q   (from the code folder)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rck.abstraction import LinearAbstraction, push_gaussian
from rck.kl import gaussian_kl, grid_kl, knn_kl
from rck.scm import GeneralSCM, LinearGaussianSCM, Mechanism
from rck.sheaf import CausalNetwork


# ----- KL tools -----------------------------------------------------------------

def test_gaussian_kl_known_value():
    # KL( N(0,1) || N(1,4) ) = 0.5 * (1/4 + 1/4 - 1 + ln 4)
    expected = 0.5 * (0.25 + 0.25 - 1 + np.log(4))
    assert gaussian_kl([0], [[1]], [1], [[4]]) == pytest.approx(expected, abs=1e-12)


def test_gaussian_kl_zero_and_positive():
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    assert gaussian_kl([1, 2], S, [1, 2], S) == 0.0
    assert gaussian_kl([1, 2], S, [1, 2.5], S) > 0


def test_gaussian_kl_rejects_degenerate():
    with pytest.raises(ValueError):
        gaussian_kl([0, 0], [[1, 1], [1, 1]], [0, 0], np.eye(2))


def test_grid_kl_matches_closed_form():
    from scipy.stats import norm
    g = np.linspace(-15, 15, 30001)
    value = grid_kl(norm.pdf(g, 0, 1), norm.pdf(g, 1, 2), g[1] - g[0])
    assert value == pytest.approx(gaussian_kl([0], [[1]], [1], [[4]]), abs=1e-6)


def test_knn_kl_tracks_closed_form():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(40_000, 1))
    y = rng.normal(1, 2, size=(40_000, 1))
    assert knn_kl(x, y) == pytest.approx(gaussian_kl([0], [[1]], [1], [[4]]), abs=0.03)
    assert abs(knn_kl(x, rng.normal(0, 1, size=(40_000, 1)))) < 0.02


# ----- SCMs (RQ1) ---------------------------------------------------------------

def chain():
    return LinearGaussianSCM(C=[[0, 0, 0], [0.7, 0, 0], [0, -0.4, 0]], noise_var=[1.0, 0.5, 2.0],
                             names=["X1", "X2", "X3"])


def test_law_is_pushforward_of_noise():
    scm, rng = chain(), np.random.default_rng(1)
    x = scm.sample(400_000, rng)
    assert np.allclose(np.cov(x.T), scm.law()[1], atol=0.02)


def test_round_trip_and_naturality():
    scm, rng = chain(), np.random.default_rng(2)
    z = scm.sample_noise(1000, rng)
    assert np.allclose(scm.inverse_mixing(scm.mixing(z)), z, atol=1e-12)
    for intervened in (scm.do_hard({1: 2.0}), scm.do_soft(2, coeffs={1: 1.3}, gain=0.5, shift=-1.0)):
        A, b = scm.intervention_map(intervened)
        assert np.allclose(scm.mixing(z) @ A.T + b, intervened.mixing(z), atol=1e-12)


def test_transport_reproduces_interventional_law():
    scm = chain()
    intervened = scm.do_soft(1, coeffs={0: -2.0}, shift=0.7)
    A, b = scm.intervention_map(intervened)
    m, S = scm.law()
    assert gaussian_kl(A @ m + b, A @ S @ A.T, *intervened.law()) < 1e-12


def test_hard_intervened_model_cannot_be_a_source():
    scm = chain().do_hard({0: 1.0})
    assert not scm.is_invertible
    with pytest.raises(ValueError):
        scm.intervention_map(chain())


def test_soft_intervention_may_not_change_parents():
    scm = chain()
    with pytest.raises(ValueError):
        scm.do_soft(2, coeffs={0: 1.0})          # X1 is not a parent of X3
    with pytest.raises(ValueError):
        scm.do_soft(1, coeffs={0: 0.0})          # would delete the edge X1 -> X2


def test_general_scm_classes_invert():
    rng = np.random.default_rng(3)
    mechs = [Mechanism("anm"),
             Mechanism("pnl", [0], f=lambda p: p[:, 0], g=np.arcsinh, g_inv=np.sinh),
             Mechanism("lsnm", [0, 1], f=lambda p: p[:, 0] * p[:, 1], h=lambda p: 1 + p[:, 1] ** 2)]
    scm = GeneralSCM(mechs, ["a", "b", "c"])
    z = scm.sample_noise(5000, rng)
    assert np.allclose(scm.inverse_mixing(scm.mixing(z)), z, atol=1e-9)


# ----- abstraction (RQ2) ------------------------------------------------------------

def study_pair(b1, b2):
    micro = LinearGaussianSCM(C=[[0, 0, 0], [0, 0, 0], [b1, b2, 0]], noise_var=[1.0, 0.64, 0.25],
                              names=["H1", "H2", "Mk"])
    F = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _, cov = push_gaussian(F, micro.law())
    c = cov[0, 1] / cov[0, 0]
    macro = LinearGaussianSCM(C=[[0, 0], [c, 0]], noise_var=[cov[0, 0], cov[1, 1] - c * c * cov[0, 0]],
                              names=["T", "Mk"])
    return LinearAbstraction(micro, macro, F)


@pytest.mark.parametrize("b, consistent", [((0.6, 0.6), True), ((0.9, 0.2), False), ((-1.0, -1.0), True)])
def test_interventional_consistency_iff_effects_factor_through_total(b, consistent):
    alpha = study_pair(*b)
    assert alpha.naturality_residual() < 1e-12 and alpha.noise_is_preserved()
    assert alpha.ic_gap(alpha.micro, alpha.macro) < 1e-12            # observational agreement always
    gap = alpha.ic_gap(alpha.micro.do_hard({0: 2.0, 1: 0.0}), alpha.macro.do_hard({0: 2.0}), coords=[1])
    assert (gap < 1e-12) == consistent


def test_pushforward_is_affine_for_any_matrix():
    rng = np.random.default_rng(4)
    for _ in range(50):
        F = rng.normal(size=(2, 4))
        R1, R2 = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        S1, S2, lam = R1 @ R1.T, R2 @ R2.T, rng.random()
        assert np.allclose(F @ (lam * S1 + (1 - lam) * S2) @ F.T,
                           lam * F @ S1 @ F.T + (1 - lam) * F @ S2 @ F.T)


# ----- sheaf (RQ3) --------------------------------------------------------------------

def random_network(rng):
    def random_scm(n, tag):
        C = np.tril(rng.normal(size=(n, n)) * 0.6, k=-1)
        C[C == 0] = 0.0
        return LinearGaussianSCM(C, rng.uniform(0.3, 1.5, size=n), [f"{tag}{i}" for i in range(n)])
    net = CausalNetwork()
    net.add_node("P", random_scm(4, "P"))
    net.add_node("Q", random_scm(3, "Q"))
    backbone = LinearGaussianSCM(C=[[0.0]], noise_var=[1.0], names=["E"])
    net.add_edge("E", backbone, "P", rng.normal(size=(1, 4)), "Q", rng.normal(size=(1, 3)))
    return net


def test_extension_is_right_inverse_and_preserves_kl():
    rng = np.random.default_rng(5)
    for _ in range(25):
        net = random_network(rng)
        mu = (rng.normal(size=1), np.array([[rng.uniform(0.2, 3)]]))
        nu = (rng.normal(size=1), np.array([[rng.uniform(0.2, 3)]]))
        for extend in (net.extend_det, net.extend_kernel):
            back = net.restrict("Q", "E", extend("Q", "E", mu))
            assert np.allclose(back[0], mu[0]) and np.allclose(back[1], mu[1])
        e_mu, e_nu = net.extend_kernel("Q", "E", mu), net.extend_kernel("Q", "E", nu)
        assert gaussian_kl(*e_mu, *e_nu) == pytest.approx(gaussian_kl(*mu, *nu), rel=1e-9, abs=1e-12)


def test_restriction_never_increases_kl():
    rng = np.random.default_rng(6)
    for _ in range(25):
        net = random_network(rng)
        p = net.nodes["P"]
        law1, law2 = p.law(), p.do_soft(0, gain=1.7, shift=0.4).law()
        r1, r2 = net.restrict("P", "E", law1), net.restrict("P", "E", law2)
        assert gaussian_kl(*r1, *r2) <= gaussian_kl(*law1, *law2) + 1e-12


def test_energy_zero_iff_projections_agree():
    rng = np.random.default_rng(7)
    net = random_network(rng)
    law_p = net.nodes["P"].law()
    target = net.restrict("P", "E", law_p)
    section = {"P": law_p, "Q": net.extend_kernel("Q", "E", target)}
    assert net.energy(section) < 1e-12
    section["Q"] = net.nodes["Q"].do_soft(0, gain=3.0).law()
    assert net.energy(section) > 1e-3
