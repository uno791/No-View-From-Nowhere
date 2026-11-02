"""
RQ3 experiment. Can sheaves and cosheaves of causal knowledge formalise
transfer across a network of subjects, and when does a global section exist?

The network is the three-node chain of the proposal, A - X - B - Y - C,
later augmented with a direct edge W between A and C to test path dependence.

Story used for the variables (synthetic, chosen for readability). Three
people hold different causal models of the same student's mathematics:

    A  school teacher       A1 algebra mastery, A2 geometry mastery, A3 final school mark
    B  university lecturer  B1 algebra diagnostic, B2 functions diagnostic,
                            B3 calculus mark, B4 linear algebra mark, B5 statistics mark
    C  graduate employer    C1 numerical test, C2 case interview, C3 first-year job rating

    X  shared backbone of A and B: "school mathematics foundation"
    Y  shared backbone of B and C: "quantitative competence"
    W  shared backbone of A and C: "overall attainment" (added in part D)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.abstraction import push_gaussian                           # noqa: E402
from core.kl import gaussian_kl                                      # noqa: E402
from core.reporting import FIGURES_DIR, save                         # noqa: E402
from core.scm import LinearGaussianSCM                               # noqa: E402
from core.sheaf import (CausalNetwork, best_reachable_with_gains,     # noqa: E402
                       soft_intervention_matching_edge_variances)

SEED = 20260919


def unit_variance_row(weights, scm):
    """Scale a weight row so that the backbone variable has variance 1 under the
    subject's observational law. This makes the observational laws of both
    subjects project onto the same backbone law N(0, 1)."""
    w = np.asarray(weights, dtype=float)
    _, S = scm.law()
    return w / np.sqrt(w @ S @ w)


def build_network(with_direct_edge=False):
    A = LinearGaussianSCM(C=[[0, 0, 0], [0.5, 0, 0], [0.7, 0.6, 0]],
                          noise_var=[1.0, 0.75, 0.5], names=["A1", "A2", "A3"])
    B = LinearGaussianSCM(C=[[0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0],
                             [0, 0, 0.6, 0, 0], [0, 0, 0.3, 0.5, 0]],
                          noise_var=[1.0, 0.8, 0.6, 0.7, 0.5], names=["B1", "B2", "B3", "B4", "B5"])
    Cm = LinearGaussianSCM(C=[[0, 0, 0], [0.8, 0, 0], [0.5, 0.4, 0]],
                           noise_var=[1.0, 0.5, 0.6], names=["C1", "C2", "C3"])
    backbone = lambda name: LinearGaussianSCM(C=[[0.0]], noise_var=[1.0], names=[name])

    net = CausalNetwork()
    net.add_node("A", A)
    net.add_node("B", B)
    net.add_node("C", Cm)
    net.add_edge("X", backbone("X"), "A", unit_variance_row([0.6, 0.4, 0.0], A),
                 "B", unit_variance_row([0.5, 0.3, 0.2, 0.0, 0.0], B))
    net.add_edge("Y", backbone("Y"), "B", unit_variance_row([0.0, 0.0, 0.4, 0.3, 0.3], B),
                 "C", unit_variance_row([0.5, 0.5, 0.0], Cm))
    if with_direct_edge:
        net.add_edge("W", backbone("W"), "A", unit_variance_row([0.0, 0.3, 0.7], A),
                     "C", unit_variance_row([0.0, 0.4, 0.6], Cm))
    return net


def lift(scm, F, gain, shift):
    """Micro intervention corresponding to the backbone intervention Z -> gain * Z + shift.

    Every noise gain is multiplied by `gain` (so every variable, and hence the
    backbone variable, is scaled by it) and the first root is shifted by the
    amount that moves the backbone variable by `shift`. Parent sets are untouched.
    """
    out = scm
    for i in range(scm.n):
        out = out.do_soft(i, gain=gain)
    M0 = np.linalg.inv(np.eye(scm.n) - scm.C)
    return out.do_soft(0, shift=shift / float((F @ M0[:, 0]).item()))


def soft_interventions(net):
    """Each subject works with its own soft-intervened model (a 0-cochain)."""
    return {
        "A": net.nodes["A"].do_soft(1, coeffs={0: 1.2}),
        "B": net.nodes["B"].do_soft(2, coeffs={1: -0.2}).do_soft(3, coeffs={2: 0.9}),
        "C": net.nodes["C"].do_soft(1, coeffs={0: 0.1}),
    }


# ---------------------------------------------------------------------------

def part_a(net):
    """The sheaf is well formed: observational agreement and interventional consistency."""
    out = {}
    obs = {name: scm.law() for name, scm in net.nodes.items()}
    out["observational_energy"] = net.energy(obs)
    gaps = {}
    for (node, edge), F in net.F.items():
        backbone = net.edges[edge][0]
        for gain, shift in ((1.0, 0.0), (1.7, 0.0), (0.6, -1.3)):
            macro = backbone.do_soft(0, gain=gain, shift=shift)
            micro = lift(net.nodes[node], F, gain, shift)
            pm, pS = push_gaussian(F, micro.law())
            qm, qS = macro.law()
            gaps[f"{node}-{edge} gain={gain} shift={shift}"] = gaussian_kl(pm, pS, qm, qS)
    out["ic_gaps"] = gaps
    out["max_ic_gap"] = max(gaps.values())
    out["stacked_rank"] = {n: int(np.linalg.matrix_rank(net.stacked_restriction(n))) for n in net.nodes}
    out["incident_edges"] = {n: len(net.incident_edges(n)) for n in net.nodes}
    return out


def part_b(net):
    """Relative causal knowledge along the chain A -> B -> C."""
    out = {}
    laws = {k: v.law() for k, v in soft_interventions(net).items()}
    chi_A = laws["A"]

    # B1. Deterministic extension reproduces the closed form of the proposal.
    F_AX, K_BX = net.F[("A", "X")], net.regression_matrix("B", "X")
    rck_AB_det = net.rck(["A", "X", "B"], chi_A, mode="det")
    formula = K_BX @ F_AX @ chi_A[1] @ F_AX.T @ K_BX.T
    out["det_matches_formula"] = float(np.max(np.abs(rck_AB_det[1] - formula)))
    out["det_rank"] = int(np.linalg.matrix_rank(rck_AB_det[1]))
    out["right_inverse_error"] = float(np.max(np.abs(net.F[("B", "X")] @ K_BX - np.eye(1))))

    # B2. Embedded costalk condition: what B receives projects back onto what A sent.
    sent = net.restrict("A", "X", chi_A)
    for mode in ("det", "kernel"):
        received = net.rck(["A", "X", "B"], chi_A, mode=mode)
        back = net.restrict("B", "X", received)
        out[f"costalk_kl_{mode}"] = gaussian_kl(back[0], back[1], sent[0], sent[1])

    # B3. Kernel extension gives a full measure on B's variables.
    rck_AB = net.rck(["A", "X", "B"], chi_A, mode="kernel")
    out["kernel_rank"] = int(np.linalg.matrix_rank(rck_AB[1]))
    # The kernel extension of B's own observational backbone law is B's observational law.
    own = net.extend_kernel("B", "X", net.restrict("B", "X", net.nodes["B"].law()))
    out["kernel_fixes_own_law"] = float(np.max(np.abs(own[1] - net.nodes["B"].law()[1])))

    # B4. KL is preserved by the extension and can only shrink under restriction.
    mu, nu = sent, net.restrict("B", "X", laws["B"])
    kl_edge = gaussian_kl(mu[0], mu[1], nu[0], nu[1])
    e_mu, e_nu = net.extend_kernel("B", "X", mu), net.extend_kernel("B", "X", nu)
    out["kl_on_edge"] = kl_edge
    out["kl_after_extension"] = gaussian_kl(e_mu[0], e_mu[1], e_nu[0], e_nu[1])
    obsB = net.nodes["B"].law()
    out["kl_node_level"] = gaussian_kl(laws["B"][0], laws["B"][1], obsB[0], obsB[1])
    pB, pO = net.restrict("B", "X", laws["B"]), net.restrict("B", "X", obsB)
    out["kl_node_level_projected"] = gaussian_kl(pB[0], pB[1], pO[0], pO[1])
    # How far is "A as seen by B" from B's own observational knowledge?
    out["kl_A_seen_by_B_vs_B"] = gaussian_kl(rck_AB[0], rck_AB[1], obsB[0], obsB[1])
    bo = net.restrict("B", "X", obsB)
    out["kl_same_on_backbone"] = gaussian_kl(sent[0], sent[1], bo[0], bo[1])

    # B5. Two hops: A seen by C, through B.
    rck_AC = net.rck(["A", "X", "B", "Y", "C"], chi_A, mode="kernel")
    out["rck_AC_cov"] = rck_AC[1].tolist()
    out["rck_AB_cov_diag"] = np.diag(rck_AB[1]).tolist()
    out["sent_variance_on_X"] = float(sent[1][0, 0])
    out["arriving_variance_on_Y"] = float(net.restrict("B", "Y", rck_AB)[1][0, 0])
    return out


def part_c(net):
    """Global sections."""
    out = {}
    soft = soft_interventions(net)
    cochain = {k: v.law() for k, v in soft.items()}

    # C1. The subjects' own soft interventions do not agree on the backbones.
    one_cochain = {e: {n: float(net.restrict(n, e, cochain[n])[1][0, 0]) for n in ends}
                   for e, (_, ends) in net.edges.items()}
    out["edge_variances_before"] = one_cochain
    out["disagreement_before"] = net.edge_disagreement(cochain)
    out["energy_before"] = net.energy(cochain)

    # C2. Global sections INSIDE the causal knowledge.
    # C2a. The end subjects adapt to B. Each has a single backbone to match, and
    # rescaling all of its noise gains by one factor always reaches the target,
    # because the projected variance scales with the square of that factor.
    adapted = dict(cochain)
    out["leaf_gain"] = {}
    for leaf, e in (("A", "X"), ("C", "Y")):
        factor = float(np.sqrt(one_cochain[e]["B"] / one_cochain[e][leaf]))
        model = soft[leaf]
        for i in range(model.n):
            model = model.do_soft(i, gain=factor * model.D[i])
        adapted[leaf] = model.law()
        out["leaf_gain"][leaf] = factor
    out["energy_leaves_adapt"] = net.energy(adapted)

    # C2b. B adapts to both neighbours at once. Two backbones must be matched together.
    targets = [one_cochain["X"]["A"], one_cochain["Y"]["C"]]
    rows = np.vstack([net.F[("B", "X")], net.F[("B", "Y")]])
    # With noise gains alone this is impossible: holding the Y target, the X variance is capped.
    out["gains_only_cap_on_X"] = best_reachable_with_gains(net.nodes["B"], rows, targets[1])
    out["gains_only_target_on_X"] = targets[0]
    # With the full soft-intervention family (coefficients and gains) it is possible.
    B_new = soft_intervention_matching_edge_variances(soft["B"], rows, targets)
    agreed = dict(cochain)
    agreed["B"] = B_new.law()
    out["B_new_coefficients"] = {f"{B_new.names[i]}<-{B_new.names[j]}": float(B_new.C[i, j])
                                 for i in range(5) for j in range(5) if B_new.C[i, j] != 0}
    out["B_new_gains"] = B_new.D.tolist()
    out["edge_variances_after"] = {e: {n: float(net.restrict(n, e, agreed[n])[1][0, 0]) for n in ends}
                                   for e, (_, ends) in net.edges.items()}
    out["energy_after"] = net.energy(agreed)

    # C3. Sufficient condition: full row rank of the stacked restriction map at every node.
    # Then ANY backbone data extend to a global section. Constructive check with arbitrary targets.
    wanted = {"X": 2.0, "Y": 0.3}
    section = {}
    for n, scm in net.nodes.items():
        G = net.stacked_restriction(n)
        D = np.diag([wanted[e] for e in net.incident_edges(n)])
        Gp = np.linalg.pinv(G)
        P = np.eye(scm.n) - Gp @ G                       # projector onto the kernel of G
        section[n] = (np.zeros(scm.n), Gp @ D @ Gp.T + P @ scm.law()[1] @ P.T)
    out["arbitrary_targets"] = wanted
    out["arbitrary_section_energy"] = net.energy(section)
    out["arbitrary_section_edge_variances"] = {
        e: {n: float(net.restrict(n, e, section[n])[1][0, 0]) for n in ends}
        for e, (_, ends) in net.edges.items()}
    out["arbitrary_section_min_eig"] = float(min(np.linalg.eigvalsh(s[1]).min() for s in section.values()))

    # C4. Global sections form a convex set: blending two sections node by node
    # gives mixtures whose backbone projections still agree component by component.
    lam = 0.35
    sec1, sec2 = agreed, section
    worst = 0.0
    for e, (_, (a, b)) in net.edges.items():
        for sec in (sec1, sec2):
            pa, pb = net.restrict(a, e, sec[a]), net.restrict(b, e, sec[b])
            worst = max(worst, abs(float(pa[1][0, 0] - pb[1][0, 0])), abs(float(pa[0][0] - pb[0][0])))
    out["blend_weight"] = lam
    out["blend_component_mismatch"] = worst

    # C5. When the rank condition fails. If B's two restriction rows are parallel,
    # F_BY = kappa * F_BX, then every measure on B projects to backbone variances
    # in the fixed ratio kappa^2, so backbone data outside that ratio cannot be matched.
    kappa = 0.5
    F_BX = net.F[("B", "X")]
    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(200):
        R = rng.normal(size=(5, 5))
        S = R @ R.T
        ratios.append(float(((kappa * F_BX) @ S @ (kappa * F_BX).T / (F_BX @ S @ F_BX.T)).item()))
    out["parallel_rank"] = int(np.linalg.matrix_rank(np.vstack([F_BX, kappa * F_BX])))
    out["parallel_ratio_min"], out["parallel_ratio_max"] = min(ratios), max(ratios)
    out["parallel_kappa_squared"] = kappa ** 2

    figure_rq3(one_cochain, out["edge_variances_after"])
    return out


def part_d():
    """Path dependence: add the direct edge W between A and C."""
    net = build_network(with_direct_edge=True)
    chi_A = soft_interventions(net)["A"].law()
    direct, via_B = ["A", "W", "C"], ["A", "X", "B", "Y", "C"]
    out = {}
    T_direct, T_via = net.path_matrix(direct), net.path_matrix(via_B)
    out["T_direct"], out["T_via_B"] = T_direct.tolist(), T_via.tolist()
    out["path_matrix_difference"] = float(np.linalg.norm(T_direct - T_via))
    r_direct, r_via = net.rck(direct, chi_A), net.rck(via_B, chi_A)
    out["cov_direct_diag"], out["cov_via_B_diag"] = np.diag(r_direct[1]).tolist(), np.diag(r_via[1]).tolist()
    out["kl_direct_vs_via_B"] = gaussian_kl(r_direct[0], r_direct[1], r_via[0], r_via[1])
    out["kl_via_B_vs_direct"] = gaussian_kl(r_via[0], r_via[1], r_direct[0], r_direct[1])
    # Control: two copies of the same path agree exactly.
    out["kl_same_path"] = gaussian_kl(r_via[0], r_via[1], *net.rck(via_B, chi_A))
    # The observational cochain is still a global section on the network with a cycle.
    out["observational_energy_with_cycle"] = net.energy({n: s.law() for n, s in net.nodes.items()})
    out["stacked_rank_with_cycle"] = {n: int(np.linalg.matrix_rank(net.stacked_restriction(n))) for n in net.nodes}
    return out


def figure_rq3(before, after):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    colours = {"A": "#2166ac", "B": "#4d4d4d", "C": "#b2182b"}
    for ax, data, title in ((axes[0], before, "(a) subjects' own soft interventions"),
                            (axes[1], after, "(b) after B adapts: a global section")):
        pos, ticks = 0, []
        for e, sides in data.items():
            for node, value in sides.items():
                ax.bar(pos, value, color=colours[node], width=0.8)
                ax.text(pos, value + 0.03, f"{value:.3f}", ha="center", fontsize=7.5)
                ticks.append((pos, f"{node} on ${e}$"))
                pos += 1
            pos += 0.7
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("variance projected onto the backbone")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "rq3_global_section.pdf"))
    plt.close(fig)


def main():
    net = build_network()
    results = {"seed": SEED, "part_a": part_a(net), "part_b": part_b(net),
               "part_c": part_c(net), "part_d": part_d()}
    A, B, C, D = (results[k] for k in ("part_a", "part_b", "part_c", "part_d"))
    macros = {
        "RQthreeObsEnergy": A["observational_energy"], "RQthreeMaxIC": A["max_ic_gap"],
        "RQthreeDetFormula": B["det_matches_formula"], "RQthreeRightInv": B["right_inverse_error"],
        "RQthreeCostalkDet": B["costalk_kl_det"], "RQthreeCostalkKer": B["costalk_kl_kernel"],
        "RQthreeDetRank": str(B["det_rank"]), "RQthreeKerRank": str(B["kernel_rank"]),
        "RQthreeOwnLaw": B["kernel_fixes_own_law"],
        "RQthreeKLEdge": B["kl_on_edge"], "RQthreeKLExt": B["kl_after_extension"],
        "RQthreeKLNode": B["kl_node_level"], "RQthreeKLNodeProj": B["kl_node_level_projected"],
        "RQthreeSeenByB": B["kl_A_seen_by_B_vs_B"], "RQthreeSeenBackbone": B["kl_same_on_backbone"],
        "RQthreeSentVar": B["sent_variance_on_X"], "RQthreeArriveVar": B["arriving_variance_on_Y"],
        "RQthreeVarXA": C["edge_variances_before"]["X"]["A"], "RQthreeVarXB": C["edge_variances_before"]["X"]["B"],
        "RQthreeVarYB": C["edge_variances_before"]["Y"]["B"], "RQthreeVarYC": C["edge_variances_before"]["Y"]["C"],
        "RQthreeDisX": C["disagreement_before"]["X"], "RQthreeDisY": C["disagreement_before"]["Y"],
        "RQthreeEnergyBefore": C["energy_before"], "RQthreeEnergyAfter": C["energy_after"],
        "RQthreeArbEnergy": C["arbitrary_section_energy"], "RQthreeArbMinEig": C["arbitrary_section_min_eig"],
        "RQthreeBlendMismatch": C["blend_component_mismatch"],
        "RQthreeParRatioMin": C["parallel_ratio_min"], "RQthreeParRatioMax": C["parallel_ratio_max"],
        "RQthreePathDiff": D["path_matrix_difference"],
        "RQthreeKLDirectVia": D["kl_direct_vs_via_B"], "RQthreeKLViaDirect": D["kl_via_B_vs_direct"],
        "RQthreeKLSamePath": D["kl_same_path"], "RQthreeCycleObsEnergy": D["observational_energy_with_cycle"],
    }
    macros["RQthreeEnergyLeaves"] = C["energy_leaves_adapt"]
    macros["RQthreeLeafGainA"] = C["leaf_gain"]["A"]
    macros["RQthreeLeafGainC"] = C["leaf_gain"]["C"]
    macros["RQthreeGainCap"] = C["gains_only_cap_on_X"]
    macros["RQthreeGainTarget"] = C["gains_only_target_on_X"]
    save("rq3", results, macros)
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
