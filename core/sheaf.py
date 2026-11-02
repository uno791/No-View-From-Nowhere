"""
Network sheaf and cosheaf of causal knowledge (the code version of RQ3).

A network has
    nodes   subjects, each holding a non-intervened linear Gaussian SCM
    edges   shared backbone models, each a small linear Gaussian SCM
    F       one restriction matrix for every (node, edge) incidence

Sheaf direction (node -> edge), "project":
    restrict   pushforward along F:  N(m, S) -> N(F m, F S F^T)

Cosheaf direction (edge -> node), "embed". Two extension maps are provided.
Both are right inverses of the restriction: restrict(extend(nu)) = nu, which
is the embedded costalk condition of the proposal (equation 4.4).

    extend_det      deterministic: x -> mu + K (x - F mu). Output is a
                    degenerate Gaussian living on a low-dimensional subspace.
    extend_kernel   probabilistic: the receiving subject fills in everything
                    the backbone does not determine using its own
                    observational model. Output is a full Gaussian, so KL
                    divergences at the node are finite.

K is the regression matrix of the receiving subject,
    K = S F^T (F S F^T)^-1,   with S the subject's observational covariance,
so K x is the subject's best guess of its own variables given the backbone
value x.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .abstraction import push_gaussian
from .kl import symmetric_gaussian_kl
from .scm import LinearGaussianSCM

Law = Tuple[np.ndarray, np.ndarray]


class CausalNetwork:
    def __init__(self):
        self.nodes: Dict[str, LinearGaussianSCM] = {}
        self.edges: Dict[str, Tuple[LinearGaussianSCM, Tuple[str, str]]] = {}
        self.F: Dict[Tuple[str, str], np.ndarray] = {}

    # ----- construction -------------------------------------------------------
    def add_node(self, name: str, scm: LinearGaussianSCM):
        self.nodes[name] = scm

    def add_edge(self, name: str, backbone: LinearGaussianSCM, a: str, Fa, b: str, Fb):
        self.edges[name] = (backbone, (a, b))
        self.F[(a, name)] = np.atleast_2d(np.asarray(Fa, dtype=float))
        self.F[(b, name)] = np.atleast_2d(np.asarray(Fb, dtype=float))

    def incident_edges(self, node: str) -> List[str]:
        return [e for e, (_, ends) in self.edges.items() if node in ends]

    # ----- sheaf: restriction maps ---------------------------------------------
    def restrict(self, node: str, edge: str, law: Law) -> Law:
        return push_gaussian(self.F[(node, edge)], law)

    # ----- cosheaf: extension maps ---------------------------------------------
    def regression_matrix(self, node: str, edge: str) -> np.ndarray:
        F = self.F[(node, edge)]
        _, S = self.nodes[node].law()
        return S @ F.T @ np.linalg.inv(F @ S @ F.T)

    def extend_det(self, node: str, edge: str, edge_law: Law) -> Law:
        F = self.F[(node, edge)]
        mu, _ = self.nodes[node].law()
        K = self.regression_matrix(node, edge)
        m, s = edge_law
        return mu + K @ (m - F @ mu), K @ s @ K.T

    def extend_kernel(self, node: str, edge: str, edge_law: Law) -> Law:
        F = self.F[(node, edge)]
        mu, S = self.nodes[node].law()
        K = self.regression_matrix(node, edge)
        m, s = edge_law
        residual = S - K @ F @ S          # covariance of the node given the backbone value
        return mu + K @ (m - F @ mu), K @ s @ K.T + residual

    # ----- relative causal knowledge along a path --------------------------------
    def rck(self, path: Sequence[str], law: Law, mode: str = "kernel") -> Law:
        """Transport a measure along path = [node, edge, node, edge, ..., node].

        Alternates restriction (node -> edge) and extension (edge -> node),
        which is equation (4.5) of the proposal.
        """
        extend = self.extend_kernel if mode == "kernel" else self.extend_det
        current = law
        for k in range(0, len(path) - 2, 2):
            src, edge, dst = path[k], path[k + 1], path[k + 2]
            current = extend(dst, edge, self.restrict(src, edge, current))
        return current

    def path_matrix(self, path: Sequence[str]) -> np.ndarray:
        """Composite matrix of the deterministic transport along a path (zero-mean case)."""
        T = None
        for k in range(0, len(path) - 2, 2):
            src, edge, dst = path[k], path[k + 1], path[k + 2]
            step = self.regression_matrix(dst, edge) @ self.F[(src, edge)]
            T = step if T is None else step @ T
        return T

    # ----- global sections ----------------------------------------------------------
    def edge_disagreement(self, cochain: Dict[str, Law]) -> Dict[str, float]:
        """Symmetric KL between the two projections arriving on each edge."""
        out = {}
        for e, (_, (a, b)) in self.edges.items():
            pa = self.restrict(a, e, cochain[a])
            pb = self.restrict(b, e, cochain[b])
            out[e] = symmetric_gaussian_kl(pa[0], pa[1], pb[0], pb[1])
        return out

    def energy(self, cochain: Dict[str, Law]) -> float:
        """Total disagreement. Zero exactly when the cochain is a global section."""
        return float(sum(self.edge_disagreement(cochain).values()))

    def stacked_restriction(self, node: str) -> np.ndarray:
        """All restriction matrices leaving a node, stacked. Full row rank is the
        sufficient condition of the global section theorem."""
        return np.vstack([self.F[(node, e)] for e in self.incident_edges(node)])


def best_reachable_with_gains(scm: LinearGaussianSCM, rows: np.ndarray, fixed_target: float) -> float:
    """Largest variance of rows[0] @ X reachable by noise gains alone while the
    variance of rows[1] @ X is held at fixed_target. A small linear programme in u = d^2 >= 0."""
    from scipy.optimize import linprog
    M0 = np.linalg.inv(np.eye(scm.n) - scm.C)
    A = (np.atleast_2d(rows) @ M0) ** 2 * scm.noise_var
    res = linprog(c=-A[0], A_eq=A[1:2], b_eq=[fixed_target], bounds=[(0, None)] * scm.n)
    return float(-res.fun)


def soft_intervention_matching_edge_variances(scm: LinearGaussianSCM, rows: np.ndarray,
                                              targets: np.ndarray) -> LinearGaussianSCM:
    """Search the full soft-intervention family of a linear Gaussian SCM (every
    existing coefficient and every noise gain may change, no edge may be added or
    removed) for a model whose projected variances equal the targets.

    Solved as a small nonlinear least squares problem started at the given model,
    so the answer is a soft intervention close to where the subject already is.
    """
    from scipy.optimize import least_squares
    rows = np.atleast_2d(rows)
    targets = np.asarray(targets, dtype=float)
    edges = [(i, j) for i in range(scm.n) for j in range(scm.n) if scm.C[i, j] != 0]
    start = np.concatenate([[scm.C[i, j] for i, j in edges], np.log(np.abs(scm.D))])

    def build(theta):
        C = np.zeros_like(scm.C)
        for value, (i, j) in zip(theta[:len(edges)], edges):
            C[i, j] = value
        return LinearGaussianSCM(C, scm.noise_var, scm.names, D=np.exp(theta[len(edges):]), c=scm.c)

    def residual(theta):
        _, S = build(theta).law()
        return np.array([float((r @ S @ r)) for r in rows]) - targets

    sol = least_squares(residual, start, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    model = build(sol.x)
    if any(model.C[i, j] == 0 for i, j in edges):
        raise ValueError("The search removed a parent; that would not be a soft intervention.")
    return model
