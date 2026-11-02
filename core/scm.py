"""
Structural causal models as "noise in, causal variables out" maps.

This file is the code version of RQ1. An SCM is stored in the form the
theory uses: an exogenous probability space (the noise), a mixing map M
that turns noise into endogenous values, and the endogenous law obtained
by pushing the noise through M. In the categorical language this is the
functor  I -> Prob  that sends the single arrow of I to M.

Two classes are provided.

LinearGaussianSCM   every quantity is available in closed form (matrices),
                    so KL divergences are exact.
GeneralSCM          arbitrary mechanisms from the three invertible model
                    classes of the proposal (additive noise, post-nonlinear,
                    location-scale). Works on samples.

Variables are always listed in a topological order of the DAG, so parents
come before children.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Linear Gaussian SCM
# ---------------------------------------------------------------------------

class LinearGaussianSCM:
    """X_i = sum_j C[i, j] X_j + D[i] Z_i + c[i],   Z ~ N(0, diag(noise_var)).

    C          strictly lower triangular matrix of causal coefficients
    D          noise gains (1 for an untouched variable, 0 after a hard intervention)
    c          constant shifts (0 for an untouched variable)
    noise_var  variances of the independent exogenous variables

    Interventions never touch noise_var. That is the code form of the statement
    "the exogenous component of an intervention morphism is the identity".
    """

    def __init__(self, C, noise_var, names: Sequence[str],
                 D=None, c=None, hard: Optional[Dict[int, float]] = None):
        self.C = np.asarray(C, dtype=float)
        n = self.C.shape[0]
        if self.C.shape != (n, n) or np.any(np.triu(self.C) != 0):
            raise ValueError("C must be square and strictly lower triangular "
                             "(variables in topological order).")
        self.noise_var = np.asarray(noise_var, dtype=float)
        self.names = list(names)
        self.D = np.ones(n) if D is None else np.asarray(D, dtype=float)
        self.c = np.zeros(n) if c is None else np.asarray(c, dtype=float)
        self.hard = dict(hard or {})
        self.n = n

    # ----- the mixing map M (noise -> endogenous values) -------------------
    @property
    def mixing_matrix(self) -> np.ndarray:
        """Linear part of M:  (I - C)^-1 diag(D)."""
        return np.linalg.solve(np.eye(self.n) - self.C, np.diag(self.D))

    @property
    def offset(self) -> np.ndarray:
        """Constant part of M:  (I - C)^-1 c."""
        return np.linalg.solve(np.eye(self.n) - self.C, self.c)

    def mixing(self, z: np.ndarray) -> np.ndarray:
        """Apply M to noise samples z of shape (N, n)."""
        return z @ self.mixing_matrix.T + self.offset

    @property
    def is_invertible(self) -> bool:
        return bool(np.all(self.D != 0))

    def inverse_mixing(self, x: np.ndarray) -> np.ndarray:
        """Apply M^-1 to endogenous samples: recover the noise, one equation at a time.

        z_i = (x_i - sum_j C[i, j] x_j - c_i) / D_i.
        Only possible when every D_i is nonzero, which fails after a hard intervention.
        """
        if not self.is_invertible:
            raise ValueError("Mixing map is not invertible: a hard intervention destroyed a noise input.")
        return (x @ (np.eye(self.n) - self.C).T - self.c) / self.D

    # ----- the probability spaces -------------------------------------------
    def sample_noise(self, N: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(N, self.n)) * np.sqrt(self.noise_var)

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        return self.mixing(self.sample_noise(N, rng))

    def law(self):
        """Endogenous measure chi = M(zeta), a Gaussian N(mean, cov)."""
        M = self.mixing_matrix
        return self.offset.copy(), M @ np.diag(self.noise_var) @ M.T

    # ----- interventions -----------------------------------------------------
    def do_hard(self, assignment: Dict[int, float]) -> "LinearGaussianSCM":
        """Hard intervention do(X_i = value): mechanism replaced by a constant."""
        C, D, c = self.C.copy(), self.D.copy(), self.c.copy()
        for i, value in assignment.items():
            C[i, :] = 0.0       # incoming edges removed
            D[i] = 0.0          # own noise disconnected
            c[i] = value        # constant mechanism
        return LinearGaussianSCM(C, self.noise_var, self.names, D, c, {**self.hard, **assignment})

    def do_soft(self, i: int, coeffs: Optional[Dict[int, float]] = None,
                gain: Optional[float] = None, shift: Optional[float] = None) -> "LinearGaussianSCM":
        """Soft intervention on X_i: change its mechanism, keep its parent set.

        coeffs  new coefficients for EXISTING parents only, as {parent_index: value}
        gain    new noise gain (must stay nonzero)
        shift   new constant term
        """
        C, D, c = self.C.copy(), self.D.copy(), self.c.copy()
        if coeffs:
            for j, value in coeffs.items():
                if self.C[i, j] == 0:
                    raise ValueError(f"{self.names[j]} is not a parent of {self.names[i]}; "
                                     "a soft intervention may not change the parent set.")
                if value == 0:
                    raise ValueError("Setting a coefficient to zero would remove a parent.")
                C[i, j] = value
        if gain is not None:
            if gain == 0:
                raise ValueError("A zero gain is a hard intervention, not a soft one.")
            D[i] = gain
        if shift is not None:
            c[i] = shift
        return LinearGaussianSCM(C, self.noise_var, self.names, D, c, self.hard)

    def intervention_map(self, intervened: "LinearGaussianSCM"):
        """Endogenous component of the intervention morphism:  I = M_I o M^-1.

        Returns (A, b) with I(x) = A x + b. Needs self (the source) to be invertible.
        """
        if not self.is_invertible:
            raise ValueError("The source model must be non-intervened (invertible mixing map).")
        A = intervened.mixing_matrix @ np.linalg.inv(self.mixing_matrix)
        b = intervened.offset - A @ self.offset
        return A, b


# ---------------------------------------------------------------------------
# General SCM for the three invertible model classes
# ---------------------------------------------------------------------------

@dataclass
class Mechanism:
    """One structural equation X_i = mechanism(parents, Z_i).

    kind = "anm"    X = f(P) + Z                      inverse: Z = X - f(P)
    kind = "pnl"    X = g(f(P) + Z), g invertible     inverse: Z = g^-1(X) - f(P)
    kind = "lsnm"   X = f(P) + h(P) Z, h > 0          inverse: Z = (X - f(P)) / h(P)
    kind = "const"  X = value (hard intervention)     no inverse
    """
    kind: str
    parents: List[int] = field(default_factory=list)
    f: Optional[Callable] = None
    g: Optional[Callable] = None
    g_inv: Optional[Callable] = None
    h: Optional[Callable] = None
    value: float = 0.0

    def _f(self, pa):
        return self.f(pa) if self.f is not None else 0.0

    def forward(self, pa: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.kind == "anm":
            return self._f(pa) + z
        if self.kind == "pnl":
            return self.g(self._f(pa) + z)
        if self.kind == "lsnm":
            return self._f(pa) + self.h(pa) * z
        if self.kind == "const":
            return np.full_like(z, self.value)
        raise ValueError(self.kind)

    def inverse(self, pa: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.kind == "anm":
            return x - self._f(pa)
        if self.kind == "pnl":
            return self.g_inv(x) - self._f(pa)
        if self.kind == "lsnm":
            return (x - self._f(pa)) / self.h(pa)
        raise ValueError("A constant mechanism has no inverse: the noise value is lost.")


class GeneralSCM:
    """Markovian SCM with standard normal noise and arbitrary invertible mechanisms."""

    def __init__(self, mechanisms: List[Mechanism], names: Sequence[str]):
        for i, mech in enumerate(mechanisms):
            if any(p >= i for p in mech.parents):
                raise ValueError("Variables must be listed in topological order.")
        self.mechanisms = mechanisms
        self.names = list(names)
        self.n = len(mechanisms)

    def sample_noise(self, N: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(N, self.n))

    def mixing(self, z: np.ndarray) -> np.ndarray:
        """M: substitute recursively, parents first."""
        x = np.zeros_like(z)
        for i, mech in enumerate(self.mechanisms):
            x[:, i] = mech.forward(x[:, mech.parents], z[:, i])
        return x

    def inverse_mixing(self, x: np.ndarray) -> np.ndarray:
        """M^-1: every parent value is a known coordinate of x, so solve each equation for its noise."""
        z = np.zeros_like(x)
        for i, mech in enumerate(self.mechanisms):
            z[:, i] = mech.inverse(x[:, mech.parents], x[:, i])
        return z

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        return self.mixing(self.sample_noise(N, rng))

    def do_hard(self, assignment: Dict[int, float]) -> "GeneralSCM":
        mechs = list(self.mechanisms)
        for i, value in assignment.items():
            mechs[i] = Mechanism(kind="const", parents=[], value=value)
        return GeneralSCM(mechs, self.names)

    def do_soft(self, i: int, new_mechanism: Mechanism) -> "GeneralSCM":
        if sorted(new_mechanism.parents) != sorted(self.mechanisms[i].parents):
            raise ValueError("A soft intervention may not change the parent set.")
        mechs = list(self.mechanisms)
        mechs[i] = new_mechanism
        return GeneralSCM(mechs, self.names)

    def intervention_map(self, intervened: "GeneralSCM") -> Callable[[np.ndarray], np.ndarray]:
        """I = M_I o M^-1 as a function on endogenous samples."""
        return lambda x: intervened.mixing(self.inverse_mixing(x))
