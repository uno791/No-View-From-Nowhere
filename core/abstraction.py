"""
Causal abstraction between two linear Gaussian SCMs (the code version of RQ2).

An abstraction maps a detailed (micro) model to a coarser (macro) model.
On endogenous VALUES it is a matrix F:  macro values = F @ micro values.
On endogenous MEASURES it acts by pushforward: N(m, S) goes to N(F m, F S F^T).
The pushforward is the map that lives in the category CSprob.

Interventional consistency (IC) is a property an abstraction may or may not
have. It is checked here exactly as the theory states it: for a macro
intervention I' and the micro intervention I that corresponds to it,

    push( micro law under I ) must equal ( macro law under I' ),

and the check is done with KL divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .kl import gaussian_kl
from .scm import LinearGaussianSCM


def push_gaussian(F: np.ndarray, law: Tuple[np.ndarray, np.ndarray]):
    """Pushforward of N(m, S) along x -> F x, namely N(F m, F S F^T)."""
    m, S = law
    F = np.atleast_2d(F)
    return F @ m, F @ S @ F.T


@dataclass
class LinearAbstraction:
    micro: LinearGaussianSCM
    macro: LinearGaussianSCM
    F: np.ndarray            # endogenous component alpha_X on values, shape (n_macro, n_micro)

    def __post_init__(self):
        self.F = np.atleast_2d(np.asarray(self.F, dtype=float))

    # ----- the exogenous component is forced by naturality -------------------
    def exogenous_component(self) -> np.ndarray:
        """alpha_Z = M'^-1 o alpha_X o M (linear part).

        Naturality says alpha_X o M = M' o alpha_Z. With M' invertible this
        has exactly one solution, computed here.
        """
        return np.linalg.inv(self.macro.mixing_matrix) @ self.F @ self.micro.mixing_matrix

    def naturality_residual(self) -> float:
        """max | alpha_X M - M' alpha_Z |. Zero by construction of alpha_Z."""
        lhs = self.F @ self.micro.mixing_matrix
        rhs = self.macro.mixing_matrix @ self.exogenous_component()
        return float(np.max(np.abs(lhs - rhs)))

    def noise_is_preserved(self, tol: float = 1e-10) -> bool:
        """Does alpha_Z push the micro noise law onto the macro noise law?

        Required for alpha_Z to be a morphism between the two exogenous
        probability spaces: the pushed covariance must equal the macro noise
        covariance, which is diagonal (independent noises).
        """
        A = self.exogenous_component()
        pushed = A @ np.diag(self.micro.noise_var) @ A.T
        return bool(np.allclose(pushed, np.diag(self.macro.noise_var), atol=tol))

    # ----- interventional consistency, measured by KL ------------------------
    def ic_gap(self, micro_intervened: LinearGaussianSCM, macro_intervened: LinearGaussianSCM,
               coords=None) -> float:
        """KL( push(micro law under I) || macro law under I' ).

        coords restricts the comparison to some macro coordinates. This is
        needed for hard interventions, where the intervened coordinate is a
        point mass on both sides and has no density.
        """
        pm, pS = push_gaussian(self.F, micro_intervened.law())
        qm, qS = macro_intervened.law()
        if coords is not None:
            idx = np.asarray(coords)
            pm, pS = pm[idx], pS[np.ix_(idx, idx)]
            qm, qS = qm[idx], qS[np.ix_(idx, idx)]
        return gaussian_kl(pm, pS, qm, qS)

    def value_level_gap(self, micro_intervened, macro_intervened, x: np.ndarray) -> float:
        """max | alpha_X(I(x)) - I'(alpha_X(x)) | over sample points x.

        This is the front face of the IC cube, checked on values.
        """
        A, b = self.micro.intervention_map(micro_intervened)
        A2, b2 = self.macro.intervention_map(macro_intervened)
        lhs = (x @ A.T + b) @ self.F.T
        rhs = (x @ self.F.T) @ A2.T + b2
        return float(np.max(np.abs(lhs - rhs)))
