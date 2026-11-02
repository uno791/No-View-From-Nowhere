"""
Symbolic verification with SymPy.

Every identity that the documents derive by hand is re-derived here with
symbols instead of numbers. A check passes when the difference between the
two sides simplifies to exactly zero. Run:  python experiments/symbolic_checks.py
"""

from __future__ import annotations

import sympy as sp

CHECKS = []


def check(name, expr):
    """Record a check. `expr` must simplify to zero (scalar or matrix)."""
    ok = sp.simplify(expr) == (sp.zeros(*expr.shape) if isinstance(expr, sp.MatrixBase) else 0)
    CHECKS.append((name, bool(ok)))


def main():
    a, at, ah, x0, y, x = sp.symbols("a a_tilde a_hat x_0 y x", real=True)
    sx, sy = sp.symbols("sigma_X sigma_Y", positive=True)
    zx, zy = sp.symbols("z_X z_Y", real=True)

    # ---------------- RQ1 ----------------
    M = sp.Matrix([[1, 0], [a, 1]])
    Sz = sp.diag(sx ** 2, sy ** 2)
    Sx = sp.Matrix([[sx ** 2, a * sx ** 2], [a * sx ** 2, a ** 2 * sx ** 2 + sy ** 2]])
    check("RQ1 endogenous covariance is M Sz M^T", M * Sz * M.T - Sx)
    check("RQ1 inverse mixing matrix is I - C", M.inv() - sp.Matrix([[1, 0], [-a, 1]]))

    # Hard intervention do(X = x0):  H(x, y) = (x0, a x0 + y - a x).
    H = lambda v: sp.Matrix([x0, a * x0 + v[1] - a * v[0]])
    M_of_z = M * sp.Matrix([zx, zy])
    MH_of_z = sp.Matrix([x0, a * x0 + zy])
    check("RQ1 hard naturality square  H o M = M_H", H(M_of_z) - MH_of_z)
    row = sp.Matrix([[-a, 1]])
    check("RQ1 hard pushforward variance of Y is sigma_Y^2", (row * Sx * row.T)[0] - sy ** 2)

    # Soft intervention a -> a_tilde:  S = M_S M^-1.
    Ms = sp.Matrix([[1, 0], [at, 1]])
    S = Ms * M.inv()
    check("RQ1 soft transport matrix", S - sp.Matrix([[1, 0], [at - a, 1]]))
    check("RQ1 soft naturality  S M = M_S", S * M - Ms)
    check("RQ1 soft pushforward covariance", S * Sx * S.T - Ms * Sz * Ms.T)
    Mh = sp.Matrix([[1, 0], [ah, 1]])
    check("RQ1 composition of two soft interventions", (Mh * Ms.inv()) * S - Mh * M.inv())

    # Seeing versus doing for do(Y = y0): the conditional law of X given Y differs from N(0, sx^2).
    var_y = a ** 2 * sx ** 2 + sy ** 2
    cond_var = sx ** 2 - (a * sx ** 2) ** 2 / var_y
    check("RQ1 conditional variance of X given Y", cond_var - sx ** 2 * sy ** 2 / var_y)

    # ---------------- RQ2 ----------------
    lam, a1, a2 = sp.symbols("lambda a_1 a_2", real=True)
    var_measure_blend = lam * (a1 ** 2 + 1) + (1 - lam) * (a2 ** 2 + 1)
    var_map_blend = (lam * a1 + (1 - lam) * a2) ** 2 + 1
    check("RQ2 gap between the two blends is lam (1 - lam) (a1 - a2)^2",
          var_measure_blend - var_map_blend - lam * (1 - lam) * (a1 - a2) ** 2)

    # Affineness on covariances for a fully symbolic 1 x 2 map and two symbolic covariances.
    f1, f2 = sp.symbols("f_1 f_2", real=True)
    p, q, r, p2, q2, r2 = sp.symbols("p q r p_2 q_2 r_2", real=True)
    F = sp.Matrix([[f1, f2]])
    C1, C2 = sp.Matrix([[p, q], [q, r]]), sp.Matrix([[p2, q2], [q2, r2]])
    check("RQ2 affineness of S -> F S F^T",
          F * (lam * C1 + (1 - lam) * C2) * F.T - (lam * F * C1 * F.T + (1 - lam) * F * C2 * F.T))

    # Interventional consistency of T = w1 H1 + w2 H2. Value-level gap of the front face
    # for the hard intervention do(H1 = h1, H2 = h2) against do(T = w1 h1 + w2 h2).
    b1, b2, c, w1, w2, h1, h2, u1, u2, mk = sp.symbols("b_1 b_2 c w_1 w_2 h_1 h_2 u_1 u_2 m", real=True)
    micro_after = b1 * h1 + b2 * h2 + (mk - b1 * u1 - b2 * u2)          # mark after the micro intervention
    t_new, t_old = w1 * h1 + w2 * h2, w1 * u1 + w2 * u2
    macro_after = c * t_new + (mk - c * t_old)                          # mark after the macro intervention
    gap = sp.expand(micro_after - macro_after)
    check("RQ2 IC gap vanishes when (b1, b2) = c (w1, w2)", gap.subs({b1: c * w1, b2: c * w2}))
    coefficient_h1 = sp.expand(gap).coeff(h1)
    check("RQ2 IC gap has the coefficient (b1 - c w1) on h1", coefficient_h1 - (b1 - c * w1))

    # ---------------- RQ3 ----------------
    # Right inverse: F K = 1 for K = S F^T (F S F^T)^-1, generic 2 x 2 covariance.
    Sg = sp.Matrix([[p, q], [q, r]])
    K = Sg * F.T * (F * Sg * F.T).inv()
    check("RQ3 extension is a right inverse of restriction  F K = 1", F * K - sp.eye(1))
    residual = Sg - K * F * Sg
    check("RQ3 residual covariance is invisible to the restriction  F R F^T = 0", F * residual * F.T)
    s = sp.symbols("s", positive=True)
    check("RQ3 restriction after kernel extension returns the backbone variance",
          F * (K * sp.Matrix([[s]]) * K.T + residual) * F.T - sp.Matrix([[s]]))

    width = max(len(n) for n, _ in CHECKS)
    for name, ok in CHECKS:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}")
    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} symbolic checks passed.")
    return not failed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
