"""
QSVT phase-factor generation for the 1D KvN-QSVT pipeline (tau = 1).

The circuit V_gate in sub_function_1D.py consumes two phase vectors,
cos_phi_vec (length 2R+1) and sin_phi_vec (length 2R+2), and applies the
one-step propagator block

    ( P_cos(H~) - i P_sin(H~) ) / 2  ~  exp(-i tau H~) / 2,

where P_cos / P_sin are the polynomials realized by the cos / sin phase
sets.  One can show (and this script verifies numerically) that each branch
of V_gate is exactly equivalent to standard reflection-convention quantum
signal processing:

    f(x) = Re <0| e^{i phi_0 Z} R(x) e^{i phi_1 Z} R(x) ... R(x) e^{i phi_d Z} |0>,
    R(x) = [[x, sqrt(1-x^2)], [sqrt(1-x^2), -x]],

with d = 2R signal operators for the cos branch and d = 2R+1 for the sin
branch.  This script solves the phase-factor equations by Gauss-Newton
least squares (analytic Jacobian) so that

    P_cos(x) = J_0(tau) + 2 sum_{k=1}^{R} (-1)^k J_{2k}(tau)   T_{2k}(x),
    P_sin(x) = 2 sum_{k=0}^{R} (-1)^k J_{2k+1}(tau) T_{2k+1}(x)

(the Jacobi-Anger truncations at index R, Eq. (20) of the paper).  The
solved phases are drop-in compatible with V_gate and are written to

    output/qsvt_phases/cos{tau}x_R{R}.csv
    output/qsvt_phases/sin{tau}x_R{R}.csv

in the same one-row CSV format as output/cos1x.csv / output/sin1x.csv.

NOTE: the phase files shipped with the original repository
(output/cos1x.csv, 5 entries; output/sin1x.csv, 6 entries) correspond to
R = 2 (polynomial degrees 4 and 5), and output/cos25x.csv (37 entries)
to R = 18; the file names do not encode R.  This script regenerates
R = 2 as a cross-check and produces R = 3, 5, 7, 9 for the R-dependence
study of the revised paper.

Usage:  python qsvt_phase_generation_1D.py
"""
import csv
import os
import time

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import least_squares
from scipy.special import jv

TAU = 1.0
R_LIST = [2, 3, 5, 7, 9]
OUTDIR = "output/qsvt_phases"
SEED = 20260702

# ---------------------------------------------------------------------------
# 2x2 reflection-convention QSP: value and analytic Jacobian, batched over x
# ---------------------------------------------------------------------------


def _refl_stack(xs):
    xs = np.asarray(xs, float)
    s = np.sqrt(np.clip(1.0 - xs * xs, 0.0, None))
    R = np.zeros((len(xs), 2, 2), dtype=complex)
    R[:, 0, 0] = xs
    R[:, 0, 1] = s
    R[:, 1, 0] = s
    R[:, 1, 1] = -xs
    return R


def _ez(phi):
    return np.array([[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]],
                    dtype=complex)


def qsp_value_and_jac(xs, phis, R=None, want_jac=True):
    """f(x)=Re M[0,0], M = E0 R E1 R ... R Ed;  df/dphi_k analytic."""
    if R is None:
        R = _refl_stack(xs)
    n, d1 = len(xs), len(phis)
    E = [_ez(p) for p in phis]
    # suffixes S[k] = E_k R S[k+1];  S[d] = E_d
    S = [None] * d1
    S[-1] = np.broadcast_to(E[-1], (n, 2, 2)).copy()
    for k in range(d1 - 2, -1, -1):
        S[k] = E[k] @ (R @ S[k + 1])
    f = S[0][:, 0, 0].real
    if not want_jac:
        return f, None
    # prefixes P[k] = E0 R ... E_{k-1} R  (P[0] = I)
    jac = np.empty((n, d1))
    P = np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy()
    iZ = np.array([[1j, 0], [0, -1j]], dtype=complex)
    for k in range(d1):
        M_k = P @ (iZ @ S[k])
        jac[:, k] = M_k[:, 0, 0].real
        if k < d1 - 1:
            P = (P @ E[k]) @ R
    return f, jac


def solve_phases(coef, n_phi, rng, label, tol=1e-12, tries=60):
    """Find phases s.t. Re<0|...|0> = Chebyshev poly `coef` on [-1,1]."""
    deg = len(coef) - 1
    m = 4 * (deg + 1)
    xs = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    Rst = _refl_stack(xs)
    # safeguard: reflection-QSP requires |target| <= 1
    xd = np.linspace(-1, 1, 4001)
    mx = np.max(np.abs(C.chebval(xd, coef)))
    scale = min(1.0, (1.0 - 1e-12) / mx)
    target = C.chebval(xs, coef) * scale

    def fun(p):
        f, _ = qsp_value_and_jac(xs, p, Rst, want_jac=False)
        return f - target

    def jac(p):
        _, J = qsp_value_and_jac(xs, p, Rst)
        return J

    inits = [np.zeros(n_phi),
             np.concatenate(([np.pi / 4], np.zeros(n_phi - 2), [np.pi / 4]))]
    best = (None, np.inf)
    for t in range(tries):
        x0 = inits[t] if t < len(inits) else rng.uniform(-np.pi, np.pi, n_phi)
        sol = least_squares(fun, x0, jac=jac, xtol=3e-16, ftol=3e-16,
                            gtol=3e-16, max_nfev=400)
        err = np.max(np.abs(fun(sol.x)))
        if err < best[1]:
            best = (sol.x, err)
        if best[1] < tol:
            break
    if best[1] > 1e-9:
        print(f"    [warn] {label}: residual {best[1]:.2e} after {tries} tries")
    return best[0], best[1], scale


def jacobi_anger_targets(R, tau):
    c_cos = np.zeros(2 * R + 1)
    c_cos[0] = jv(0, tau)
    for k in range(1, R + 1):
        c_cos[2 * k] = 2 * (-1) ** k * jv(2 * k, tau)
    c_sin = np.zeros(2 * R + 2)
    for k in range(0, R + 1):
        c_sin[2 * k + 1] = 2 * (-1) ** k * jv(2 * k + 1, tau)
    return c_cos, c_sin


# ---------------------------------------------------------------------------
# Literal 4x4 emulation of V_gate branches (for drop-in validation)
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)
CPIX = np.kron(X, P0) + np.kron(I2, P1)


def _gadget(phi):
    rzm = np.diag([np.exp(-1j * phi), np.exp(1j * phi)])
    return CPIX @ np.kron(rzm, I2) @ CPIX


def branch_values(xs, phis, is_sin):
    xs = np.asarray(xs, float)
    s = np.sqrt(np.clip(1 - xs ** 2, 0, None))
    U = np.zeros((len(xs), 2, 2), complex)
    U[:, 0, 0], U[:, 0, 1], U[:, 1, 0], U[:, 1, 1] = xs, -1j * s, 1j * s, -xs
    IU = np.zeros((len(xs), 4, 4), complex)
    IU[:, :2, :2] = U
    IU[:, 2:, 2:] = U
    IUd = np.conj(np.transpose(IU, (0, 2, 1)))
    M = np.broadcast_to(np.kron(HAD, I2), (len(xs), 4, 4)).copy()
    if not is_sin:
        Rr = (len(phis) - 1) // 2
        for j in reversed(range(1, Rr + 1)):
            M = IU @ (_gadget(phis[2 * j]) @ M)
            M = IUd @ (_gadget(phis[2 * j - 1]) @ M)
        M = _gadget(phis[0]) @ M
    else:
        Rr = (len(phis) - 2) // 2
        for j in reversed(range(1, Rr + 1)):
            M = IU @ (_gadget(phis[2 * j + 1]) @ M)
            M = IUd @ (_gadget(phis[2 * j]) @ M)
        M = IU @ (_gadget(phis[1]) @ M)
        M = _gadget(phis[0]) @ M
    M = np.kron(HAD, I2) @ M
    return M[:, 0, 0]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    xd = np.linspace(-1, 1, 2001)
    print(f"tau = {TAU};  generating phases for R in {R_LIST}")
    for R in R_LIST:
        t0 = time.time()
        c_cos, c_sin = jacobi_anger_targets(R, TAU)
        phc, ec, sc = solve_phases(c_cos, 2 * R + 1, rng, f"cos R={R}")
        phs, es, ss = solve_phases(c_sin, 2 * R + 2, rng, f"sin R={R}")
        # dense validation through the literal 4x4 V_gate emulation
        blk = 0.5 * (branch_values(xd, phc, False)
                     - 1j * branch_values(xd, phs, True))
        e_step = np.max(np.abs(2 * blk - np.exp(-1j * TAU * xd)))
        e_ja = np.max(np.abs(C.chebval(xd, c_cos) - 1j * C.chebval(xd, c_sin)
                             - np.exp(-1j * TAU * xd)))
        for name, vec in (("cos", phc), ("sin", phs)):
            fn = f"{OUTDIR}/{name}{int(TAU)}x_R{R}.csv"
            with open(fn, "w", newline="") as f:
                csv.writer(f).writerow(["%.17g" % float(v) for v in vec])
        print(f"  R={R}: fit residual (cos,sin)=({ec:.1e},{es:.1e}); "
              f"one-step block error |2*blk - e^(-i tau x)|_max = {e_step:.3e} "
              f"(ideal Jacobi-Anger: {e_ja:.3e})  [{time.time()-t0:.1f}s]")
    print(f"Phase files written to {OUTDIR}/")


if __name__ == "__main__":
    main()
