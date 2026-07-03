"""
Classical RK4 reference for the 1D nonlinear advection test (Cases B/C).

This script integrates the SAME semi-discrete system as the KvN Hamiltonian
(Eqs. (31)-(32) of the manuscript, reduced to 1D with n = 1):

    du_j/dt = -(1/(4*dx)) * ( u_{j+1} u_{j+2} - u_{j-1} u_{j-2} ) + omega_pe * E_j
    dE_j/dt = - omega_pe * u_j ,          omega_pe = sqrt(n/(eps0*m_q)) * q

with periodic boundary conditions, using the classical 4th-order Runge-Kutta
method, and compares it with the stored KvN-expm data (output/CaseBC/*.npy)
for m = 2, 3, 4.  It produces the figure

    output/CaseBC/1DAdvectionTest_expm_vs_RK4.pdf

used in the revised manuscript (comparison between KvN-expm and the classical
RK4 reference solution).

The physical time of the k-th stored step is t_k = k * tau / alpha, where
alpha = ||H_m||_F is the Frobenius norm used to normalize the Hamiltonian in
sub_function_1D.normalize_matrix.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Parameters (identical to main_expm_1D_caseBC.ipynb, Case B)
# ----------------------------------------------------------------------
tau = 1
num_grid = 8
delta_x = 1.0
Lambda = 1
density = 1
epsilon_0 = 1
mass = 100
q = -1
N = 2 * num_grid
omega_pe = np.sqrt(density / (epsilon_0 * mass)) * q  # = -0.1

k = 2 * np.pi / num_grid
r = np.linspace(-num_grid / 2, num_grid / 2, num_grid)
u0 = np.sin(-k * r)
E0 = np.zeros(num_grid)

# (m, Total_steps) pairs of Case B and the corresponding n_x in the file names
CASES = [(2, 210, 8), (3, 710, 10), (4, 2056, 13)]

# ----------------------------------------------------------------------
# alpha = Frobenius norm of the truncated Hamiltonian H_m.
# Computed with sub_function_1D if qulacs is available; otherwise the
# precomputed values below (num_grid = 8, Lambda = 1, mass = 100, q = -1)
# are used.
# ----------------------------------------------------------------------
ALPHA_FALLBACK = {2: 2.0100, 3: 7.0993, 4: 20.5621}


RECOMPUTE_ALPHA = False  # set True to rebuild H_m and recompute alpha (slow for m=4)


def get_alpha(m):
    if not RECOMPUTE_ALPHA:
        return ALPHA_FALLBACK[m]
    from sub_function_1D import Hamiltonian_matrix, normalize_matrix
    H = Hamiltonian_matrix(delta_x, Lambda, density, epsilon_0,
                           mass, q, m, num_grid)
    _, alpha = normalize_matrix(H)
    return alpha


# ----------------------------------------------------------------------
# RK4 integrator for the semi-discrete system
# ----------------------------------------------------------------------
def rhs(y):
    u, E = y[:num_grid], y[num_grid:]
    adv = -(np.roll(u, -1) * np.roll(u, -2)
            - np.roll(u, 1) * np.roll(u, 2)) / (4.0 * delta_x)
    du = adv + omega_pe * E
    dE = -omega_pe * u
    return np.concatenate([du, dE])


def rk4_sample(times, dt=0.005):
    """Integrate with RK4 and sample the solution at the given times."""
    y = np.concatenate([u0, E0])
    out = [y.copy()]
    ti = 1
    n_steps = int(round(times[-1] / dt))
    for i in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = (i + 1) * dt
        while ti < len(times) and t >= times[ti] - 1e-9:
            out.append(y.copy())
            ti += 1
    return np.array(out).T  # shape (2*num_grid, len(times))


# ----------------------------------------------------------------------
# Load KvN-expm data, run RK4, and compare
# ----------------------------------------------------------------------
colors = {2: "tab:blue", 3: "tab:green", 4: "tab:purple"}
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 7.6), sharex=True)

for m, steps, n_x in CASES:
    alpha = get_alpha(m)
    t_phys = np.arange(steps + 1) * tau / alpha
    base = ("output/CaseBC/1DAdvectionTest_expm_{}_numgrid_{}_nx_{}"
            "_delta_x_1_T_{}_delta_t_{}_m_{}.npy")
    u_e = np.load(base.format("u", num_grid, n_x, steps, tau, m))
    E_e = np.load(base.format("E", num_grid, n_x, steps, tau, m))

    x_rk4 = rk4_sample(t_phys)
    x_expm = np.vstack([u_e, E_e])

    rel = (np.linalg.norm(x_expm - x_rk4, axis=0)
           / np.linalg.norm(x_rk4, axis=0))

    wt = np.abs(omega_pe) * t_phys
    ax_top.plot(wt, u_e[0, :], color=colors[m], lw=1.3,
                label=rf"KvN-expm $(m={m})$")
    ax_bot.semilogy(wt, np.maximum(rel, 1e-8), color=colors[m],
                    lw=(0.7 if m == 4 else 1.4),
                    label=rf"$m={m}$")

    i10 = np.argmin(np.abs(t_phys - 10))
    print(f"m={m}: alpha={alpha:.4f}, "
          f"rel. dev. vs RK4 @ t=10: {rel[i10]:.3e}, "
          f"@ final: {rel[-1]:.3e}")

# RK4 reference trace on the top panel (use the finest time grid, m=4)
alpha4 = get_alpha(4)
t4 = np.arange(CASES[-1][1] + 1) * tau / alpha4
x_ref = rk4_sample(t4)
ax_top.plot(np.abs(omega_pe) * t4, x_ref[0, :], "k--", lw=1.6,
            label="classical RK4")

ax_top.set_ylabel(r"$\tilde{u}$ at $x$-index 0", fontsize=13)
ax_top.legend(fontsize=10, ncol=2)
ax_top.grid(True, ls="--", lw=0.6)

ax_bot.set_xlabel(r"$\omega_{p,e}\,t$", fontsize=13)
ax_bot.set_ylabel(
    r"$\|\mathbf{x}_{\mathrm{expm}}-\mathbf{x}_{\mathrm{RK4}}\|"
    r"/\|\mathbf{x}_{\mathrm{RK4}}\|$", fontsize=12)
ax_bot.legend(fontsize=11)
ax_bot.grid(True, which="both", ls="--", lw=0.6)

fig.tight_layout()
outname = "output/CaseBC/1DAdvectionTest_expm_vs_RK4.pdf"
fig.savefig(outname, format="pdf")
print(f"Saved: {outname}")
