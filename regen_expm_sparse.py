"""
Sparse-matrix regeneration of the 1D KvN-expm reference (Case B).

Physics identical to sub_function_1D.py, but the creation operators and the
Hamiltonian are assembled as scipy.sparse (CSR) matrices, and the per-step
propagator is applied with scipy.sparse.linalg.expm_multiply, avoiding the
dense 4845x4845 matmuls / dense expm that dominate the m=4 cost.

Cross-checked against the dense sub_function_1D result for m=2,3 (identical
alpha and matching trajectories).

Usage:  python regen_expm_sparse.py <m> <steps>   (e.g. 4 2056)
"""
import math
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from sub_function_1D import number_to_list, list_to_number, state_preparation_expm

num_grid = 8
delta_x = 1
Lambda = 1
density = 1
epsilon_0 = 1
mass = 100
q = -1
N = 2 * num_grid
tau = 1
k = 2 * np.pi / num_grid
r = np.linspace(-num_grid / 2, num_grid / 2, num_grid)


def creat_op_sparse(m, N, j):
    """Sparse creation operator a_j^dagger on the truncated Fock space,
    identical to make_creat_op_matrix but built as CSR."""
    M = math.comb(m + N, m)
    rows, cols, vals = [], [], []
    for i in range(M):
        num_list = number_to_list(i, N)
        if sum(num_list) < m:
            num_list[j] += 1
            rows.append(list_to_number(num_list))
            cols.append(i)
            vals.append(np.sqrt(num_list[j]))
    return sp.csr_matrix((vals, (rows, cols)), shape=(M, M))


def hamiltonian_sparse(m):
    A = [creat_op_sparse(m, N, i) for i in range(N)]
    M = A[0].shape[0]
    H = sp.csr_matrix((M, M), dtype=complex)
    # EM coupling term
    for i in range(num_grid):
        H = H + 1j * q * (density / (epsilon_0 * mass)) ** 0.5 * (
            A[i] @ A[i + num_grid].T - A[i + num_grid] @ A[i].T)
    # advection term
    for i in range(num_grid):
        pref = -(1j / (4 * delta_x * Lambda)) / (2 ** 0.5)
        H = H + pref * (A[i] + A[i].T) @ (
            A[(i - 1) % num_grid] @ A[(i + 1) % num_grid].T
            - A[(i + 1) % num_grid] @ A[(i - 1) % num_grid].T)
    return H


def normalize_sparse(H):
    alpha = np.sqrt((np.abs(H.data) ** 2).sum())
    return H / alpha, alpha


def run_expm_sparse(m, Total_steps):
    M = math.comb(m + N, m)
    n_x = math.floor(np.log2(M)) + 1
    H = hamiltonian_sparse(m)
    Hn, alpha = normalize_sparse(H)
    A_op = (-1j * Hn * tau).tocsc()

    u = np.zeros((num_grid, Total_steps + 1))
    E = np.zeros((num_grid, Total_steps + 1))
    u[:, 0] = np.sin(-k * r)
    x = np.concatenate((u[:, 0], E[:, 0]))
    psi_now, norm_psi = state_preparation_expm(N, M, x, Lambda)
    psi_now = psi_now.astype(complex)

    extract = np.pi ** (2 * num_grid / 4) / 2 ** (1 / 2)
    for t in range(1, Total_steps + 1):
        psi_t = expm_multiply(A_op, psi_now)      # exp(-i Hn tau) @ psi
        psi_t = psi_t / np.linalg.norm(psi_t)
        data = psi_t * norm_psi / Lambda * extract
        u[:, t] = data[1:num_grid + 1].real
        E[:, t] = data[num_grid + 1:2 * num_grid + 1].real
        y = np.concatenate((u[:, t], E[:, t]))
        psi_now, norm_psi = state_preparation_expm(N, M, y, Lambda)
        psi_now = psi_now.astype(complex)
    return u, E, alpha, n_x


if __name__ == "__main__":
    import os
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2056
    os.makedirs("output/CaseBC", exist_ok=True)
    u, E, alpha, n_x = run_expm_sparse(m, steps)
    T = steps * tau
    for nm, arr in (("u", u), ("E", E)):
        fn = ("output/CaseBC/1DAdvectionTest_expm_"
              f"{nm}_numgrid_{num_grid}_nx_{n_x}_delta_x_1_T_{T}"
              f"_delta_t_{tau}_m_{m}.npy")
        np.save(fn, arr)
    print(f"m={m}: alpha={alpha:.4f}, n_x={n_x}, "
          f"max|u|={np.abs(u).max():.4f}, steps={steps}  saved (sparse)")
