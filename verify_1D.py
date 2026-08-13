import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from simulation_1D import (
    CASE_B,
    StateEncoder,
    hamiltonian_matrix,
    initial_values,
    occupation_basis,
    propagate_expm,
    propagate_qsvt,
    qsvt_polynomial_coefficients,
)
from sub_function_1D import (
    HS_TestSim_U_Hamiltonian_matrix,
    Hamiltonian_matrix,
    normalize_matrix,
)


def relative_norm(first, second):
    return np.linalg.norm(first - second) / np.linalg.norm(second)


def main():
    case = replace(CASE_B[0], steps=2)
    basis = occupation_basis(case.max_particles, case.num_variables)
    sparse_hamiltonian, alpha, basis = hamiltonian_matrix(case, basis)
    dense_hamiltonian, dense_alpha = normalize_matrix(
        Hamiltonian_matrix(
            case.delta_x,
            case.Lambda,
            case.density,
            case.epsilon_0,
            case.mass,
            case.charge,
            case.max_particles,
            case.num_grid,
        )
    )
    encoder = StateEncoder(basis, case.Lambda)
    values = initial_values(case)

    sparse_expm = propagate_expm(
        case, sparse_hamiltonian, encoder, values=values
    )
    state, scale = encoder.prepare(values)
    propagator = expm(-1j * dense_hamiltonian * case.tau)
    dense_expm = np.empty_like(sparse_expm)
    dense_expm[:, 0] = values
    for step in range(1, case.steps + 1):
        state = propagator @ state
        state /= np.linalg.norm(state)
        values = state[1:case.num_variables + 1].real * scale
        dense_expm[:, step] = values
        state, scale = encoder.prepare(values)

    coefficients = qsvt_polynomial_coefficients(
        "output/qsvt_phases/cos1x_R5.csv",
        "output/qsvt_phases/sin1x_R5.csv",
    )
    polynomial_qsvt = propagate_qsvt(
        case,
        sparse_hamiltonian,
        encoder,
        coefficients,
        values=initial_values(case),
    )
    velocity = np.zeros((case.num_grid, case.steps + 1))
    electric = np.zeros((case.num_grid, case.steps + 1))
    velocity[:, 0] = initial_values(case)[:case.num_grid]
    velocity, electric, circuit_alpha = HS_TestSim_U_Hamiltonian_matrix(
        case.num_qubits,
        1,
        case.tau,
        case.delta_x,
        case.Lambda,
        case.density,
        case.epsilon_0,
        case.mass,
        case.charge,
        case.max_particles,
        case.num_grid,
        math.comb(case.max_particles + case.num_variables,
                  case.max_particles),
        velocity,
        electric,
        case.steps,
    )
    circuit_qsvt = np.vstack([velocity, electric])

    coefficients_R2 = qsvt_polynomial_coefficients(
        "output/qsvt_phases/cos1x_R2.csv",
        "output/qsvt_phases/sin1x_R2.csv",
    )
    polynomial_qsvt_R2 = propagate_qsvt(
        case,
        sparse_hamiltonian,
        encoder,
        coefficients_R2,
        values=initial_values(case),
    )
    velocity_R2 = np.zeros((case.num_grid, case.steps + 1))
    electric_R2 = np.zeros((case.num_grid, case.steps + 1))
    velocity_R2[:, 0] = initial_values(case)[:case.num_grid]
    velocity_R2, electric_R2, _ = HS_TestSim_U_Hamiltonian_matrix(
        case.num_qubits,
        1,
        case.tau,
        case.delta_x,
        case.Lambda,
        case.density,
        case.epsilon_0,
        case.mass,
        case.charge,
        case.max_particles,
        case.num_grid,
        math.comb(case.max_particles + case.num_variables,
                  case.max_particles),
        velocity_R2,
        electric_R2,
        case.steps,
        R=2,
    )
    circuit_qsvt_R2 = np.vstack([velocity_R2, electric_R2])

    records = [
        {
            "check": "alpha_sparse_dense",
            "value": abs(alpha - dense_alpha),
            "tolerance": 1.0e-12,
        },
        {
            "check": "alpha_sparse_circuit",
            "value": abs(alpha - circuit_alpha),
            "tolerance": 1.0e-12,
        },
        {
            "check": "hamiltonian_sparse_dense_max_abs",
            "value": np.max(
                np.abs(sparse_hamiltonian.toarray() - dense_hamiltonian)
            ),
            "tolerance": 1.0e-12,
        },
        {
            "check": "expm_sparse_dense_two_step_relative",
            "value": relative_norm(sparse_expm, dense_expm),
            "tolerance": 1.0e-12,
        },
        {
            "check": "qsvt_polynomial_circuit_two_step_relative",
            "value": relative_norm(polynomial_qsvt, circuit_qsvt),
            "tolerance": 1.0e-12,
        },
        {
            "check": "qsvt_R2_polynomial_circuit_two_step_relative",
            "value": relative_norm(
                polynomial_qsvt_R2, circuit_qsvt_R2
            ),
            "tolerance": 1.0e-12,
        },
    ]
    for full_case in CASE_B:
        full_basis = occupation_basis(
            full_case.max_particles, full_case.num_variables
        )
        _, full_alpha, _ = hamiltonian_matrix(full_case, full_basis)
        kvn_step = full_case.tau / full_alpha
        rk4_step = kvn_step / 128
        records.append(
            {
                "check": f"rk4_time_grid_residual_m{full_case.max_particles}",
                "value": abs(128 * rk4_step - kvn_step),
                "tolerance": 1.0e-15,
            }
        )

    for record in records:
        record["passed"] = record["value"] <= record["tolerance"]
        print(
            f"{record['check']}: {record['value']:.6e} "
            f"(tolerance {record['tolerance']:.1e})",
            flush=True,
        )
    if not all(record["passed"] for record in records):
        raise RuntimeError("numerical verification failed")

    path = Path("output/CaseABC_implementation_verification.csv")
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("check", "value", "tolerance", "passed"),
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
