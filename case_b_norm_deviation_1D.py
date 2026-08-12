from pathlib import Path

from analysis_1D import plot_case_b
from qsvt_phase_generation_1D import main as generate_phases
from simulation_1D import (
    CASE_B,
    CASE_B_NORM_QSVT_R,
    StateEncoder,
    hamiltonian_matrix,
    initial_values,
    occupation_basis,
    propagate_expm,
    propagate_qsvt,
    qsvt_polynomial_coefficients,
    save_trajectory,
)


def main():
    phase_directory = Path("output/qsvt_phases")
    required = (
        phase_directory / f"cos1x_R{CASE_B_NORM_QSVT_R}.csv",
        phase_directory / f"sin1x_R{CASE_B_NORM_QSVT_R}.csv",
    )
    if not all(path.exists() for path in required):
        generate_phases()
    coefficients = qsvt_polynomial_coefficients(
        required[0],
        required[1],
    )

    for case in CASE_B:
        basis = occupation_basis(case.max_particles, case.num_variables)
        hamiltonian, _, basis = hamiltonian_matrix(case, basis)
        encoder = StateEncoder(basis, case.Lambda)
        expm_trajectory = propagate_expm(
            case,
            hamiltonian,
            encoder,
            values=initial_values(case),
            progress=True,
        )
        save_trajectory(case, "expm", expm_trajectory)
        trajectory = propagate_qsvt(
            case,
            hamiltonian,
            encoder,
            coefficients,
            values=initial_values(case),
            progress=True,
        )
        save_trajectory(
            case, "qsvt", trajectory, qsvt_R=CASE_B_NORM_QSVT_R
        )

    print(f"Saved: {plot_case_b(qsvt_R=CASE_B_NORM_QSVT_R)}")


if __name__ == "__main__":
    main()
