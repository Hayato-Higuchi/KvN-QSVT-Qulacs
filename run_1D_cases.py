import argparse
from pathlib import Path

from analysis_1D import main as create_figures
from classical_rk4_1D_caseBC import main as compare_rk4
from qsvt_R_sweep_1D_caseB import main as run_R_sweep
from qsvt_phase_generation_1D import main as generate_phases
from simulation_1D import (
    CASE_B_NORM_QSVT_R,
    DEFAULT_QSVT_R,
    StateEncoder,
    all_cases,
    hamiltonian_matrix,
    occupation_basis,
    propagate_expm,
    propagate_qsvt,
    qsvt_polynomial_coefficients,
    save_trajectory,
    write_metadata,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-r-sweep", action="store_true"
    )
    parser.add_argument(
        "--skip-rk4", action="store_true"
    )
    arguments = parser.parse_args()

    phase_directory = Path("output/qsvt_phases")
    required = [
        phase_directory / f"{branch}1x_R{R}.csv"
        for R in (CASE_B_NORM_QSVT_R, DEFAULT_QSVT_R)
        for branch in ("cos", "sin")
    ]
    if not all(path.exists() for path in required):
        generate_phases()
    coefficients = qsvt_polynomial_coefficients(
        phase_directory / f"cos1x_R{DEFAULT_QSVT_R}.csv",
        phase_directory / f"sin1x_R{DEFAULT_QSVT_R}.csv",
    )
    case_b_norm_coefficients = qsvt_polynomial_coefficients(
        phase_directory / f"cos1x_R{CASE_B_NORM_QSVT_R}.csv",
        phase_directory / f"sin1x_R{CASE_B_NORM_QSVT_R}.csv",
    )
    metadata = []
    for case in all_cases():
        basis = occupation_basis(case.max_particles, case.num_variables)
        hamiltonian, alpha, basis = hamiltonian_matrix(case, basis)
        encoder = StateEncoder(basis, case.Lambda)

        expm = propagate_expm(
            case, hamiltonian, encoder, progress=True
        )
        save_trajectory(case, "expm", expm)

        qsvt = propagate_qsvt(
            case,
            hamiltonian,
            encoder,
            coefficients,
            progress=True,
        )
        save_trajectory(case, "qsvt", qsvt)

        if case.label == "B":
            case_b_norm_qsvt = propagate_qsvt(
                case,
                hamiltonian,
                encoder,
                case_b_norm_coefficients,
                progress=True,
            )
            save_trajectory(
                case,
                "qsvt",
                case_b_norm_qsvt,
                qsvt_R=CASE_B_NORM_QSVT_R,
            )

        record = {
            "case": case.label,
            "num_grid": case.num_grid,
            "m": case.max_particles,
            "alpha": alpha,
            "tau": case.tau,
            "physical_dt": case.tau / alpha,
            "steps": case.steps,
            "physical_final_time": case.steps * case.tau / alpha,
            "omega_dt": case.plasma_frequency * case.tau / alpha,
            "qsvt_R": DEFAULT_QSVT_R,
        }
        metadata.append(record)
        print(
            f"Case {case.label}, N_x={case.num_grid}, "
            f"m={case.max_particles}: alpha={alpha:.12g}, "
            f"physical dt={case.tau / alpha:.12g}",
            flush=True,
        )

    write_metadata(metadata)
    create_figures()

    if not arguments.skip_r_sweep:
        run_R_sweep()
    if not arguments.skip_rk4:
        compare_rk4()


if __name__ == "__main__":
    main()
