from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qsvt_phase_generation_1D import main as generate_phases
from simulation_1D import (
    CASE_B,
    StateEncoder,
    hamiltonian_matrix,
    initial_values,
    load_trajectory,
    occupation_basis,
    propagate_qsvt,
    qsvt_polynomial_coefficients,
)


R_VALUES = (3, 5, 7, 9)


def output_paths(case, R):
    directory = Path("output/CaseBC")
    suffix = (
        f"numgrid_{case.num_grid}_nx_{case.num_qubits}"
        f"_delta_x_{case.delta_x:g}_T_{case.steps * case.tau:g}"
        f"_delta_t_{case.tau:g}_m_{case.max_particles}_R{R}.npy"
    )
    return (
        directory / f"1DAdvectionTest_qsvt_u_{suffix}",
        directory / f"1DAdvectionTest_qsvt_E_{suffix}",
    )


def relative_error(values, reference):
    return np.linalg.norm(values - reference, axis=0) / np.linalg.norm(
        reference, axis=0
    )


def main():
    case = CASE_B[0]
    phase_directory = Path("output/qsvt_phases")
    required = [
        phase_directory / f"{branch}1x_R{R}.csv"
        for R in R_VALUES
        for branch in ("cos", "sin")
    ]
    if not all(path.exists() for path in required):
        generate_phases()

    basis = occupation_basis(case.max_particles, case.num_variables)
    hamiltonian, alpha, basis = hamiltonian_matrix(case, basis)
    encoder = StateEncoder(basis, case.Lambda)
    reference = load_trajectory(case, "expm")
    results = {}
    errors = {}

    for R in R_VALUES:
        coefficients = qsvt_polynomial_coefficients(
            phase_directory / f"cos1x_R{R}.csv",
            phase_directory / f"sin1x_R{R}.csv",
        )
        trajectory = propagate_qsvt(
            case,
            hamiltonian,
            encoder,
            coefficients,
            values=initial_values(case),
            progress=False,
        )
        velocity_path, electric_path = output_paths(case, R)
        velocity_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(velocity_path, trajectory[:case.num_grid])
        np.save(electric_path, trajectory[case.num_grid:])
        results[R] = trajectory
        errors[R] = relative_error(trajectory, reference)
        print(
            f"R={R}: maximum relative error="
            f"{np.max(errors[R][1:]):.6e}",
            flush=True,
        )

    normalized_time = (
        case.plasma_frequency
        * np.arange(case.steps + 1)
        * case.tau
        / alpha
    )
    colors = {
        3: "tab:blue",
        5: "tab:green",
        7: "tab:orange",
        9: "tab:purple",
    }
    figure, axis = plt.subplots(figsize=(7.0, 4.7))
    for R in R_VALUES:
        axis.semilogy(
            normalized_time[1:],
            np.maximum(errors[R][1:], 1.0e-16),
            color=colors[R],
            linewidth=1.5,
            label=rf"$R={R}$",
        )
    axis.set_xlabel(r"Normalized physical time $\omega_{p,e}t$", fontsize=12)
    axis.set_ylabel(
        r"Relative deviation $\|\mathbf{x}_{R}-\mathbf{x}_{\rm expm}\|"
        r"/\|\mathbf{x}_{\rm expm}\|$",
        fontsize=11,
    )
    axis.grid(True, which="both", linestyle="--", linewidth=0.55)
    axis.legend(fontsize=10)
    figure.tight_layout()
    figure_path = Path(
        "output/CaseBC/1DAdvectionTest_qsvt_R_dependence.pdf"
    )
    figure.savefig(figure_path)
    plt.close(figure)

    columns = [normalized_time]
    header = ["omega_pe_t"]
    for R in R_VALUES:
        columns.append(errors[R])
        header.append(f"relative_error_R{R}")
    np.savetxt(
        "output/CaseBC/1DAdvectionTest_qsvt_R_dependence.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    print(f"alpha={alpha:.12g}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
