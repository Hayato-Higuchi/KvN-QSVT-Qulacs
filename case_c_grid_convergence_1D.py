import csv
from pathlib import Path

import numpy as np

from analysis_1D import norm_deviation, plot_case_c, relative_error
from qsvt_phase_generation_1D import main as generate_phases
from simulation_1D import (
    CASE_C,
    CASE_C_QSVT_R,
    StateEncoder,
    hamiltonian_matrix,
    initial_values,
    occupation_basis,
    propagate_expm,
    propagate_qsvt,
    qsvt_polynomial_coefficients,
    save_trajectory,
)


def spatial_discretization_error(case, values):
    velocity = values[:case.num_grid]
    coordinates = (
        np.arange(case.num_grid, dtype=float) - case.num_grid / 2
    ) * case.delta_x
    wave_number = -2.0 * np.pi / case.domain_length
    discrete = -(
        np.roll(velocity, -1) * np.roll(velocity, -2)
        - np.roll(velocity, 1) * np.roll(velocity, 2)
    ) / (4.0 * case.delta_x)
    continuum = -1.5 * velocity * wave_number * np.cos(
        wave_number * coordinates
    )
    return np.linalg.norm(discrete - continuum) / np.linalg.norm(continuum)


def main():
    phase_directory = Path("output/qsvt_phases")
    required = (
        phase_directory / f"cos1x_R{CASE_C_QSVT_R}.csv",
        phase_directory / f"sin1x_R{CASE_C_QSVT_R}.csv",
    )
    if not all(path.exists() for path in required):
        generate_phases()
    coefficients = qsvt_polynomial_coefficients(*required)
    rows = []

    for case in CASE_C:
        basis = occupation_basis(case.max_particles, case.num_variables)
        hamiltonian, alpha, basis = hamiltonian_matrix(case, basis)
        encoder = StateEncoder(basis, case.Lambda)
        values = initial_values(case)
        expm = propagate_expm(
            case,
            hamiltonian,
            encoder,
            values=values,
            progress=True,
        )
        qsvt = propagate_qsvt(
            case,
            hamiltonian,
            encoder,
            coefficients,
            values=values,
            progress=True,
        )
        save_trajectory(case, "expm", expm)
        save_trajectory(case, "qsvt", qsvt, qsvt_R=CASE_C_QSVT_R)
        qsvt_expm_error = relative_error(qsvt, expm)
        rows.append(
            {
                "num_grid": case.num_grid,
                "domain_length": case.domain_length,
                "delta_x": case.delta_x,
                "wave_number": 2.0 * np.pi / case.domain_length,
                "qsvt_R": CASE_C_QSVT_R,
                "initial_spatial_discretization_relative_error": (
                    spatial_discretization_error(case, values)
                ),
                "alpha": alpha,
                "steps": case.steps,
                "physical_dt": case.tau / alpha,
                "physical_final_time": case.steps * case.tau / alpha,
                "maximum_qsvt_norm_deviation": np.max(
                    norm_deviation(qsvt)
                ),
                "maximum_expm_norm_deviation": np.max(
                    norm_deviation(expm)
                ),
                "maximum_qsvt_expm_relative_error": np.max(
                    qsvt_expm_error
                ),
                "final_qsvt_expm_relative_error": qsvt_expm_error[-1],
            }
        )

    rows[0]["spatial_discretization_observed_order"] = ""
    for previous, current in zip(rows, rows[1:]):
        current["spatial_discretization_observed_order"] = (
            np.log(
                previous[
                    "initial_spatial_discretization_relative_error"
                ]
                / current[
                    "initial_spatial_discretization_relative_error"
                ]
            )
            / np.log(previous["delta_x"] / current["delta_x"])
        )

    delta_x = np.log([row["delta_x"] for row in rows])
    spatial_error = np.log(
        [
            row["initial_spatial_discretization_relative_error"]
            for row in rows
        ]
    )
    observed_order = np.polyfit(delta_x, spatial_error, 1)[0]

    output = Path("output/CaseBC/CaseC_fixed_L_validation.csv")
    with open(output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {plot_case_c()}")
    print(f"Saved: {output}")
    print(f"Observed spatial-discretization order: {observed_order:.6f}")


if __name__ == "__main__":
    main()
