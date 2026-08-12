from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation_1D import (
    CASE_B,
    hamiltonian_matrix,
    initial_values,
    load_trajectory,
    occupation_basis,
)


RK4_SUBSTEPS = 128
RK4_CHECK_SUBSTEPS = 256


def rhs(values, case):
    velocity = values[:case.num_grid]
    electric = values[case.num_grid:]
    forward_one = np.roll(velocity, -1)
    forward_two = np.roll(velocity, -2)
    backward_one = np.roll(velocity, 1)
    backward_two = np.roll(velocity, 2)
    advection = -(
        forward_one * forward_two - backward_one * backward_two
    ) / (4.0 * case.delta_x)
    signed_frequency = case.charge * np.sqrt(
        case.density / (case.epsilon_0 * case.mass)
    )
    return np.concatenate(
        [
            advection + signed_frequency * electric,
            -signed_frequency * velocity,
        ]
    )


def rk4_trajectory(case, alpha, substeps):
    values = initial_values(case)
    trajectory = np.empty((case.num_variables, case.steps + 1), dtype=float)
    trajectory[:, 0] = values
    kvn_step = case.tau / alpha
    step = kvn_step / substeps
    if not np.isclose(step * substeps, kvn_step, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("RK4 and KvN time grids do not coincide")
    for sample in range(1, case.steps + 1):
        for _ in range(substeps):
            k1 = rhs(values, case)
            k2 = rhs(values + 0.5 * step * k1, case)
            k3 = rhs(values + 0.5 * step * k2, case)
            k4 = rhs(values + step * k3, case)
            values += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        trajectory[:, sample] = values
    return trajectory, step


def relative_error(values, reference):
    return np.linalg.norm(values - reference, axis=0) / np.linalg.norm(
        reference, axis=0
    )


def main():
    colors = {2: "tab:blue", 3: "tab:green", 4: "tab:purple"}
    figure, axis = plt.subplots(figsize=(7.0, 4.7))
    rows = []

    for case in CASE_B:
        basis = occupation_basis(case.max_particles, case.num_variables)
        _, alpha, _ = hamiltonian_matrix(case, basis)
        expm_values = load_trajectory(case, "expm")
        rk4_values, rk4_step = rk4_trajectory(
            case, alpha, RK4_SUBSTEPS
        )
        rk4_check, _ = rk4_trajectory(
            case, alpha, RK4_CHECK_SUBSTEPS
        )
        convergence = np.max(
            np.linalg.norm(rk4_values - rk4_check, axis=0)
            / np.linalg.norm(rk4_check, axis=0)
        )
        error = relative_error(expm_values, rk4_check)
        physical_time = np.arange(case.steps + 1) * case.tau / alpha
        normalized_time = case.plasma_frequency * physical_time
        axis.semilogy(
            normalized_time[1:],
            error[1:],
            color=colors[case.max_particles],
            linewidth=1.4,
            label=rf"$m={case.max_particles}$",
        )
        for step_index in range(case.steps + 1):
            rows.append(
                (
                    case.max_particles,
                    step_index,
                    physical_time[step_index],
                    normalized_time[step_index],
                    case.tau / alpha,
                    rk4_step,
                    error[step_index],
                )
            )
        print(
            f"m={case.max_particles}: alpha={alpha:.12g}, "
            f"KvN dt={case.tau / alpha:.12g}, "
            f"RK4 dt={rk4_step:.12g}, "
            f"grid residual={abs(RK4_SUBSTEPS * rk4_step - case.tau / alpha):.3e}, "
            f"RK4 convergence={convergence:.3e}, "
            f"final relative error={error[-1]:.3e}",
            flush=True,
        )

    axis.set_xlabel(r"Normalized physical time $\omega_{p,e}t$", fontsize=12)
    axis.set_ylabel(
        r"Relative deviation $\|\mathbf{x}_{\rm expm}-\mathbf{x}_{\rm RK4}\|"
        r"/\|\mathbf{x}_{\rm RK4}\|$",
        fontsize=11,
    )
    axis.grid(True, which="both", linestyle="--", linewidth=0.55)
    axis.legend(fontsize=10)
    figure.tight_layout()
    output_directory = Path("output/CaseBC")
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path = output_directory / "1DAdvectionTest_expm_vs_RK4.pdf"
    figure.savefig(figure_path)
    plt.close(figure)

    np.savetxt(
        output_directory / "1DAdvectionTest_expm_vs_RK4.csv",
        np.asarray(rows),
        delimiter=",",
        header=(
            "m,step,physical_time,omega_pe_t,kvn_physical_dt,"
            "rk4_dt,relative_error"
        ),
        comments="",
    )
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
