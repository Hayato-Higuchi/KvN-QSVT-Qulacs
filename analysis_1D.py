import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation_1D import (
    CASE_B_NORM_QSVT_R,
    CASE_C_QSVT_R,
    CASE_A,
    CASE_B,
    CASE_C,
    DEFAULT_QSVT_R,
    hamiltonian_matrix,
    load_trajectory,
    occupation_basis,
)


def case_alpha(case):
    basis = occupation_basis(case.max_particles, case.num_variables)
    _, alpha, _ = hamiltonian_matrix(case, basis)
    return alpha


def normalized_time(case, alpha):
    return (
        case.plasma_frequency
        * np.arange(case.steps + 1)
        * case.tau
        / alpha
    )


def relative_error(values, reference):
    return np.linalg.norm(values - reference, axis=0) / np.linalg.norm(
        reference, axis=0
    )


def norm_deviation(values):
    norms = np.linalg.norm(values, axis=0)
    return np.abs(norms - norms[0])


def plot_case_a():
    case = CASE_A
    alpha = case_alpha(case)
    time = normalized_time(case, alpha)
    qsvt = load_trajectory(case, "qsvt")
    expm = load_trajectory(case, "expm")
    velocity_exact = np.cos(time)
    electric_exact = np.sin(time)

    figure, axes = plt.subplots(2, 1, figsize=(7.0, 6.7), sharex=True)
    axes[0].plot(time, qsvt[0], color="tab:red", label="KvN-QSVT")
    axes[0].plot(
        time, expm[0], color="tab:blue", linestyle="--", label="KvN-expm"
    )
    axes[0].plot(
        time, velocity_exact, color="black", linestyle=":", label="analytic"
    )
    axes[0].set_ylabel(r"$\tilde{u}$")
    axes[0].legend(fontsize=9, ncol=3)
    axes[0].grid(True, linestyle="--", linewidth=0.55)

    axes[1].plot(time, qsvt[case.num_grid], color="tab:red")
    axes[1].plot(
        time, expm[case.num_grid], color="tab:blue", linestyle="--"
    )
    axes[1].plot(time, electric_exact, color="black", linestyle=":")
    axes[1].set_xlabel(r"Normalized physical time $\omega_{p,e}t$")
    axes[1].set_ylabel(r"$\tilde{E}$")
    axes[1].grid(True, linestyle="--", linewidth=0.55)
    figure.tight_layout()
    path = Path(
        "output/CaseA/1DPlasmaOscillationTest_u_E_nx_5_"
        "delta_x_1_T_200_delta_t_1_m_1.pdf"
    )
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_norm_deviation(cases, parameter, path, qsvt_R=None):
    colors = {
        2: "tab:blue",
        3: "tab:green",
        4: "tab:purple",
        11: "tab:red",
        22: "tab:blue",
        33: "tab:green",
        44: "tab:purple",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    plotted_values = []
    for case in cases:
        alpha = case_alpha(case)
        time = normalized_time(case, alpha)
        qsvt = load_trajectory(case, "qsvt", qsvt_R=qsvt_R)
        expm = load_trajectory(case, "expm")
        value = case.max_particles if parameter == "m" else case.num_grid
        color = colors[value]
        qsvt_deviation = norm_deviation(qsvt)[1:]
        expm_deviation = norm_deviation(expm)[1:]
        plotted_values.extend((qsvt_deviation, expm_deviation))
        axis.semilogy(
            time[1:],
            np.maximum(qsvt_deviation, 1.0e-16),
            color=color,
            linewidth=2.0,
            alpha=0.55,
            zorder=2,
            label=(
                rf"KvN-QSVT ($R={qsvt_R}$), ${parameter}={value}$"
                if qsvt_R is not None
                else rf"KvN-QSVT, ${parameter}={value}$"
            ),
        )
        axis.semilogy(
            time[1:],
            np.maximum(expm_deviation, 1.0e-16),
            color=color,
            linestyle="--",
            linewidth=1.35,
            marker="o",
            markevery=max(1, len(expm_deviation) // 18),
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.9,
            zorder=3,
            label=rf"KvN-expm, ${parameter}={value}$",
        )
    positive_values = np.concatenate(plotted_values)
    positive_values = positive_values[positive_values > 0.0]
    axis.set_ylim(0.5 * positive_values.min(), 1.5 * positive_values.max())
    axis.set_xlabel(r"Normalized physical time $\omega_{p,e}t$", fontsize=12)
    axis.set_ylabel(
        r"Norm deviation $|\|\mathbf{x}(t)\|_2-\|\mathbf{x}(0)\|_2|$",
        fontsize=11,
    )
    axis.grid(True, which="both", linestyle="--", linewidth=0.55)
    axis.legend(fontsize=8.5, ncol=2)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_case_b(qsvt_R=CASE_B_NORM_QSVT_R):
    return plot_norm_deviation(
        CASE_B,
        "m",
        "output/CaseBC/1DAdvectionTest_"
        "L2norm_deviation_error_x_various_m.pdf",
        qsvt_R=qsvt_R,
    )


def plot_case_c(qsvt_R=CASE_C_QSVT_R):
    return plot_norm_deviation(
        CASE_C,
        "N_x",
        "output/CaseBC/1DAdvectionTest_"
        "L2norm_deviation_error_x_various_Nx.pdf",
        qsvt_R=qsvt_R,
    )


def write_validation(path="output/CaseABC_validation.csv"):
    path = Path(path)
    rows = []
    for case in (CASE_A,) + CASE_B + CASE_C:
        qsvt_R = CASE_C_QSVT_R if case.label == "C" else None
        qsvt = load_trajectory(case, "qsvt", qsvt_R=qsvt_R)
        expm = load_trajectory(case, "expm")
        error = relative_error(qsvt, expm)
        rows.append(
            {
                "case": case.label,
                "num_grid": case.num_grid,
                "domain_length": case.domain_length,
                "delta_x": case.delta_x,
                "m": case.max_particles,
                "qsvt_R": qsvt_R or DEFAULT_QSVT_R,
                "maximum_qsvt_expm_relative_error": np.max(error[1:]),
                "final_qsvt_expm_relative_error": error[-1],
                "maximum_qsvt_norm_deviation": np.max(norm_deviation(qsvt)),
                "maximum_expm_norm_deviation": np.max(norm_deviation(expm)),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    paths = [plot_case_a(), plot_case_b(), plot_case_c()]
    validation = write_validation()
    for path in paths:
        print(f"Saved: {path}")
    print(f"Saved: {validation}")


if __name__ == "__main__":
    main()
