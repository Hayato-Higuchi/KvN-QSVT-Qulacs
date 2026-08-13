import csv
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev
from scipy.optimize import least_squares
from scipy.special import jv


TAU = 1.0
R_VALUES = (2, 3, 5, 7, 9)
OUTPUT_DIRECTORY = Path("output/qsvt_phases")
RANDOM_SEED = 20260702


def reflection_stack(nodes):
    nodes = np.asarray(nodes, dtype=float)
    complement = np.sqrt(np.clip(1.0 - nodes**2, 0.0, None))
    reflection = np.empty((nodes.size, 2, 2), dtype=np.complex128)
    reflection[:, 0, 0] = nodes
    reflection[:, 0, 1] = complement
    reflection[:, 1, 0] = complement
    reflection[:, 1, 1] = -nodes
    return reflection


def phase_matrix(phi):
    return np.array(
        [[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]],
        dtype=np.complex128,
    )


def qsp_value_and_jacobian(nodes, phases, reflection=None, jacobian=True):
    if reflection is None:
        reflection = reflection_stack(nodes)
    count = len(nodes)
    phase_count = len(phases)
    phase_matrices = [phase_matrix(phi) for phi in phases]
    suffix = [None] * phase_count
    suffix[-1] = np.broadcast_to(
        phase_matrices[-1], (count, 2, 2)
    ).copy()
    for index in range(phase_count - 2, -1, -1):
        suffix[index] = phase_matrices[index] @ (
            reflection @ suffix[index + 1]
        )
    values = suffix[0][:, 0, 0].real
    if not jacobian:
        return values, None
    derivatives = np.empty((count, phase_count))
    prefix = np.broadcast_to(
        np.eye(2, dtype=np.complex128), (count, 2, 2)
    ).copy()
    iz = np.array([[1j, 0.0], [0.0, -1j]], dtype=np.complex128)
    for index in range(phase_count):
        differentiated = prefix @ (iz @ suffix[index])
        derivatives[:, index] = differentiated[:, 0, 0].real
        if index < phase_count - 1:
            prefix = (prefix @ phase_matrices[index]) @ reflection
    return values, derivatives


def jacobi_anger_coefficients(R, tau):
    cosine = np.zeros(2 * R + 1)
    cosine[0] = jv(0, tau)
    for order in range(1, R + 1):
        cosine[2 * order] = 2 * (-1) ** order * jv(2 * order, tau)
    sine = np.zeros(2 * R + 2)
    for order in range(R + 1):
        sine[2 * order + 1] = (
            2 * (-1) ** order * jv(2 * order + 1, tau)
        )
    return cosine, sine


def solve_phases(coefficients, phase_count, generator, attempts=60):
    degree = len(coefficients) - 1
    count = 4 * (degree + 1)
    nodes = np.cos(np.pi * (np.arange(count) + 0.5) / count)
    reflection = reflection_stack(nodes)
    target = chebyshev.chebval(nodes, coefficients)

    def residual(phases):
        values, _ = qsp_value_and_jacobian(
            nodes, phases, reflection, jacobian=False
        )
        return values - target

    def jacobian(phases):
        _, value = qsp_value_and_jacobian(nodes, phases, reflection)
        return value

    initial_values = [
        np.zeros(phase_count),
        np.concatenate(
            ([np.pi / 4], np.zeros(phase_count - 2), [np.pi / 4])
        ),
    ]
    best_phases = None
    best_error = np.inf
    for attempt in range(attempts):
        if attempt < len(initial_values):
            initial = initial_values[attempt]
        else:
            initial = generator.uniform(-np.pi, np.pi, phase_count)
        solution = least_squares(
            residual,
            initial,
            jac=jacobian,
            xtol=1.0e-14,
            ftol=1.0e-14,
            gtol=1.0e-14,
            max_nfev=600,
        )
        error = np.max(np.abs(residual(solution.x)))
        if error < best_error:
            best_phases = solution.x
            best_error = error
        if best_error < 1.0e-12:
            break
    if best_error > 1.0e-9:
        raise RuntimeError(f"phase residual is {best_error:.3e}")
    return best_phases, best_error


def save_phases(path, phases):
    with open(path, "w", newline="") as stream:
        csv.writer(stream).writerow(
            [f"{float(value):.17g}" for value in phases]
        )


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(RANDOM_SEED)
    for R in R_VALUES:
        cosine, sine = jacobi_anger_coefficients(R, TAU)
        cosine_phases, cosine_error = solve_phases(
            cosine, 2 * R + 1, generator
        )
        sine_phases, sine_error = solve_phases(
            sine, 2 * R + 2, generator
        )
        save_phases(
            OUTPUT_DIRECTORY / f"cos{int(TAU)}x_R{R}.csv",
            cosine_phases,
        )
        save_phases(
            OUTPUT_DIRECTORY / f"sin{int(TAU)}x_R{R}.csv",
            sine_phases,
        )
        print(
            f"R={R}: cosine residual={cosine_error:.3e}, "
            f"sine residual={sine_error:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
