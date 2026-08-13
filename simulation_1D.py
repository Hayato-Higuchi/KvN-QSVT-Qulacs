import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class Case1D:
    label: str
    num_grid: int
    max_particles: int
    steps: int
    delta_x: float
    Lambda: float
    density: float
    epsilon_0: float
    mass: float
    charge: float
    tau: float = 1.0

    @property
    def num_variables(self):
        return 2 * self.num_grid

    @property
    def dimension(self):
        return math.comb(self.max_particles + self.num_variables,
                         self.max_particles)

    @property
    def num_qubits(self):
        return math.floor(math.log2(self.dimension)) + 1

    @property
    def plasma_frequency(self):
        return abs(self.charge) * math.sqrt(
            self.density / (self.epsilon_0 * self.mass)
        )

    @property
    def domain_length(self):
        return self.num_grid * self.delta_x


CASE_A = Case1D("A", 8, 1, 200, 1.0, 1.0e4, 1.0, 1.0, 1.0, -1.0)
CASE_B = (
    Case1D("B", 8, 2, 210, 1.0, 1.0, 1.0, 1.0, 100.0, -1.0),
    Case1D("B", 8, 3, 710, 1.0, 1.0, 1.0, 1.0, 100.0, -1.0),
    Case1D("B", 8, 4, 2056, 1.0, 1.0, 1.0, 1.0, 100.0, -1.0),
)
CASE_C_LENGTH = 44.0
CASE_C = (
    Case1D(
        "C", 11, 2, 128, CASE_C_LENGTH / 11,
        1.0, 1.0, 1.0, 100.0, -1.0,
    ),
    Case1D(
        "C", 22, 2, 250, CASE_C_LENGTH / 22,
        1.0, 1.0, 1.0, 100.0, -1.0,
    ),
    Case1D(
        "C", 33, 2, 374, CASE_C_LENGTH / 33,
        1.0, 1.0, 1.0, 100.0, -1.0,
    ),
    Case1D(
        "C", 44, 2, 500, CASE_C_LENGTH / 44,
        1.0, 1.0, 1.0, 100.0, -1.0,
    ),
)

DEFAULT_QSVT_R = 5
CASE_B_NORM_QSVT_R = 3
CASE_C_QSVT_R = 2


def number_to_occupations(index, length):
    occupations = [0] * length
    if index == 0:
        return occupations
    total = 1
    cumulative = math.comb(length, length - 1)
    while index > cumulative:
        total += 1
        cumulative += math.comb(total + length - 1, length - 1)
    cumulative -= math.comb(total + length - 1, length - 1)
    for mode in reversed(range(1, length)):
        occupation = 0
        cumulative += math.comb(total + mode - 1, mode - 1)
        while index > cumulative:
            occupation += 1
            cumulative += math.comb(total - occupation + mode - 1, mode - 1)
        cumulative -= math.comb(total - occupation + mode - 1, mode - 1)
        total -= occupation
        occupations[mode] = occupation
    occupations[0] = total
    return occupations


def occupation_basis(max_particles, num_variables):
    dimension = math.comb(max_particles + num_variables, max_particles)
    return np.asarray(
        [number_to_occupations(index, num_variables)
         for index in range(dimension)],
        dtype=np.int16,
    )


def creation_operators(basis, max_particles):
    dimension, num_variables = basis.shape
    lookup = {tuple(row): index for index, row in enumerate(basis)}
    totals = basis.sum(axis=1)
    operators = []
    for mode in range(num_variables):
        columns = np.flatnonzero(totals < max_particles)
        rows = np.empty(columns.size, dtype=np.int32)
        values = np.empty(columns.size, dtype=np.float64)
        for offset, column in enumerate(columns):
            occupation = basis[column].copy()
            occupation[mode] += 1
            rows[offset] = lookup[tuple(occupation)]
            values[offset] = math.sqrt(occupation[mode])
        operators.append(
            csr_matrix(
                (values, (rows, columns)),
                shape=(dimension, dimension),
                dtype=np.complex128,
            )
        )
    return operators


def hamiltonian_matrix(case, basis=None):
    if basis is None:
        basis = occupation_basis(case.max_particles, case.num_variables)
    creation = creation_operators(basis, case.max_particles)
    dimension = basis.shape[0]
    hamiltonian = csr_matrix((dimension, dimension), dtype=np.complex128)
    coupling = 1j * case.charge * math.sqrt(
        case.density / (case.epsilon_0 * case.mass)
    )
    for grid in range(case.num_grid):
        hamiltonian += coupling * (
            creation[grid] @ creation[grid + case.num_grid].T
            - creation[grid + case.num_grid] @ creation[grid].T
        )
    advection = -1j / (
        4.0 * case.delta_x * case.Lambda * math.sqrt(2.0)
    )
    for grid in range(case.num_grid):
        left = (grid - 1) % case.num_grid
        right = (grid + 1) % case.num_grid
        hamiltonian += advection * (
            (creation[grid] + creation[grid].T)
            @ (
                creation[left] @ creation[right].T
                - creation[right] @ creation[left].T
            )
        )
    hamiltonian.sum_duplicates()
    hamiltonian.eliminate_zeros()
    alpha = math.sqrt(
        float(hamiltonian.multiply(hamiltonian.conjugate()).sum().real)
    )
    return hamiltonian / alpha, alpha, basis


class StateEncoder:
    def __init__(self, basis, Lambda):
        self.basis = np.asarray(basis)
        self.Lambda = float(Lambda)
        self.max_particles = int(self.basis.max())

    def prepare(self, values):
        scaled = self.Lambda * np.asarray(values, dtype=float)
        ratios = np.empty((scaled.size, self.max_particles + 1), dtype=float)
        ratios[:, 0] = 1.0
        if self.max_particles >= 1:
            ratios[:, 1] = math.sqrt(2.0) * scaled
        for order in range(1, self.max_particles):
            ratios[:, order + 1] = (
                math.sqrt(2.0 / (order + 1))
                * scaled
                * ratios[:, order]
                - math.sqrt(order / (order + 1)) * ratios[:, order - 1]
            )
        coefficients = np.ones(self.basis.shape[0], dtype=float)
        for mode in range(scaled.size):
            coefficients *= ratios[mode, self.basis[:, mode]]
        norm = np.linalg.norm(coefficients)
        return (
            coefficients / norm,
            norm / (self.Lambda * math.sqrt(2.0)),
        )


def initial_values(case):
    if case.label == "A":
        velocity = np.ones(case.num_grid)
    else:
        coordinates = (
            np.arange(case.num_grid, dtype=float) - case.num_grid / 2
        ) * case.delta_x
        length = case.num_grid * case.delta_x
        velocity = np.sin(-(2.0 * np.pi / length) * coordinates)
    electric = np.zeros(case.num_grid)
    return np.concatenate([velocity, electric])


def _phase_matrix(phi):
    return np.array(
        [[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]],
        dtype=np.complex128,
    )


def _reflection_qsp_values(nodes, phases):
    nodes = np.asarray(nodes, dtype=float)
    complement = np.sqrt(np.clip(1.0 - nodes**2, 0.0, None))
    reflection = np.empty((nodes.size, 2, 2), dtype=np.complex128)
    reflection[:, 0, 0] = nodes
    reflection[:, 0, 1] = complement
    reflection[:, 1, 0] = complement
    reflection[:, 1, 1] = -nodes
    product = np.broadcast_to(
        _phase_matrix(phases[0]), (nodes.size, 2, 2)
    ).copy()
    for phase in phases[1:]:
        product = product @ reflection @ _phase_matrix(phase)
    return product[:, 0, 0].real


def load_phases(path):
    with open(path, newline="") as stream:
        return np.asarray(next(csv.reader(stream)), dtype=float)


def qsvt_polynomial_coefficients(cos_path, sin_path):
    cos_phases = load_phases(cos_path)
    sin_phases = load_phases(sin_path)
    degree = max(cos_phases.size - 1, sin_phases.size - 1)
    count = max(64, 4 * (degree + 1))
    nodes = np.cos(np.pi * (np.arange(count) + 0.5) / count)
    cos_values = _reflection_qsp_values(nodes, cos_phases)
    sin_values = _reflection_qsp_values(nodes, sin_phases)
    cos_coefficients = chebyshev.chebfit(
        nodes, cos_values, cos_phases.size - 1
    )
    sin_coefficients = chebyshev.chebfit(
        nodes, sin_values, sin_phases.size - 1
    )
    coefficients = np.zeros(degree + 1, dtype=np.complex128)
    coefficients[:cos_coefficients.size] += cos_coefficients
    coefficients[:sin_coefficients.size] -= 1j * sin_coefficients
    return coefficients


def apply_chebyshev_polynomial(hamiltonian, state, coefficients):
    state = np.asarray(state, dtype=np.complex128)
    result = coefficients[0] * state
    if coefficients.size == 1:
        return result
    previous = state
    current = hamiltonian @ state
    result += coefficients[1] * current
    for degree in range(2, coefficients.size):
        following = 2.0 * (hamiltonian @ current) - previous
        result += coefficients[degree] * following
        previous, current = current, following
    return result


def propagate_expm(case, hamiltonian, encoder, values=None, progress=False):
    if values is None:
        values = initial_values(case)
    trajectory = np.empty((case.num_variables, case.steps + 1), dtype=float)
    trajectory[:, 0] = values
    state, scale = encoder.prepare(values)
    generator = -1j * case.tau * hamiltonian
    interval = max(1, case.steps // 10)
    for step in range(1, case.steps + 1):
        evolved = expm_multiply(generator, state, traceA=0.0)
        evolved /= np.linalg.norm(evolved)
        values = evolved[1:case.num_variables + 1].real * scale
        trajectory[:, step] = values
        state, scale = encoder.prepare(values)
        if progress and (step % interval == 0 or step == case.steps):
            print(f"{case.label} expm {step}/{case.steps}", flush=True)
    return trajectory


def propagate_qsvt(
    case,
    hamiltonian,
    encoder,
    coefficients,
    values=None,
    progress=False,
):
    if values is None:
        values = initial_values(case)
    trajectory = np.empty((case.num_variables, case.steps + 1), dtype=float)
    trajectory[:, 0] = values
    state, scale = encoder.prepare(values)
    interval = max(1, case.steps // 10)
    for step in range(1, case.steps + 1):
        evolved = apply_chebyshev_polynomial(
            hamiltonian, state, coefficients
        )
        values = evolved[1:case.num_variables + 1].real * scale
        trajectory[:, step] = values
        state, scale = encoder.prepare(values)
        if progress and (step % interval == 0 or step == case.steps):
            print(f"{case.label} qsvt {step}/{case.steps}", flush=True)
    return trajectory


def output_paths(case, method, output_root="output", qsvt_R=None):
    if qsvt_R is not None and method != "qsvt":
        raise ValueError("qsvt_R can only be specified for QSVT output")
    output_root = Path(output_root)
    order_part = f"_R{qsvt_R}" if qsvt_R is not None else ""
    if case.label == "A":
        directory = output_root / "CaseA"
        stem = "1DPlasmaOscillationTest"
        method_part = "expm_" if method == "expm" else ""
        suffix = (
            f"nx_{case.num_qubits}_delta_x_{case.delta_x:g}"
            f"_T_{case.steps * case.tau:g}_delta_t_{case.tau:g}"
            f"_m_{case.max_particles}{order_part}"
        )
        return {
            "u": directory / f"{stem}_{method_part}u_{suffix}.npy",
            "E": directory / f"{stem}_{method_part}E_{suffix}.npy",
        }
    directory = output_root / "CaseBC"
    prefix = f"1DAdvectionTest_{method}"
    suffix = (
        f"numgrid_{case.num_grid}_nx_{case.num_qubits}"
        f"_delta_x_{case.delta_x:g}_T_{case.steps * case.tau:g}"
        f"_delta_t_{case.tau:g}_m_{case.max_particles}{order_part}"
    )
    return {
        "u": directory / f"{prefix}_u_{suffix}.npy",
        "E": directory / f"{prefix}_E_{suffix}.npy",
    }


def save_trajectory(
    case, method, trajectory, output_root="output", qsvt_R=None
):
    paths = output_paths(case, method, output_root, qsvt_R=qsvt_R)
    paths["u"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["u"], trajectory[:case.num_grid])
    np.save(paths["E"], trajectory[case.num_grid:])
    return paths


def load_trajectory(case, method, output_root="output", qsvt_R=None):
    paths = output_paths(case, method, output_root, qsvt_R=qsvt_R)
    return np.vstack([np.load(paths["u"]), np.load(paths["E"])])


def run_case(case, method, output_root="output", coefficients=None,
             progress=False, qsvt_R=None):
    basis = occupation_basis(case.max_particles, case.num_variables)
    hamiltonian, alpha, basis = hamiltonian_matrix(case, basis)
    encoder = StateEncoder(basis, case.Lambda)
    if method == "expm":
        trajectory = propagate_expm(
            case, hamiltonian, encoder, progress=progress
        )
    elif method == "qsvt":
        if qsvt_R is None:
            qsvt_R = (
                CASE_C_QSVT_R
                if case.label == "C"
                else DEFAULT_QSVT_R
            )
        if coefficients is None:
            coefficients = qsvt_polynomial_coefficients(
                Path(output_root)
                / f"qsvt_phases/cos1x_R{qsvt_R}.csv",
                Path(output_root)
                / f"qsvt_phases/sin1x_R{qsvt_R}.csv",
            )
        trajectory = propagate_qsvt(
            case,
            hamiltonian,
            encoder,
            coefficients,
            progress=progress,
        )
    else:
        raise ValueError("method must be 'expm' or 'qsvt'")
    output_R = (
        qsvt_R
        if method == "qsvt" and qsvt_R != DEFAULT_QSVT_R
        else None
    )
    paths = save_trajectory(
        case, method, trajectory, output_root, qsvt_R=output_R
    )
    metadata = {
        "case": case.label,
        "num_grid": case.num_grid,
        "domain_length": case.domain_length,
        "delta_x": case.delta_x,
        "m": case.max_particles,
        "alpha": alpha,
        "tau": case.tau,
        "physical_dt": case.tau / alpha,
        "steps": case.steps,
        "physical_final_time": case.steps * case.tau / alpha,
        "omega_dt": case.plasma_frequency * case.tau / alpha,
        "qsvt_R": qsvt_R if method == "qsvt" else "",
    }
    return trajectory, metadata, paths


def all_cases():
    return (CASE_A,) + CASE_B + CASE_C


def write_metadata(records, path="output/CaseABC_time_steps.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case",
        "num_grid",
        "domain_length",
        "delta_x",
        "m",
        "alpha",
        "tau",
        "physical_dt",
        "steps",
        "physical_final_time",
        "omega_dt",
        "qsvt_R",
    )
    unique = {
        (record["case"], record["num_grid"], record["m"]): record
        for record in records
    }
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(unique):
            writer.writerow(unique[key])
    return path
