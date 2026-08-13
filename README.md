# Koopman-von Neumann Linearized Electromagnetic Fluid Simulation using Quantum Singular Value Transformation

This repository provides an implementation of a quantum algorithm for simulating electromagnetic fluid dynamics using the Koopman-von Neumann (KvN) framework with Quantum Singular Value Transformation (QSVT).

## Requirements

This project implements quantum simulations for 1D and 2D electromagnetic fluid dynamics problems, including plasma oscillation tests, advection tests, and more complex scenarios like the Kelvin-Helmholtz instability.

### Prerequisites

1. **Clone this repository**
   ```bash
   git clone https://github.com/Hayato-Higuchi/KvN-QSVT-Qulacs.git
   cd KvN-QSVT-Qulacs
   ```

2. **Install Python dependencies**

   This project requires Python 3.9 or later. Install dependencies using pip:
   ```bash
   pip install -e .
   ```

   Or install with optional dependencies:
   ```bash
   # For MPI parallel execution
   pip install -e ".[mpi]"

   # For Jupyter notebook support
   pip install -e ".[notebook]"

   # For development (includes testing and linting tools)
   pip install -e ".[dev]"

   # Install all optional dependencies
   pip install -e ".[all]"
   ```

3. **Core dependencies**
   - Python >= 3.9
   - NumPy >= 1.24.0, < 2.0
   - SciPy >= 1.10.0
   - Qulacs >= 0.6.0 (quantum circuit simulator)
   - Matplotlib >= 3.6.0
   - mpmath >= 1.3.0

4. **Optional dependencies**
   - MPI4py >= 3.1.0 (for parallel execution)
   - Jupyter >= 1.0.0 (for notebook execution)

## Executing Simulations

The repository contains three types of test cases across different dimensionalities:

### 1D Cases

**Case A: Plasma Oscillation Test**
- Simulates 1D plasma oscillations using KvN-QSVT and matrix exponential methods
- Main execution notebooks:
  - `main_qsvt_1D_caseA.ipynb`: QSVT-based simulation
  - `main_expm_1D_caseA.ipynb`: Matrix exponential simulation
  - `analysis_1D_caseA.ipynb`: Results comparison and visualization

**Case B/C: Advection Test**
- Tests advection dynamics in 1D systems
- Main execution notebooks:
  - `main_qsvt_1D_caseBC.ipynb`: QSVT-based simulation
  - `main_expm_1D_caseBC.ipynb`: Matrix exponential simulation
  - `analysis_1D_caseBC.ipynb`: Results comparison and visualization

### 2D Cases

**Case D: 2D Electromagnetic Fluid Dynamics**
- Simulates 2D electromagnetic fluid dynamics with advanced phenomena
- Main execution files:
  - `main_qsvt_2D_caseD.py`: QSVT-based simulation (MPI-parallelized)
  - `main_expm_2D_caseD.ipynb`: Matrix exponential simulation
  - `analysis_2D_caseD.ipynb`: Results comparison and visualization

### Running Notebooks

For Jupyter notebook-based simulations:
```bash
jupyter notebook
```

Then open and execute the desired notebook (e.g., `main_qsvt_1D_caseA.ipynb`).

All Case A--C data and figures can also be generated in one command:

```bash
python run_1D_cases.py
```

The matrix, circuit, and time-grid checks can be run with:

```bash
python verify_1D.py
```

The normalized Hamiltonian is defined by
`H_tilde = H / alpha`, with `alpha = ||H||_F`.  Both KvN-QSVT and
KvN-expm use `tau = 1`, corresponding to the common physical step
`dt = tau / alpha` for each parameter set.  The physical variables obtained
after each step define the input state for the next step in both methods.
The main Case A and Case B KvN-QSVT results use `R = 5` phase factors.
The Case B norm-deviation comparison uses `R = 3` so that the QSVT
truncation contribution can be resolved alongside the finite-`m` KvN
deviation. This calculation and its figure can be regenerated independently
with:

```bash
python case_b_norm_deviation_1D.py
```

The Case C grid study uses `R = 2` to make the finite-`R` difference from
KvN-expm visible. It keeps the domain length fixed at `L = 44` and uses
`delta_x = L / num_grid`. It can be regenerated independently with:

```bash
python case_c_grid_convergence_1D.py
```

### Running MPI-Parallelized Simulations

For 2D simulations using MPI:
```bash
mpiexec -n <num_processes> python main_qsvt_2D_caseD.py
```

Example with 4 processes:
```bash
mpiexec -n 4 python main_qsvt_2D_caseD.py
```

## Project Structure

```
.
├── README.md                           # This file
├── pyproject.toml                      # Project configuration and dependencies
├── sub_function_1D.py                  # Core functions for 1D KvN-QSVT implementation
├── sub_function_2D.py                  # Core functions for 2D KvN-QSVT implementation (MPI-parallel)
├── simulation_1D.py                    # Case A--C numerical propagators and time grids
├── analysis_1D.py                      # Case A--C figure generation and validation
├── run_1D_cases.py                     # Complete Case A--C workflow
├── verify_1D.py                        # Matrix, circuit, and time-grid checks
├── case_b_norm_deviation_1D.py         # Case B finite-m and finite-R comparison
├── case_c_grid_convergence_1D.py       # Case C fixed-domain grid study
├── main_qsvt_1D_caseA.ipynb           # 1D QSVT simulation (Case A)
├── main_expm_1D_caseA.ipynb           # 1D matrix exponential simulation (Case A)
├── analysis_1D_caseA.ipynb            # Analysis notebook for Case A
├── main_qsvt_1D_caseBC.ipynb          # 1D QSVT simulation (Case B/C)
├── main_expm_1D_caseBC.ipynb          # 1D matrix exponential simulation (Case B/C)
├── analysis_1D_caseBC.ipynb           # Analysis notebook for Case B/C
├── main_qsvt_2D_caseD.py              # 2D QSVT simulation (Case D, MPI)
├── main_expm_2D_caseD.ipynb           # 2D matrix exponential simulation (Case D)
├── analysis_2D_caseD.ipynb            # Analysis notebook for Case D
├── circuit_data/                       # Quantum circuit data
└── output/                             # Simulation results
    ├── CaseA/                          # Case A results
    ├── CaseBC/                         # Case B/C results
    ├── CaseD/                          # Case D results
    ├── cos1x.csv, sin1x.csv           # Precomputed QSVT phase angles
    └── normalized_U_matrix_*.npz       # Precomputed unitary matrices
```


## Reproducing the figures in the paper

All figures in the paper can be regenerated from the data stored in `output/` by running the analysis notebooks. The parameter sets used for each case are:

| Figure | Case | Script(s) | Key parameters |
|---|---|---|---|
| Fig. 2 | A | `main_{qsvt,expm}_1D_caseA.ipynb` → `analysis_1D_caseA.ipynb` | `num_grid=8, m=1, tau=1, Total_steps=200, Lambda=1e4, mass=1` |
| Fig. 3 | B | `case_b_norm_deviation_1D.py` | `num_grid=8, m={2,3,4}, R=3, Total_steps={210,710,2056}, tau=1, delta_x=1, Lambda=1, mass=100` |
| RK4 comparison | B | `classical_rk4_1D_caseBC.py` | RK4 reference at the exact KvN sampling times; `dt_RK4=(tau/alpha)/128` |
| QSVT R dependence | B | `qsvt_R_sweep_1D_caseB.py` | `R={3,5,7,9}` with the same per-step state input as KvN-expm |
| Grid dependence | C | `case_c_grid_convergence_1D.py` | `L=44, m=2, R=2, num_grid={11,22,33,44}, delta_x={4,2,4/3,1}, Total_steps={128,250,374,500}` |
| Figs. 6–7 | D | `main_qsvt_2D_caseD.py` (MPI), `main_expm_2D_caseD.ipynb` → `analysis_2D_caseD.ipynb` | `num_grid=20, m=2; QSVT: tau=25, Total_steps=2000; expm: tau=500, Total_steps=100` |

The physical time of the k-th stored step is `t_k = k * tau / alpha`, where `alpha` is the Frobenius norm of the truncated Hamiltonian used for normalization (printed by the main scripts).

### Classical RK4 reference

`classical_rk4_1D_caseBC.py` integrates the same central-difference semi-discrete system with a classical fourth-order Runge-Kutta method and compares it with the KvN-expm data for `m = 2, 3, 4`, producing `output/CaseBC/1DAdvectionTest_expm_vs_RK4.pdf`.

```bash
python classical_rk4_1D_caseBC.py
```

## Key Features

- **Quantum Singular Value Transformation (QSVT)**: Implementation of QSVT-based Hamiltonian simulation for linear quantum algorithms
- **Koopman-von Neumann Framework**: Linearization of nonlinear fluid dynamics equations using the KvN formalism
- **1D and 2D Simulations**: Support for both 1D and 2D electromagnetic fluid dynamics
- **Classical Comparison**: Matrix exponential methods for benchmarking quantum algorithms
- **MPI Parallelization**: High-performance parallel execution for large-scale 2D simulations
- **Qulacs Integration**: Efficient quantum circuit simulation using the Qulacs library

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{HiguchiandIto2025prxq,
  title={A Quantum Algorithm for Nonlinear Electromagnetic Fluid Dynamics via Koopman-von~Neumann Linearization},
  author={Higuchi, Hayato and Ito, Yuki and Sakamoto, Kazuki and Fujii, Keisuke and Yoshikawa, Akimasa},
  journal={arXiv preprint arXiv:2509.22503},
  year={2025},
  url={https://arxiv.org/abs/2509.22503}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Hayato Higuchi (higuchi@qunasys.com)
- Yuki Ito
- Kazuki Sakamoto
- Keisuke Fujii
- Akimasa Yoshikawa

## Contact

For questions, issues, or contributions, please:
- Open an issue on the [GitHub repository](https://github.com/Hayato-Higuchi/KvN-QSVT-Qulacs/issues)
- Contact the maintainer at higuchi@qunasys.com

### R-dependence study

Generate the QSVT phase factors for `R = 3, 5, 7, 9` (`tau = 1`) and run the
Case-B R sweep:

```bash
python qsvt_phase_generation_1D.py
python qsvt_R_sweep_1D_caseB.py
```
