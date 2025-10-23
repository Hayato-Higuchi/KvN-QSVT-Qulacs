# Koopman-von Neumann Linearized Electromagnetic Fluid Simulation using Quantum Singular Value Transformation

Quantum algorithm implementation for simulating electromagnetic fluid dynamics using the Koopman-von Neumann (KvN) framework with Quantum Singular Value Transformation (QSVT).

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
@article{ito2024quantum,
  title={Quantum Algorithm for Electromagnetic Fluid Simulation based on Koopman-von Neumann Mechanics},
  author={Ito, Yuki and Higuchi, Hayato and Sakamoto, Kazuki and Yoshikawa, Akimasa and Fujii, Keisuke},
  journal={arXiv preprint arXiv:2509.22503},
  year={2024},
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
