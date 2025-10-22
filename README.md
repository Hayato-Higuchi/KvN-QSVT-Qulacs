# Quantum Singular Value Transformation for Koopman-von Neumann Dynamics

This repository contains the implementation of quantum algorithms for solving fluid dynamics problems using the Koopman-von Neumann (KvN) framework with Quantum Singular Value Transformation (QSVT).

## Overview

The code implements quantum algorithms for simulating 2D incompressible fluid flows, including the Kelvin-Helmholtz instability, using QSVT-based linear solvers.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Qulacs (quantum circuit simulator)
- MPI4py (for parallel execution)

## Installation

```bash
pip install numpy scipy qulacs mpi4py
```

## Files

- `KvN_2D_function.py`: Core functions for KvN-QSVT implementation
- `KvN_2D_mpi_exec_Kelvin-Helmholtz.py`: MPI-based execution script for Kelvin-Helmholtz simulation
- `KvN_2D_comparision_QSVT_expm_KH.ipynb`: Jupyter notebook comparing QSVT and matrix exponential methods
- `KvN_2D_exec_expm.ipynb`: Jupyter notebook for matrix exponential execution

## Usage

### Single Process Execution

For Jupyter notebooks, simply open and run the cells in order:

```bash
jupyter notebook KvN_2D_comparision_QSVT_expm_KH.ipynb
```

### MPI Parallel Execution

For parallel execution using MPI:

```bash
mpiexec -n <num_processes> python KvN_2D_mpi_exec_Kelvin-Helmholtz.py
```

## Directory Structure

```
.
├── README.md
├── KvN_2D_function.py
├── KvN_2D_mpi_exec_Kelvin-Helmholtz.py
├── KvN_2D_comparision_QSVT_expm_KH.ipynb
├── KvN_2D_exec_expm.ipynb
├── circuit_data/          # Quantum circuit data
├── results/               # Simulation results
├── .gitignore
└── pyproject.toml
```

## Citation

If you use this code in your research, please cite the original paper:

[Citation information to be added]

## License

[License information to be added]

## Contact

For questions or issues, please open an issue on the GitHub repository.
