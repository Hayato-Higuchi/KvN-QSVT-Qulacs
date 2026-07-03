import time
import numpy as np
from mpi4py import MPI 
from scipy.special import comb
from sub_function_2D import *

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Start timing (synchronize across all processes)
comm.barrier()
start_time = time.time()

# Time step width
tau = 25
# Number of time steps
Total_steps = 2000
# Total time
T = Total_steps * tau
# Number of spatial grid points
num_grid = 20
# Spatial grid spacing
delta_x = 1
delta_y = 1
# Scale normalization factor
Lambda = 10**0  # Factor to reduce the nonlinear term; 1 may also be fine
# Permeability
mu_0 = 1
# Density
density = 1
# Permittivity
epsilon_0 = 1
# Mass
mass = 1
# Charge
q = -1
# Number of variables
N = 5 * num_grid**2
# Upper bound on total particle number
m = 2
# Size of the Hamiltonian matrix
M = int(comb(m + N, m)) 
# Number of x-qubits
n_x = math.floor(np.log2(M)) + 1
# Number of ancilla qubits required for U
n_a = 1
eta = 1
T_real = Total_steps * tau / (192 * eta * (m / 2)**(5 / 2))

k = 2 * np.pi / num_grid
x_min = -num_grid / 2
x_max = num_grid / 2

# u arrays
ux = np.zeros((num_grid, num_grid, Total_steps + 1))
uy = np.zeros((num_grid, num_grid, Total_steps + 1))
ux[:, :, 0] = np.ones((num_grid, num_grid))
uy[:, :, 0] = np.ones((num_grid, num_grid))
# E, B arrays
Ex = np.zeros((num_grid, num_grid, Total_steps + 1))
Ey = np.zeros((num_grid, num_grid, Total_steps + 1))
Bz = np.zeros((num_grid, num_grid, Total_steps + 1))
# Convert to 1D arrays
ux_flatten = ux[:, :, 0].T.ravel()
uy_flatten = uy[:, :, 0].T.ravel()
Ex_flatten = Ex[:, :, 0].T.ravel()
Ey_flatten = Ey[:, :, 0].T.ravel()
Bz_flatten = Bz[:, :, 0].T.ravel()
# Prepare initial state
x = np.concatenate([ux_flatten, uy_flatten, Ex_flatten, Ey_flatten, Bz_flatten])

ux, uy, Ex, Ey, Bz, alpha = HS_TestSim_U_Hamiltonian_matrix_sparse(n_x, n_a, tau, delta_x, delta_y, Lambda, density, mu_0, epsilon_0, mass, q, m, num_grid, N, M, ux, uy, Ex, Ey, Bz, Total_steps)

# Stop timing (synchronize across all processes)
comm.barrier()
end_time = time.time()

# Show result on process 0
if rank == 0:
    print(f"Execution time (parallel): {end_time - start_time:.2f} seconds")

# Save in binary format
filename = 'output/CaseD/2DKelvin-Helmholtz_u_numgrid_{}_nx_{}_delta_x_{}_T_{}_delta_t_{}_m_{}.npy'.format(num_grid, n_x, delta_x, T, tau, m)
np.save(filename, u)
filename = 'output/CaseD/2DKelvin-Helmholtz_E_numgrid_{}_nx_{}_delta_x_{}_T_{}_delta_t_{}_m_{}.npy'.format(num_grid, n_x, delta_x, T, tau, m)
np.save(filename, E)
