from qulacs import QuantumCircuit as QulacsCircuit
from qulacs.gate import DenseMatrix,SparseMatrix, to_matrix_gate, CNOT, RX, RZ, H
from qulacs import QuantumState
import numpy as np
import math
from scipy.linalg import sqrtm
import csv
from mpi4py import MPI
from scipy.special import comb
from scipy.sparse import lil_matrix, csr_matrix, issparse,  eye, vstack, hstack, diags, coo_matrix
from mpmath import mp
import time
from scipy.sparse import save_npz,load_npz


"""
Function file for 2D KvN-embedded QSVT Hamiltonian simulation (Qulacs version, MPI-parallel version)
    Changes in the MPI version
        from scipy.special import comb
        'cos{}x.csv'
        'sin{}x.csv'
        qs=QuantumState(num_qubits, use_multi_cpu=True)

    Required variables
    Time step width: tau
    Total simulation time: T
    Number of time steps: Total_steps = int(T/tau)
    Number of grid points in x-space: num_grid
    Spatial grid spacing in x: delta_x
    Scale normalization factor: Lambda (factor to reduce the nonlinear term)
    Density: density
    Permittivity: epsilon_0
    Mass: mass
    Charge: q
    Number of variables: N = 5*num_grid
    Upper bound on total particle number: m
    Size of Hamiltonian matrix: M = comb(m+N, m)
    Number of x-qubits: n_x = math.floor(np.log2(M))+1
    Number of ancilla qubits needed for U: n_a = 1
"""


def load_phi_vec(dt):
    """
        Function to load precomputed phase (phi) values for QSVT
    """
    with open('output/cos{}x.csv'.format(dt)) as f:
        reader = csv.reader(f)
        for row in reader:
            cos_phi_vec = np.array(row)
    cos_phi_vec = [float(s) for s in cos_phi_vec]

    with open('output/sin{}x.csv'.format(dt)) as f:
        reader = csv.reader(f)
        for row in reader:
            sin_phi_vec = np.array(row)
    sin_phi_vec = [float(s) for s in sin_phi_vec]

    return cos_phi_vec, sin_phi_vec


def list_to_number_speedup_sparse(lil, len_n):
    """
        Function to assign a number from a given list lil (lil format) and the length of the list len_n
    """
    sum_k = np.sum(lil.data[0])
    n=0
    if sum_k == 0:
        return n
    elif sum_k > 1:
        n = math.comb(sum_k+len_n-1, len_n) -1

    for k in reversed(lil.rows[0]):
        nk = lil[0, k]
        n += (math.comb(sum_k+k, k)-math.comb(sum_k-nk+k, k))
        sum_k -= nk

    return n+1

def number_to_list_speedup_sparse(num, len_list, upper_bound_m):
    """
        Function to return a list (lil format) of length len_list corresponding to number num and upper bound of total particle number upper_bound_m
    """
    ans_list = lil_matrix((1, len_list), dtype=int)
    if num == 0:
        return ans_list

    left = 1
    right = upper_bound_m

    while left < right:
        m = (left + right) // 2
        sum_m = math.comb(len_list+m, m) - 1
        if num <= sum_m:
            right = int(m)
        else:
            left = int(m+1)

    m = int(left)

    num -= (math.comb(len_list+m-1, m-1) - 1)

    left_index = 0
    right_index = len_list - 1

    while m>0:
        while left_index < right_index:
            mid_index = (left_index+right_index) // 2
            if mid_index == 0:
                sum_list = 1
            else:
                sum_list = math.comb(m+mid_index, mid_index)

            if num <= sum_list:
                right_index = int(mid_index)
            else:
                left_index = int(mid_index+1)

        if left_index == 0:
            ans_list[0, 0] = m
            return ans_list

        left_m = 1
        right_m = m
        while left_m < right_m:
            mid_m = (left_m + right_m) // 2
            sum_list = math.comb(m+left_index, left_index)-math.comb(m+left_index-mid_m-1, left_index)
            if num <= sum_list:
                right_m = int(mid_m)
            else:
                left_m = int(mid_m+1)

        ans_list[0, int(left_index)] = int(left_m)

        num -= (math.comb(m+left_index, left_index)-math.comb(m+left_index-left_m, left_index))
        m -= int(left_m)
        left_index = 0

    return ans_list



def make_creat_op_matrix_speedup_sparse(m, N, j):
    """
        Sparse matrix representation of the creation operator \hat{a}_j^\dagger for {|0>, ..., |M-1>} (M= m+N C m)
    """
    M = math.comb(m + N, m)  # Matrix size
    matrix = lil_matrix((M, M))  # Initialize sparse matrix

    for i in range(math.comb(m + N-1, m-1)):
        num_list = number_to_list_speedup_sparse(i, N, m)  # Convert number to corresponding basis list
        data = num_list[0,j] + 1
        num_list[0,j] = int(data)
        row = list_to_number_speedup_sparse(num_list, N)  # Convert basis list to number
        matrix[row, i] = np.sqrt(int(data))  # Set non-zero element

    return matrix.tocsr()  # Convert to CSR format and return


def Hamiltonian_matrix_2D_sparse(delta_x, delta_y, Lambda, density, epsilon_0, mu_0, mass, q, m, num_grid):
    """
    2D Hamiltonian creation (optimized order of creation-annihilation operator products) implemented in sparse matrix format (improved version)
    """
    num_variable = 5 * (num_grid**2)
    A_list =[]
    for i in range(num_variable):
        A_list.append(make_creat_op_matrix_speedup_sparse(m, num_variable, i))

    len_matrix = A_list[0].shape[0]

    # List to store elements of sparse matrix
    row, col, data = [], [], []
    # Add each term
    for i in range(num_grid):
        for j in range(num_grid):
            term = (-1j / (4 * delta_x * Lambda)) / 2**(1/2) * (
                (A_list[0*(num_grid**2) + ((i+1) % num_grid)*(num_grid**1) + (j % num_grid)] +
                 A_list[0*(num_grid**2) + ((i+1) % num_grid)*(num_grid**1) + (j % num_grid)].T) @
                (A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                 A_list[0*(num_grid**2) + ((i+2) % num_grid)*(num_grid**1) + (j % num_grid)].T -
                 A_list[0*(num_grid**2) + ((i+2) % num_grid)*(num_grid**1) + (j % num_grid)]) @
                A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T
            )
            term_coo = term.tocoo()  # Convert sparse format to coo format
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = (-1j / (4 * delta_y * Lambda)) / 2**(1/2) * (
                (A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+1) % num_grid)] +
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+1) % num_grid)].T) @
                (A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                 A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+2) % num_grid)].T -
                 A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+2) % num_grid)] @
                 A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T)
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = (-1j / (4 * delta_x * Lambda)) / 2**(1/2) * (
                (A_list[0*(num_grid**2) + ((i+1) % num_grid)*(num_grid**1) + (j % num_grid)] +
                 A_list[0*(num_grid**2) + ((i+1) % num_grid)*(num_grid**1) + (j % num_grid)].T) @
                (A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                 A_list[1*(num_grid**2) + ((i+2) % num_grid)*(num_grid**1) + (j % num_grid)].T -
                 A_list[1*(num_grid**2) + ((i+2) % num_grid)*(num_grid**1) + (j % num_grid)] @
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T)
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = (-1j / (4 * delta_y * Lambda)) / 2**(1/2) * (
                (A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+1) % num_grid)] +
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+1) % num_grid)].T) @
                (A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+2) % num_grid)].T -
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+2) % num_grid)] @
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T)
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = 1j * q * (density / (epsilon_0 * mass))**(1/2) * (
                A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[2*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T -
                A_list[2*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = 1j * q * (density / (epsilon_0 * mass))**(1/2) * (
                A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[3*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T -
                A_list[3*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = (1j * (mu_0 * density / mass)**(1/2)) / 2**(1/2) / Lambda * (
                (A_list[4*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] +
                 A_list[4*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T) @
                (A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T -
                 A_list[1*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                 A_list[0*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T)
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = 1j * (1 / (epsilon_0 * mu_0))**(1/2) * (1 / (2 * delta_y)) * (
                A_list[2*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[4*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+1) % num_grid)].T -
                A_list[4*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j+1) % num_grid)] @
                A_list[2*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = -1j * (1 / (epsilon_0 * mu_0))**(1/2) * (1 / (2 * delta_y)) * (
                A_list[2*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[4*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j-1) % num_grid)].T -
                A_list[4*(num_grid**2) + (i % num_grid)*(num_grid**1) + ((j-1) % num_grid)] @
                A_list[2*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = -1j * (1 / (epsilon_0 * mu_0))**(1/2) * (1 / (2 * delta_x)) * (
                A_list[3*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[4*(num_grid**2) + ((i+1) % num_grid)*(num_grid**1) + (j % num_grid)].T -
                A_list[4*(num_grid**2) + ((i+1) % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[3*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)

            term = 1j * (1 / (epsilon_0 * mu_0))**(1/2) * (1 / (2 * delta_x)) * (
                A_list[3*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[4*(num_grid**2) + ((i-1) % num_grid)*(num_grid**1) + (j % num_grid)].T -
                A_list[4*(num_grid**2) + ((i-1) % num_grid)*(num_grid**1) + (j % num_grid)] @
                A_list[3*(num_grid**2) + (i % num_grid)*(num_grid**1) + (j % num_grid)].T
            )
            term_coo = term.tocoo()
            row.extend(term_coo.row)
            col.extend(term_coo.col)
            data.extend(term_coo.data)
    # Generate final sparse matrix
    H = coo_matrix((data, (row, col)), shape=(len_matrix, len_matrix))
    return H.tocsr()


def U_Hamiltonian_matrix_sparse(delta_x, delta_y, Lambda, density, mu_0, epsilon_0, mass, q, m, num_grid):
    """
        Function to convert Hamiltonian to unitary matrix (sparse compatible)
    """
    H = Hamiltonian_matrix_2D_sparse(delta_x, delta_y, Lambda, density, epsilon_0, mu_0, mass, q, m, num_grid)
    # Normalization
    H, alpha = normalize_matrix_sparse(H)
    # Get matrix size
    rows, _ = H.shape
    #start_time = time.time()
    # If H size is a power of 2
    if (rows > 0) and (rows & (rows - 1)) == 0:
        # Calculate square root of I - H^2
        H_squared = H @ H
        I = eye(rows, format="csr")
        # Get diagonal elements
        diag_elements = (I - H_squared).diagonal()
        # Apply square root to diagonal elements
        sqrt_diag = np.sqrt(diag_elements)
        # Construct sparse matrix with diagonal elements (this approximation holds if diagonally dominant)
        Matrix_sqrt = diags(sqrt_diag, format="csr")#
        #Matrix_sqrt = csr_matrix(sqrtm((I - H_squared).toarray()))

        # Construct unitary matrix U
        top = hstack([H, 1j * Matrix_sqrt])
        bottom = hstack([-1j * Matrix_sqrt, -H])
        U = vstack([top, bottom], format="csr")
    else:
        # Expand size to power of 2
        uni_rows = 2 ** (math.floor(np.log2(rows)) + 1)
        # Store data in COO format
        rows, cols, data = [], [], []
        # Store original matrix data
        H_coo = H.tocoo()
        rows.extend(H_coo.row)
        cols.extend(H_coo.col)
        data.extend(H_coo.data)
        # Extend to H_prime with offset
        for i, j, v in zip(H_coo.row, H_coo.col, H_coo.data):
            rows.append(i)
            cols.append(j)
            data.append(v)
        # Create matrix in COO format
        H_prime = coo_matrix((data, (rows, cols)), shape=(uni_rows, uni_rows)).tocsr()
        # Calculate square root of I - H'^2
        H_squared = H_prime @ H_prime
        I = eye(uni_rows, format="csr")
        # Get diagonal elements
        diag_elements = (I - H_squared).diagonal()
        # Apply square root to diagonal elements
        sqrt_diag = np.sqrt(diag_elements)
        # Construct sparse matrix with diagonal elements
        Matrix_sqrt = diags(sqrt_diag, format="csr")
        #Matrix_sqrt = csr_matrix(sqrtm((I - H_squared).toarray()))

        # Construct unitary matrix U
        top = hstack([H_prime, 1j * Matrix_sqrt])
        bottom = hstack([-1j * Matrix_sqrt, -H_prime])
        U = vstack([top, bottom], format="csr")
    #end_time = time.time()
    #print(f"Unitary matrix construction: {end_time - start_time:.2f} seconds")
    return U, alpha


def statevector_measurement(n_x, qc, qs):
    """
        Measure the quantum circuit using the statevector
    """
    num_qubits = qc.get_qubit_count()
    qc.update_quantum_state(qs)

    state_vector = qs.get_vector().real
    ret = np.zeros(2**num_qubits)
    for i in range(len(state_vector)):
        # The bitstring is represented in the order ibin = x00…a1a0bcd
        ibin = bin(i)[2:].zfill(int(np.log2(len(state_vector))))
        if ibin[n_x:] == '0' * (num_qubits - n_x):
            ret[int(ibin[:n_x], 2)] = state_vector[i]
    psi = np.zeros(2**n_x)
    for i in range(2**n_x):
        psi[i] = ret[i]

    return psi


def C_Pi_NOT_gate(qc,target_qubit,num_control_qubits):
    """
        CΠ NOT gate (= X*Π + I*(I-Π))
            num_control_qubits: number of control qubits
    """
    # CΠ NOT gate
    multi_controlled_not_gate = DenseMatrix(target_qubit, [[0,1],[1,0]])  # (target qubit index, matrix)
    for i in range(num_control_qubits):
        multi_controlled_not_gate.add_control_qubit(2+i,0)  # (control qubit index, activation condition)
    qc.add_gate(multi_controlled_not_gate)


def v_signal_processing(qc, q_c, q_b, n_a, cos_phi, sin_phi):
    # CΠ NOT gate
    C_Pi_NOT_gate(qc,q_b,n_a)
    # e^(-iφ_cos Z)
    controlled_rz_gate = DenseMatrix(q_b, [[np.exp(-1j*cos_phi),0],[0,np.exp(1j*cos_phi)]])  # RZ(q_b, 2*cos_phi_vec[2*j])
    controlled_rz_gate.add_control_qubit(q_c,0)
    qc.add_gate(controlled_rz_gate)
    # e^(-iφ_sin Z)
    controlled_rz_gate = DenseMatrix(q_b, [[np.exp(-1j*sin_phi),0],[0,np.exp(1j*sin_phi)]])  # RZ(q_b, 2*sin_phi_vec[2*j+1])
    controlled_rz_gate.add_control_qubit(q_c,1)
    qc.add_gate(controlled_rz_gate)
    # CΠ NOT gate
    C_Pi_NOT_gate(qc,q_b,n_a)


def V_gate_sparse(qc, n_a, n_x, U, cos_phi_vec, sin_phi_vec):
    """
        Block encoding of exp(-iHdt)=cos()-isin()
    """
    l_cos = len(cos_phi_vec) # 2R     + 1
    l_sin = len(sin_phi_vec) # 2R + 1 + 1

    q_c = 0
    q_b = 1

    # for U gate(as qubits)
    u_qubits_list = list(range(2+n_a, 2+n_a+n_x))+[2] # Qubit arrangement for block encoding of U using ancilla q_a

    # append Hadamard gate to quantum registers
    qc.add_gate(H(q_c))
    qc.add_gate(H(q_b))

    for j in reversed(range(l_sin // 2)):
        if j != 0:
            ##### STEP1 #####
            # signal processing
            v_signal_processing(qc, q_c, q_b, n_a, cos_phi_vec[2*j], sin_phi_vec[2*j+1])
            # U gate
            qc.add_gate(SparseMatrix(u_qubits_list, U))

            ##### STEP2 #####
            # signal processing
            v_signal_processing(qc, q_c, q_b, n_a, cos_phi_vec[2*j-1], sin_phi_vec[2*j])
            # U* gate
            qc.add_gate(SparseMatrix(u_qubits_list, U.conj().T))

        else:
            ##### STEP3 #####
            # signal processing
            v_signal_processing(qc, q_c, q_b, n_a, cos_phi_vec[0], sin_phi_vec[1])
            # controlled U gate
            dim = int(2**(len(u_qubits_list) + 1))  # Matrix size
            target_dim = 2**len(u_qubits_list)

            # Set COO format elements in batch
            rows, cols, data = [], [], []
            # Add matrix elements for control bit |1>
            U_coo = U.tocoo()
            rows.extend(U_coo.row + target_dim)
            cols.extend(U_coo.col + target_dim)
            data.extend(U_coo.data)
            identity_sparse = eye(target_dim, format="coo", dtype=np.complex128)
            rows.extend(identity_sparse.row)
            cols.extend(identity_sparse.col)
            data.extend(identity_sparse.data)

            # Create sparse matrix in COO format
            controlled_u_matrix = coo_matrix((data, (rows, cols)), shape=(dim, dim))
            # Convert to CSR format if needed
            controlled_u_matrix = controlled_u_matrix.tocsr()

            # c-U gate
            qc.add_gate(SparseMatrix(u_qubits_list+[q_c], controlled_u_matrix))

            ##### STEP4 #####
            # signal processing
            # CΠ NOT gate
            C_Pi_NOT_gate(qc,q_b,n_a)
            # e^(-iφs Z)
            controlled_rz_gate = DenseMatrix(q_b, [[np.exp(-1j*sin_phi_vec[0]),0],[0,np.exp(1j*sin_phi_vec[0])]])#RZ(q_b, 2*sin_phi_vec[2*j+1])
            controlled_rz_gate.add_control_qubit(q_c,1)
            qc.add_gate(controlled_rz_gate)
            # CΠ NOT gate
            C_Pi_NOT_gate(qc,q_b,n_a)

    # phase gate(λ=-π/2)
    p_gate = DenseMatrix(q_c, [[1,0],[0,np.exp(-1j*np.pi/2)]])#Phase gate(-iπ/2)
    qc.add_gate(p_gate)
    # append Hadamard gate to quantum registers 'h', 'p'
    qc.add_gate(H(q_c))
    qc.add_gate(H(q_b))



def HS_TestSim_U_Hamiltonian_matrix_sparse(n_x, n_a, tau, delta_x, delta_y, Lambda, density, mu_0, epsilon_0, mass, q, m, num_grid, N, M, ux, uy, Ex, Ey, Bz, Total_steps):
    """
        Test execution function for QSVT Hamiltonian simulation: sequential time evolution with fine tau
            Input: n_x, n_a, tau, delta_x, Lambda, density, epsilon_0, mass, q, m, num_grid, M, x
                tau = alpha*T
            Output: u, E, alpha
    """
    # Convert to 1D array
    ux_flatten = ux[:,:,0].ravel()
    uy_flatten = uy[:,:,0].ravel()
    Ex_flatten = Ex[:,:,0].ravel()
    Ey_flatten = Ey[:,:,0].ravel()
    Bz_flatten = Bz[:,:,0].ravel()
    # Initial state preparation
    x = np.concatenate([ux_flatten, uy_flatten, Ex_flatten, Ey_flatten, Bz_flatten])
    psi_now, norm_a, norm_n = state_preparation_sparse_mp(N,M,x,Lambda,m)
    # Quantum circuit preparation
    num_qubits = 1 + 1 + n_a + n_x
    qs = initialize_quantum_state(n_a, num_qubits, psi_now)
    qc = QulacsCircuit(num_qubits)
    qc.update_quantum_state(qs)
    # Hamiltonian
    start_time2 = time.time()
    U, alpha = U_Hamiltonian_matrix_sparse(delta_x,delta_y,Lambda,density,mu_0,epsilon_0,mass,q,m,num_grid)
    end_time2 = time.time()
    print(f"Execution time (make_hamiltonian): {end_time2 - start_time2:.2f} seconds")
    # Save U
    save_npz("normalized_U_matrix_ng{}.npz".format(num_grid), U)
    np.savez("alpha_value_U_ng{}.npz".format(num_grid), alpha=alpha)  # Save `alpha` to npz
    #U = load_npz("normalized_U_matrix_ng{}.npz".format(num_grid))
    #data = np.load("alpha_value_U_ng{}.npz".format(num_grid))  # Load the `.npz` file
    #alpha_loaded = data["alpha"]
    cos_phi_vec, sin_phi_vec = load_phi_vec(tau)

    pi_Nby4 = mp.power(np.pi,(N/4))
    pi_n = float(mp.floor(mp.log10(pi_Nby4)))
    pi_a = float(pi_Nby4 / mp.power(10,pi_n))

    psi = psi_now
    # Time evolution
    start_time3 = time.time()
    for t in range(1, Total_steps + 1):
        V_gate_sparse(qc, n_a, n_x, U, cos_phi_vec, sin_phi_vec)
        # Measurement
        psi = statevector_measurement(n_x, qc, qs)
        # Quantum circuit initialization, next time step state preparation
        norm = np.sqrt(sum([abs(_) ** 2 for _ in psi]))
        psi /= norm
        qc = QulacsCircuit(num_qubits)
        qs.set_zero_state()
        qs = initialize_quantum_state(n_a, num_qubits,psi)
        qc.update_quantum_state(qs)
        # Restore normalization
        data = psi * norm_a / Lambda * pi_a * 10**(norm_n+pi_n) / 2 ** (1 / 2) #* 2 ** (2 / 2)
        # Extract physical quantities
        for i in range(num_grid):
            for j in range(num_grid):
                ux[i, j, t] = data[j + i * num_grid+1]
                uy[i, j, t] = data[j + i * num_grid + num_grid**2+1]
                Ex[i, j, t] = data[j + i * num_grid + 2*num_grid**2+1]
                Ey[i, j, t] = data[j + i * num_grid + 3*num_grid**2+1]
                Bz[i, j, t] = data[j + i * num_grid + 4*num_grid**2+1]
        psi_now = psi
        print("step={}".format(t))
    end_time3 = time.time()
    print(f"Execution time (time evolution): {end_time3 - start_time3:.2f} seconds")
    return ux, uy, Ex, Ey, Bz, alpha


def normalize_matrix(A):
    """
        Function to normalize a matrix
    """
    # Calculate the sum of squares of all elements of matrix A
    norm_factor = np.sqrt(np.sum(np.abs(A)**2))
    # Verify that the normalization factor is not zero
    if norm_factor == 0:
        raise ValueError("The norm of the matrix is zero; cannot normalize.")
    # Normalize matrix A
    A_normalized = A / norm_factor

    return A_normalized, norm_factor

def normalize_matrix_sparse(A):
    """
        Function to normalize a matrix
    """
    norm_factor = np.sqrt(A.multiply(A.conjugate()).sum())

    # Verify that the normalization factor is not zero
    if norm_factor == 0:
        raise ValueError("The norm of the matrix is zero; cannot normalize.")

    # Normalize matrix A
    A_normalized = A / norm_factor

    return A_normalized, norm_factor.real


def normalized(F,fac=[]):
    """
        Function to normalize a 1D array
            F: given 1D array
            fac: normalization coefficient
    """
    f_norm = np.sqrt(sum([abs(f_)**2 for f_ in F]))
    fac.append(f_norm)
    F=F/fac[0]

    return F,fac

def initialize_quantum_state(n_a, num_qubits, psi):
    # Create quantum state
    qs = QuantumState(num_qubits)
    qs.set_zero_state()
    # Create array for psi otimes ancilla qubit state
    psi_anc = np.zeros(2**(num_qubits))
    for i in range(len(psi)):
        psi_anc[i*2**(2+n_a)] = psi[i]
    # Set initial state
    qs.load(psi_anc)

    return qs

def initialize_quantum_state_sparse(n_a, num_qubits, psi):
    # Create quantum state
    qs = QuantumState(num_qubits)
    qs.set_zero_state()
    # Create array for psi otimes ancilla qubit state
    psi_anc = np.zeros(2**(num_qubits), dtype=complex)
    for i in range(psi.nnz):
        psi_anc[int(psi.rows[i][0])*2**(2+n_a)] = complex(psi.data[i][0])
    # Set initial state
    qs.load(psi_anc)

    return qs

def normmalized_hermite_notreturn(n,x):
    """
        Function to calculate Hermite polynomial (non-recursive)
    """
    if n==-1:
        return 0
    elif n==0:
        return np.pi**(-1/4)
    elif n>0:
        value0 = 0
        value1 = np.pi**(-1/4)
        for i in range(n):
            tmp0 = value1
            value1 = (2/(i+1))**(1/2)*x*value1-((i)/(i+1))**(1/2)*value0
            value0 = tmp0

        return value1


def state_preparation_sparse_mp(N, M, x, Lambda, m):
    """
    Function to prepare initial state psi(x) (using mpmath for high-precision calculation)
    return:
        mpf(coeff,norm)
    """
    Lambda = Lambda
    x_tilde = Lambda*x

    # Manage `coeff` as a list of `mpf`
    coeff = [mp.mpf(0) for _ in range(M)]

    for i in range(M):
        num_list = number_to_list_speedup_sparse(i, N, m)
        C = mp.mpf(1.0)
        nz = N - len(num_list.data[0])
        C *= mp.power(mp.pi, mp.mpf(-1/4) * nz)  # High-precision calculation using `mpmath` `pi`

        for j in range(len(num_list.data[0])):
            C *= normmalized_hermite_notreturn(int(num_list.data[0][j]),x_tilde[int(num_list.rows[0][j])])
        coeff[i] = mp.mpf(C)
    # Calculate L2 norm using `mpmath`
    norm = mp.sqrt(mp.fsum(mp.power(c,2) for c in coeff))
    # Normalize `coeff`
    coeff = [c / norm for c in coeff]
    # transform mpmath to float
    complex_coeff = np.array([float(mp.mpf(val)) for val in coeff], dtype=np.float64)
    # number of n in norm a*10^n
    norm_n = float(mp.floor(mp.log10(norm)))
    # number of a in norm a*10^n
    norm_a = float(norm / (mp.power(10, norm_n)))
    # Zero padding
    target_size = 2**(math.floor(np.log2(M)) + 1)
    if M < target_size:
        complex_coeff  = np.pad(complex_coeff, (0, target_size - M), mode="constant")

    return complex_coeff, norm_a, norm_n

def state_preparation_sparse_mp_expm(N, M, x, Lambda, m):
    """
    Function to prepare initial state psi(x) (using mpmath for high-precision calculation)
    return:
        mpf(coeff,norm)
    """
    Lambda = Lambda
    x_tilde = Lambda*x

    # Manage `coeff` as a list of `mpf`
    coeff = [mp.mpf(0) for _ in range(M)]

    for i in range(M):
        num_list = number_to_list_speedup_sparse(i, N, m)
        C = mp.mpf(1.0)
        nz = N - len(num_list.data[0])
        C *= mp.power(mp.pi, mp.mpf(-1/4) * nz)  # High-precision calculation using `mpmath` `pi`

        for j in range(len(num_list.data[0])):
            C *= normmalized_hermite_notreturn(int(num_list.data[0][j]),x_tilde[int(num_list.rows[0][j])])
        coeff[i] = mp.mpf(C)
    # Calculate L2 norm using `mpmath`
    norm = mp.sqrt(mp.fsum(mp.power(c,2) for c in coeff))
    # Normalize `coeff`
    coeff = [c / norm for c in coeff]
    # transform mpmath to float
    complex_coeff = np.array([float(mp.mpf(val)) for val in coeff], dtype=np.float64)
    # number of n in norm a*10^n
    norm_n = float(mp.floor(mp.log10(norm)))
    # number of a in norm a*10^n
    norm_a = float(norm / (mp.power(10, norm_n)))

    return complex_coeff, norm_a, norm_n
