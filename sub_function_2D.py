from qulacs import QuantumCircuit as QulacsCircuit
from qulacs.gate import DenseMatrix, to_matrix_gate, CNOT, RX, RZ, H
from qulacs import QuantumState
import numpy as np
import math
from scipy.linalg import sqrtm
import csv
from mpi4py import MPI 
from scipy.special import comb

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
    with open('cos{}x.csv'.format(dt)) as f:
        reader = csv.reader(f)
        for row in reader:
            cos_phi_vec = np.array(row)
    cos_phi_vec = [float(s) for s in cos_phi_vec]
    
    with open('sin{}x.csv'.format(dt)) as f:
        reader = csv.reader(f)
        for row in reader:
            sin_phi_vec = np.array(row)
    sin_phi_vec = [float(s) for s in sin_phi_vec]

    return cos_phi_vec, sin_phi_vec


def list_to_number(ns):
    """    
        Assign a unique number from a given list ns
    """
    sum_k = np.sum(ns)
    len_n = len(ns)
    n=0
    if sum_k == 0:
        return n
    elif sum_k > 1:
        for j in range(1, sum_k):
            n += int(comb(j+len_n-1, len_n-1))
    for k in reversed(range(1, len_n)):
        nk = ns[k]
        if nk > 0:
            for j in range(nk):
                n += int(comb(sum_k-j+k-1, k-1))
        sum_k -= nk
    
    return n+1


def number_to_list(num, len_list):
    """    
        Return a list of length len_list corresponding to number num
    """
    ans_list = [0]*len_list
    if num == 0:
        return ans_list
    m = 1
    sum_m = int(comb(1+len_list-1, len_list-1))
    while num > sum_m:
        m += 1
        sum_m += int(comb(m+len_list-1, len_list-1))
    sum_m -= int(comb(m+len_list-1, len_list-1))
    for i in reversed(range(1, len_list)):
        n=0
        sum_m += int(comb(m+i-1, i-1))
        while num > sum_m:
            n += 1
            sum_m += int(comb(m-n+i-1, i-1))
        sum_m -= int(comb(m-n+i-1, i-1))
        m -= n
        ans_list[i]=n
    ans_list[0]=m
    
    return ans_list


def make_creat_op_matrix(m, N, j):
    """
        Matrix representation of the creation operator \hat{a}_j^\dagger with respect to {|0>, ..., |M-1>} (M = C(m+N, m))
    """
    M = int(comb(m+N, m))
    matrix = np.zeros((M, M))
    for i in range(M):
        num_list = number_to_list(i, N)
        if sum(num_list) < m:
            num_list[j] += 1
            matrix[list_to_number(num_list)][i] += np.sqrt(num_list[j])

    return matrix


def Hamiltonian_matrix_2D(delta_x,delta_y,Lambda,density,epsilon_0,mu_0,mass,q,m,num_grid):
    """
        Construct the 2D Hamiltonian (with an ordering of products of creation/annihilation operators)
        # x_0 = u_x
        # x_1 = u_y
        # x_2 = E_x
        # x_3 = E_y
        # x_4 = B_z
    """
    A_list = []
    num_variable = 5 * (num_grid**2)
    for i in range(num_variable):
        A_list.append(make_creat_op_matrix(m, num_variable, i))

    len_matrix = A_list[0].shape[0]

    H = np.zeros((len_matrix,len_matrix), dtype=complex)
    # Order following Eq. 36 and onward
    for i in range(num_grid):
        for j in range(num_grid):
            # ux,a+1,b  ux,a+2,b
            H += (-1j/ (4*delta_x*Lambda)) / 2**(1/2)  * \
                ( (A_list[0*(num_grid**2) + ((i+1)%num_grid)*(num_grid**1) + (j%num_grid)] \
                    + A_list[0*(num_grid**2) + ((i+1)%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) \
                    @ (A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[0*(num_grid**2) + ((i+2)%num_grid)*(num_grid**1) + (j%num_grid)].transpose() \
                        - A_list[0*(num_grid**2) + ((i+2)%num_grid)*(num_grid**1) + (j%num_grid)]) \
                             @A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose() )
    
    for i in range(num_grid):
        for j in range(num_grid):
            H += (-1j/ (4*delta_y*Lambda)) / 2**(1/2)  * \
                ( (A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+1)%num_grid)] \
                    + A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+1)%num_grid)].transpose()) \
                    @ (A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+2)%num_grid)].transpose() \
                        - A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+2)%num_grid)] \
                            @A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) )

    for i in range(num_grid):
        for j in range(num_grid):
            H += (-1j/ (4*delta_x*Lambda)) / 2**(1/2)  * \
                ( (A_list[0*(num_grid**2) + ((i+1)%num_grid)*(num_grid**1) + (j%num_grid)] \
                    + A_list[0*(num_grid**2) + ((i+1)%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) \
                    @ (A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[1*(num_grid**2) + ((i+2)%num_grid)*(num_grid**1) + (j%num_grid)].transpose() \
                        - A_list[1*(num_grid**2) + ((i+2)%num_grid)*(num_grid**1) + (j%num_grid)] \
                            @A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) )
    
    for i in range(num_grid):
        for j in range(num_grid):
            H += (-1j/ (4*delta_y*Lambda)) / 2**(1/2)  * \
                ( (A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+1)%num_grid)] \
                    + A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+1)%num_grid)].transpose()) \
                    @ (A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+2)%num_grid)].transpose() \
                        - A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+2)%num_grid)] \
                            @A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) )
    
    for i in range(num_grid):
        for j in range(num_grid):
            H += 1j*q*(density/(epsilon_0*mass))**(1/2)  * \
                    (A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[2*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose() \
                        - A_list[2*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                            @A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) 

    for i in range(num_grid):
        for j in range(num_grid):
            H += 1j*q*(density/(epsilon_0*mass))**(1/2)  * \
                    (A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[3*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose() \
                        - A_list[3*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                            @A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) 

    for i in range(num_grid):
        for j in range(num_grid):
            H += (1j*(mu_0*density/mass)**(1/2)) / 2**(1/2) / Lambda  * \
                ( (A_list[4*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    + A_list[4*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) \
                    @ (A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose() \
                        - A_list[1*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                            @A_list[0*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) )

    for i in range(num_grid):
        for j in range(num_grid):
            H += 1j*(1/(epsilon_0*mu_0))**(1/2) * (1/(2*delta_y))   * \
                    (A_list[2*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[4*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+1)%num_grid)].transpose() \
                        - A_list[4*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j+1)%num_grid)] \
                            @A_list[2*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) 

    for i in range(num_grid):
        for j in range(num_grid):
            H += -1j*(1/(epsilon_0*mu_0))**(1/2) * (1/(2*delta_y))   * \
                    (A_list[2*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[4*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j-1)%num_grid)].transpose() \
                        - A_list[4*(num_grid**2) + (i%num_grid)*(num_grid**1) + ((j-1)%num_grid)] \
                            @A_list[2*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) 

    for i in range(num_grid):
        for j in range(num_grid):
            H += -1j*(1/(epsilon_0*mu_0))**(1/2) * (1/(2*delta_x))   * \
                    (A_list[3*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[4*(num_grid**2) + ((i+1)%num_grid)*(num_grid**1) + (j%num_grid)].transpose() \
                        - A_list[4*(num_grid**2) + ((i+1)%num_grid)*(num_grid**1) + (j%num_grid)] \
                            @A_list[3*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) 

    for i in range(num_grid):
        for j in range(num_grid):
            H += 1j*(1/(epsilon_0*mu_0))**(1/2) * (1/(2*delta_x))   * \
                    (A_list[3*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)] \
                    @A_list[4*(num_grid**2) + ((i-1)%num_grid)*(num_grid**1) + (j%num_grid)].transpose() \
                        - A_list[4*(num_grid**2) + ((i-1)%num_grid)*(num_grid**1) + (j%num_grid)] \
                            @A_list[3*(num_grid**2) + (i%num_grid)*(num_grid**1) + (j%num_grid)].transpose()) 

    return H


def U_Hamiltonian_matrix(delta_x,delta_y,Lambda,density,mu_0,epsilon_0,mass,q,m,num_grid):
    """
        Unitarize the Hamiltonian matrix
    """
    H = Hamiltonian_matrix_2D(delta_x,delta_y,Lambda,density,epsilon_0,mu_0,mass,q,m,num_grid)
    # Normalization
    H, alpha = normalize_matrix(H)
    # Get matrix size
    rows, _ = H.shape
    # If size of H is a power of two
    if (rows > 0) and (rows & (rows - 1)) == 0:
        U = np.zeros((2*rows,2*rows), dtype=complex)
        H_squared = np.dot(H, H)
        I = np.eye(rows)
        Matrix_sqrt = sqrtm(I-H_squared)
        for i in range(rows):
            for j in range(rows):
                U[i,j] = H[i,j]
                U[i+rows,j] = 1j*Matrix_sqrt[i,j]
                U[i,j+rows] = -1j*Matrix_sqrt[i,j]
                U[i+rows,j+rows] = -H[i,j]
    else:
        uni_rows = 2**(math.floor(np.log2(rows))+1)
        U = np.zeros((2**(math.floor(np.log2(rows))+2),2**(math.floor(np.log2(rows))+2)),dtype=complex)
        H_prime = np.zeros((uni_rows,uni_rows), dtype=complex)
        for i in range(rows):
            for j in range(rows):
                H_prime[i,j] = H[i,j]
        H_squared = np.dot(H_prime, H_prime)
        I = np.eye(uni_rows)
        Matrix_sqrt = sqrtm(I-H_squared)
        for i in range(uni_rows):
            for j in range(uni_rows):
                U[i,j] = H_prime[i,j]
                U[i+uni_rows,j] = 1j*Matrix_sqrt[i,j]
                U[i,j+uni_rows] = -1j*Matrix_sqrt[i,j]
                U[i+uni_rows,j+uni_rows] = -H_prime[i,j]

    return U,alpha


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
 

def V_gate(qc, n_a, n_x, U, cos_phi_vec, sin_phi_vec):
    """
        Block-encoding of exp(-iH dt) = cos() - i sin()
    """
    l_cos = len(cos_phi_vec) # 2R     + 1
    l_sin = len(sin_phi_vec) # 2R + 1 + 1

    q_c = 0
    q_b = 1

    # For the U gate (as qubits)
    u_qubits_list = list(range(2+n_a, 2+n_a+n_x))+[2]  # Allocate ancilla q_a for the qubit layout used in block-encoding U

    # Append Hadamard gates to the quantum registers
    qc.add_gate(H(q_c))
    qc.add_gate(H(q_b))

    for j in reversed(range(l_sin // 2)):
        if j != 0:
            ##### STEP1 #####
            # signal processing
            v_signal_processing(qc, q_c, q_b, n_a, cos_phi_vec[2*j], sin_phi_vec[2*j+1])
            # U gate
            qc.add_gate(DenseMatrix(u_qubits_list, U))

            ##### STEP2 #####
            # signal processing
            v_signal_processing(qc, q_c, q_b, n_a, cos_phi_vec[2*j-1], sin_phi_vec[2*j])
            # U* gate
            qc.add_gate(DenseMatrix(u_qubits_list, U.conj().T))

        else:
            ##### STEP3 #####
            # signal processing
            v_signal_processing(qc, q_c, q_b, n_a, cos_phi_vec[0], sin_phi_vec[1])
            # controlled U gate
            controlled_u_gate = DenseMatrix(u_qubits_list, U)
            controlled_u_gate.add_control_qubit(q_c,1)
            qc.add_gate(controlled_u_gate)

            ##### STEP4 #####
            # signal processing
            # CΠ NOT gate
            C_Pi_NOT_gate(qc,q_b,n_a)
            # e^(-iφs Z)
            controlled_rz_gate = DenseMatrix(q_b, [[np.exp(-1j*sin_phi_vec[0]),0],[0,np.exp(1j*sin_phi_vec[0])]])  # RZ(q_b, 2*sin_phi_vec[2*j+1])
            controlled_rz_gate.add_control_qubit(q_c,1)
            qc.add_gate(controlled_rz_gate)
            # CΠ NOT gate
            C_Pi_NOT_gate(qc,q_b,n_a)

    # Phase gate (λ = -π/2)
    p_gate = DenseMatrix(q_c, [[1,0],[0,np.exp(-1j*np.pi/2)]])  # Phase gate (-iπ/2)
    qc.add_gate(p_gate)
    # Append Hadamard gates to the quantum registers 'h', 'p'
    qc.add_gate(H(q_c))
    qc.add_gate(H(q_b))


def HS_TestSim_U_Hamiltonian_matrix(n_x, n_a, tau, delta_x, delta_y, Lambda, density, mu_0, epsilon_0, mass, q, m, num_grid, N, M, ux, uy, Ex, Ey, Bz, Total_steps):
    """
        Test runner for QSVT Hamiltonian simulation: evolve sequentially with fine-grained tau
            Input: n_x, n_a, tau, delta_x, Lambda, density, epsilon_0, mass, q, m, num_grid, M, x
                tau = alpha*T
            Output: u, E, alpha
    """
    psi = np.zeros((2**(math.floor(np.log2(M)) + 1), Total_steps + 1))
    # Flatten to 1D arrays
    ux_flatten = ux[:,:,0].T.ravel()
    uy_flatten = uy[:,:,0].T.ravel()
    Ex_flatten = Ex[:,:,0].T.ravel()
    Ey_flatten = Ey[:,:,0].T.ravel()
    Bz_flatten = Bz[:,:,0].T.ravel()
    # Prepare initial state
    x = np.concatenate([ux_flatten, uy_flatten, Ex_flatten, Ey_flatten, Bz_flatten])
    psi[:,0], norm_psi = state_preparation(N,M,x,Lambda)
    # Prepare quantum circuit
    num_qubits = 1 + 1 + n_a + n_x
    qs = initialize_quantum_state(n_a, num_qubits, psi[:,0])
    qc = QulacsCircuit(num_qubits)
    qc.update_quantum_state(qs)
    # Hamiltonian
    U, alpha = U_Hamiltonian_matrix(delta_x,delta_y,Lambda,density,mu_0,epsilon_0,mass,q,m,num_grid)
    cos_phi_vec, sin_phi_vec = load_phi_vec(tau)
    
    # Time evolution
    for t in range(1, Total_steps + 1):
        V_gate(qc, n_a, n_x, U, cos_phi_vec, sin_phi_vec)
        # Measurement
        psi[:, t] = statevector_measurement(n_x, qc, qs)
        # Re-initialize the circuit and prepare the state for the next time step
        norm = np.sqrt(sum([abs(_) ** 2 for _ in psi[:, t]]))
        qc = QulacsCircuit(num_qubits)
        qs.set_zero_state()
        qs = initialize_quantum_state(n_a, num_qubits, psi[:,t]/norm)
        qc.update_quantum_state(qs)
        # Undo normalization
        psi[:, t] = psi[:, t] * norm_psi / Lambda * np.pi ** (N / 4) / 2 ** (1 / 2) * 2 ** (2 / 2) 
        # Extract physical quantities
        for i in range(num_grid):
            for j in range(num_grid):
                ux[i, j, t] = psi[i + j * num_grid+1, t]
                uy[i, j, t] = psi[i + j * num_grid + num_grid**2+1, t]
                Ex[i, j, t] = psi[i + j * num_grid + 2*num_grid**2+1, t]
                Ey[i, j, t] = psi[i + j * num_grid + 3*num_grid**2+1, t]
                Bz[i, j, t] = psi[i + j * num_grid + 4*num_grid**2+1, t]
            
        print("step={}".format(t))

    return ux, uy, Ex, Ey, Bz, alpha


def normalize_matrix(A):
    """
        Normalize a matrix
    """
    # Compute the sum of squares of all elements of A
    norm_factor = np.sqrt(np.sum(np.abs(A)**2))
    # Ensure the normalization factor is nonzero
    if norm_factor == 0:
        raise ValueError("The norm of the matrix is zero; cannot normalize.")
    # Normalize A
    A_normalized = A / norm_factor
    
    return A_normalized, norm_factor


def normalized(F,fac=[]):
    """
        Normalize a 1D array
            F: given 1D array
            fac: normalization factor
    """
    f_norm = np.sqrt(sum([abs(f_)**2 for f_ in F]))
    fac.append(f_norm)
    F=F/fac[0]

    return F,fac


def is_unitary(matrix):
    """
        Check whether a matrix is unitary
    """
    # Conjugate transpose of the matrix
    conjugate_transpose = np.conjugate(matrix).T
    # Check whether the product equals the identity
    product = np.dot(matrix, conjugate_transpose)
    is_identity = np.allclose(product, np.eye(matrix.shape[0]))
    # Check whether the inverse exists and equals the conjugate transpose
    has_inverse = np.linalg.inv(matrix) is not None
    is_hermitian = np.allclose(np.linalg.inv(matrix), conjugate_transpose)

    return is_identity and has_inverse and is_hermitian


def is_hermitian(matrix):
    """
    Check whether a matrix is Hermitian
    """
    # Ensure the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check whether it equals its conjugate transpose (up to numerical tolerance)
    return np.allclose(matrix, np.conj(matrix.T))


def initialize_quantum_state(n_a, num_qubits, psi):
    # Create the quantum state
    qs = QuantumState(num_qubits, use_multi_cpu=True)
    qs.set_zero_state()
    # Build the array for psi ⊗ ancilla-qubit state
    psi_anc = np.zeros(2**(num_qubits))
    for i in range(len(psi)):
        psi_anc[i*2**(2+n_a)] = psi[i]
    # Load the initial state
    qs.load(psi_anc)
    
    return qs

def normmalized_hermite(n,x):
    """
        Compute normalized Hermite polynomials
    """
    if n==-1:
        return 0
    elif n==0:
        return np.pi**(-1/4)
    elif n>0:
        return (2/(n))**(1/2)*x*normmalized_hermite(n-1,x)-((n-1)/(n))**(1/2) * normmalized_hermite(n-2,x)
    
    
def state_preparation(N,M,x,Lambda):
    """
        Prepare the initial state psi(x)
            Input: N, M, x, Lambda
            Output: psi(x), ||psi(x)||
    """
    x_tilde = Lambda * x
    coeff = []
    for i in range(M):
        num_list = number_to_list(i,N)
        C = 1.0
        for j in range(N):
            C = C*normmalized_hermite(num_list[j],x_tilde[j])
        coeff = np.append(coeff,C)
    norm = np.sqrt(sum([abs(_)**2 for _ in coeff]))
    coeff = coeff/norm
    if M < 2**(math.floor(np.log2(M))+1):
        for _ in range(2**(math.floor(np.log2(M))+1)-M):
            coeff = np.append(coeff, 0)
    return coeff,norm
