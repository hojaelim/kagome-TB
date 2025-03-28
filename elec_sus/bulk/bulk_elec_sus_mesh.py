"""
Electronic susceptibility calcualtion for mesh bulk
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Define Fermi-Dirac distribution

def fd_dist(energy, mu, beta=1000):
    x = (energy - mu) * beta
    if x > 15:
        return 0.0
    elif x < -15:
        return 1.0
    else:
        return 1.0 / (np.exp(x) + 1.0)

# Define Hamiltonian and compute eigenvalues on k-mesh

N = 3

def bloch(kx, ky, R, matrix):
    matrix = (1 + np.exp(-1j * (kx * R[0] + ky * R[1]))) * matrix
    return matrix

class hamiltonian:
    def __init__(self, E, a1, a2, a3, t, t2=0.5):
        self.E = E
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.t = -t
        self.t2 = -t2
        
    def matrix_0(self):
        matrix = np.zeros((N, N), dtype=complex)
        for i in range(N):
            matrix[i][i] = self.E
        return matrix
    
    def matrix_AB(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][1] = self.t
        return matrix
    
    def matrix_AC(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][2] = self.t
        return matrix
    
    def matrix_BA(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[1][0] = self.t
        return matrix
    
    def matrix_BC(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[1][2] = self.t
        return matrix
    
    def matrix_CA(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[2][0] = self.t
        return matrix
    
    def matrix_CB(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[2][1] = self.t
        return matrix

a1 = np.array([1, 0])
a2 = np.array([-1/2, np.sqrt(3)/2])
a3 = np.array([-1/2, -np.sqrt(3)/2])
phi = 0

general = hamiltonian(0,a1,a2,a3,1)
b1, b2 = 4*np.pi/3, 2*np.pi/np.sqrt(3)

# Define k-mesh for band structure
Nx, Ny = 200, 348
kx_vals = np.linspace(-b1, b1, Nx)
ky_vals = np.linspace(-b2, b2, Ny)
energies = np.zeros((Nx, Ny, N), dtype=float)

# Define q-mesh for susceptibility
Nq = 100
qxs = np.linspace(-2*np.pi, 2*np.pi, Nq, endpoint=False)
qys = np.linspace(-2*np.pi, 2*np.pi, Nq, endpoint=False)

def compute_chi(qx, qy, energies, kx_vals, ky_vals, mu):

    Nx, Ny, N_bands = energies.shape
    chi_val = 0.0

    for i in range(Nx):
        for j in range(Ny):
            E_k = energies[i, j, :]

            kx_ = kx_vals[i] + qx
            ky_ = ky_vals[j] + qy
            
            i_q = np.argmin(np.abs(kx_vals - kx_))
            j_q = np.argmin(np.abs(ky_vals - ky_))

            E_kq = energies[i_q, j_q, :]
            # sum over bands
            for e_n in E_k:
                f_n = fd_dist(e_n, mu)
                for e_m in E_kq:
                    f_m = fd_dist(e_m, mu)
                    if e_n != e_m:
                        denom = (e_n - e_m)
                        chi_val += (f_n - f_m) / denom

    # Normalize by total k-points
    chi_val /= (Nx * Ny)
    return -chi_val

# Compute eigenvalues at each k-point
for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        k_vec = np.array([kx, ky])
        total_matrix = general.matrix_0()
        total_matrix += bloch(kx, ky, -a2, general.matrix_AB()) + bloch(kx, ky, a1, general.matrix_AC()) + bloch(kx, ky, -a3, general.matrix_BC())
        total_matrix += bloch(kx, ky, a2, general.matrix_BA()) + bloch(kx, ky, -a1, general.matrix_CA()) + bloch(kx, ky, a3, general.matrix_CB())
        total_matrix = 0.5 * (total_matrix + total_matrix.conj().T)
        eigvals, _ = np.linalg.eigh(total_matrix)
        energies[i, j, :] = np.sort(eigvals.real)

print("Energy calculation complete.")
# Create a list of (iqx, iqy, qx, qy)
q_list = []
for iqx, qx in enumerate(qxs):
    for iqy, qy in enumerate(qys):
        q_list.append((iqx, iqy, qx, qy))

# store results
chi_map = np.zeros((Nq, Nq), dtype=float)

def compute_one_q(args):
    iqx, iqy, qx, qy = args
    chi_val = compute_chi(qx, qy, energies, kx_vals, ky_vals, mu=0)
    print(f"Computed χ(qx={qx:.3f}, qy={qy:.3f}) = {chi_val:.5f}", flush=True)
    return iqx, iqy, chi_val

# Parallel execution
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_one_q)(arg) for arg in q_list
)

for (iqx, iqy, val) in results:
    chi_map[iqy, iqx] = val

print("Susceptibility calculation complete.")


# Save results

with open("chi_map_0mev_fold.txt", "w") as f:
    f.write("# qx qy chi(qx, qy)\n")
    for i, qx in enumerate(qxs):
        for j, qy in enumerate(qys):
            f.write(f"{qx:.6f} {qy:.6f} {chi_map[j, i]:.6e}\n")

print("Results saved to chi_map_0.009mev.txt.")