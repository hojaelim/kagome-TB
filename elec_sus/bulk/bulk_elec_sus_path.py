"""
Electronic susceptibility calcualtion for path bulk
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

N = 3

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
b1 = 4*np.pi/3
b2 = 2*np.pi/(np.sqrt(3))
Nx, Ny = 200, 348
kx_vals = np.linspace(-b1, b1, Nx)
ky_vals = np.linspace(-b2, b2, Ny)
energies = np.zeros((Nx, Ny, 3), dtype=float)

for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        k_vec = np.array([kx, ky])
        total_matrix = general.matrix_0()
        total_matrix += bloch(kx, ky, -a2, general.matrix_AB()) + bloch(kx, ky, a1, general.matrix_AC()) + bloch(kx, ky, -a3, general.matrix_BC())
        total_matrix += bloch(kx, ky, a2, general.matrix_BA()) + bloch(kx, ky, -a1, general.matrix_CA()) + bloch(kx, ky, a3, general.matrix_CB())
        total_matrix = 0.5 * (total_matrix + total_matrix.conj().T)
        eigvals, _ = np.linalg.eigh(total_matrix)
        energies[i, j, :] = np.sort(eigvals.real)

print("Energy calculation complete.", flush=True)


def generate_k_path(points, n_points):
    q_vecs = []
    for i in range(len(points)-1):
        start, end = points[i], points[i+1]
        for alpha in np.linspace(0, 1, n_points):
            q_vecs.append((1 - alpha)*start + alpha*end)
    return np.array(q_vecs)

# Define high-symmetry points

Gamma = (0.0, 0.0)
M     = (np.pi, np.pi/np.sqrt(3))
K     = (b1, 0)

points_path = np.array([Gamma, K, M, Gamma])
labels = [r'$\Gamma$', 'K', 'M', r'$\Gamma$']

n_points = 300
q_vecs = generate_k_path(points_path, n_points)

q_dist = [0.0]
for i in range(1, len(q_vecs)):
    dq = np.linalg.norm(q_vecs[i] - q_vecs[i-1])
    q_dist.append(q_dist[-1] + dq)
q_dist = np.array(q_dist)

mu_list = np.array([0])

def compute_chi(qx, qy, energies, kx_vals, ky_vals, mu):

    Nx, Ny, N_bands = energies.shape
    chi_val = 0.0

    for i in range(Nx):
        for j in range(Ny):
            E_k = energies[i, j, :]
            # shift (kx, ky) by (qx, qy)
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

def compute_chi_parallel(q, mu):
    qx, qy = q
    chi_val = compute_chi(qx, qy, energies, kx_vals, ky_vals, mu)
    print(f"Computed χ(qx={qx:.3f}, qy={qy:.3f}) for μ={mu*1000:.1f} meV = {chi_val:.5f}", flush=True)
    return chi_val

results = {}
for mu in mu_list:
    chi_path = Parallel(n_jobs=8, verbose=5)(
        delayed(compute_chi_parallel)(q, mu) for q in q_vecs
    )
    results[mu] = np.array(chi_path)
    print(f"Finished μ = {mu*1000:.1f} meV", flush=True)

# 5) Save results

with open("chi_path_results_test.txt", "w") as f:
    f.write("# q_dist    qx    qy")
    for mu in mu_list:
        f.write(f"    chi_mu_{int(mu*1000)}meV")
    f.write("\n")
    for d, q in zip(q_dist, q_vecs):
        line = f"{d:.6f} {q[0]:.6f} {q[1]:.6f}"
        for mu in mu_list:
            line += f" {results[mu][np.where(q_dist==d)[0][0]]:.6e}"
        line += "\n"
        f.write(line)
print("Results saved to chi_path_results_bulk.txt.", flush=True)