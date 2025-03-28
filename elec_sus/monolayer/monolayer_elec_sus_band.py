"""
Band electronic susceptibility calculation for monolayer
No BZ folding
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.linalg import eigh  # for eigenvalue calculation


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

N = 6  # number of bands

def c_function(k_vec, r_vec):
    return np.cos(np.dot(k_vec, r_vec))

def s_function(k_vec, r_vec):
    return np.sin(np.dot(k_vec, r_vec))

gm1 = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
gm2 = np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex)
gm3 = np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex)
gm4 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex)
gm5 = np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex)
gm6 = np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex)
gm7 = np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex)
gm8 = (1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=complex)

px = np.array([[0,1],[1,0]], dtype=complex)
py = np.array([[0,-1j],[1j,0]], dtype=complex)
pz = np.array([[1,0],[0,-1]], dtype=complex)

I2 = np.eye(2)
I3 = np.eye(3)
I6 = np.eye(6)

r1 = 0.5 * np.array([np.sqrt(3),1])
r2 = 0.5 * np.array([np.sqrt(3),-1])
r3 = np.array([0,1])
r4 = r2 - r3
r5 = r1 + r3
r6 = r1 + r2

class hamiltonian:
    def __init__(self, eps, t, t2, deps, dt):
        self.eps = eps; self.t = t; self.t2 = t2; self.deps = deps; self.dt = dt
        
    def matrix_NN(self, k):
        term_1 = c_function(k, r1)*gm1 - s_function(k, r1)*gm4 + c_function(k, r2)*gm2 - s_function(k, r2)*gm5 + 2*c_function(k, r3)*gm3
        term_2 = c_function(k, r1)*gm1 + s_function(k, r1)*gm4 + c_function(k, r2)*gm2 + s_function(k, r2)*gm5
        term_1 = np.kron(term_1, I2)
        term_2 = np.kron(term_2, px)
        return -self.t * (term_1 + term_2)
    
    def matrix_NNN(self, k):
        term_1 = c_function(k, r4)*gm1 - s_function(k, r4)*gm4 + c_function(k, r5)*gm2 - s_function(k, r5)*gm5
        term_2 = c_function(k, r4)*gm1 + s_function(k, r4)*gm4 + c_function(k, r5)*gm2 + s_function(k, r5)*gm5 + 2*c_function(k, r6)*gm3
        term_1 = np.kron(term_1, I2)
        term_2 = np.kron(term_2, px)
        return -self.t2 * (term_1 + term_2)
    
    def matrix_0(self):
        return self.eps * I6
    
    def matrix_onsite_diff(self):
        return (self.deps/6) * np.kron((2*I3+3*gm7+np.sqrt(3)*gm8), pz)
    
    def matrix_staggered(self, k):
        return -2 * self.dt * s_function(k, r3) * np.kron(gm6, pz)

# Set model parameters and create Hamiltonian
eps = 0.01
t = 0.42
t2 = 0.03
deps = -0.033
dt = 0.01
general = hamiltonian(eps, t, t2, deps, dt)

# Define BZ
b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2

# Define k-mesh for band structure:
Nx, Ny = 200, 348
kx_vals = np.linspace(-b1*2, b1*2, Nx)
ky_vals = np.linspace(-b2*2, b2*2, Ny)
energies = np.zeros((Nx, Ny, 6), dtype=float)
for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        k_vec = np.array([kx, ky])
        H_k = (general.matrix_NN(k_vec) + general.matrix_NNN(k_vec) +
               general.matrix_0() + general.matrix_onsite_diff() +
               general.matrix_staggered(k_vec))
        H_k = 0.5 * (H_k + H_k.conj().T)
        eigvals, _ = np.linalg.eigh(H_k)
        energies[i, j, :] = np.sort(eigvals.real)
print("Energy calculation complete.", flush=True)

# Define a High-Symmetry Path (1D)

Gamma = np.array([0.0, 0.0])
X = np.array([b1, 0.0])
M = np.array([b1, b2])
Y = np.array([0.0, b2])
points_path = [Gamma, X, M, Y, Gamma]
labels = [r'$\Gamma$', 'X', 'M', 'Y', r'$\Gamma$']

def generate_k_path(points, n_points=500):
    q_vecs = []
    for i in range(len(points)-1):
        start, end = points[i], points[i+1]
        for alpha in np.linspace(0, 1, n_points):
            q_vecs.append((1 - alpha)*start + alpha*end)
    return np.array(q_vecs)

n_points = 100
q_vecs = generate_k_path(points_path, n_points)

# Compute distance along the q-path:
q_dist = [0.0]
for i in range(1, len(q_vecs)):
    dq = np.linalg.norm(q_vecs[i] - q_vecs[i-1])
    q_dist.append(q_dist[-1] + dq)
q_dist = np.array(q_dist)

# Compute χ(q) along the path for several μ values in parallel

# Define a list of μ values in meV converted to eV:
mu_list = [-0.006, 0.0, 0.006, 0.009, 0.016, 0.026]

# Compute chi using parallel computation
def compute_chi(qx, qy, energies, kx_vals, ky_vals, mu):

    Nx, Ny, N_bands = energies.shape
    chi_val = 0.0

    for i in range(Nx):
        for j in range(Ny):
            E_k = energies[i, j, :]

            # shift (kx, ky) by (qx, qy)
            # Folding disabled due to it being a band calculation
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

# For each μ value, compute χ along the path in parallel
results = {}  
for mu in mu_list:
    chi_path = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_chi_parallel)(q, mu) for q in q_vecs
    )
    results[mu] = np.array(chi_path)
    print(f"Finished μ = {mu*1000:.1f} meV", flush=True)

# Save the results to a text file

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
print("Results saved to chi_path_results.txt.", flush=True)
