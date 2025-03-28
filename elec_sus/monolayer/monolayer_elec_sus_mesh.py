"""
Mesh electronic susceptibility calculation for monolayer
BZ folding
"""
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.linalg import eigh 


# Define FD distribution, Hamiltonian, BZ folding

def fd_dist(energy, mu, beta=1000):
    x = (energy - mu) * beta
    if x > 15:
        return 0.0
    elif x < -15:
        return 1.0
    else:
        return 1.0 / (np.exp(x) + 1.0)


def fold_to_bz(k, kmin, kmax):
    L = kmax - kmin
    return ((k - kmin) % L) + kmin


N = 6

def c_function(k_vec, r_vec):
    return np.cos(np.dot(k_vec, r_vec))

def s_function(k_vec, r_vec):
    return np.sin(np.dot(k_vec, r_vec))

gm1 = np.array([[0,1,0], [1,0,0], [0,0,0]], dtype=complex)
gm2 = np.array([[0,0,1], [0,0,0], [1,0,0]], dtype=complex)
gm3 = np.array([[0,0,0], [0,0,1], [0,1,0]], dtype=complex)
gm4 = np.array([[0,-1j,0], [1j,0,0], [0,0,0]], dtype=complex)
gm5 = np.array([[0,0,-1j], [0,0,0], [1j,0,0]], dtype=complex)
gm6 = np.array([[0,0,0], [0,0,-1j], [0,1j,0]], dtype=complex)
gm7 = np.array([[1,0,0], [0,-1,0], [0,0,0]], dtype=complex)
gm8 = (1/np.sqrt(3)) * np.array([[1,0,0], [0,1,0], [0,0,-2]], dtype=complex)

px = np.array([[0,1], [1,0]], dtype=complex)
py = np.array([[0,-1j], [1j,0]], dtype=complex)
pz = np.array([[1,0], [0,-1]], dtype=complex)

I2 = np.eye(2)
I3 = np.eye(3)
I6 = np.eye(6)

r1 = 0.5 * np.array([np.sqrt(3), 1])
r2 = 0.5 * np.array([np.sqrt(3), -1])
r3 = np.array([0,1])
r4 = r2 - r3
r5 = r1 + r3
r6 = r1 + r2

class hamiltonian:
    def __init__(self, eps, t, t2, deps, dt):
        self.eps = eps
        self.t = t
        self.t2 = t2
        self.deps = deps
        self.dt = dt
        
    def matrix_NN(self, k):
        term_1 = (c_function(k, r1)*gm1 - s_function(k, r1)*gm4 + c_function(k, r2)*gm2 - s_function(k, r2)*gm5 + 2*c_function(k, r3)*gm3)
        term_2 = (c_function(k, r1)*gm1 + s_function(k, r1)*gm4 + c_function(k, r2)*gm2 + s_function(k, r2)*gm5)
        term_1 = np.kron(term_1, I2)
        term_2 = np.kron(term_2, px)
        return -self.t * (term_1 + term_2)
    
    def matrix_NNN(self, k):
        term_1 = (c_function(k, r4)*gm1 - s_function(k, r4)*gm4 + c_function(k, r5)*gm2 - s_function(k, r5)*gm5)
        term_2 = (c_function(k, r4)*gm1 + s_function(k, r4)*gm4 + c_function(k, r5)*gm2 + s_function(k, r5)*gm5 + 2*c_function(k, r6)*gm3)
        term_1 = np.kron(term_1, I2)
        term_2 = np.kron(term_2, px)
        return -self.t2 * (term_1 + term_2)

    def matrix_0(self):
        return self.eps * I6

    def matrix_onsite_diff(self):
        return (self.deps/6) * np.kron((2*I3 + 3*gm7 + np.sqrt(3)*gm8), pz)
    
    def matrix_staggered(self, k):
        return -2 * self.dt * s_function(k, r3) * np.kron(gm6, pz)

# Model parameters
eps = 0.01
t = 0.42
t2 = 0.03
deps = -0.033
dt = 0.01
general = hamiltonian(eps, t, t2, deps, dt)

# Define the Brillouin zone (BZ)
b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2

# Define q-mesh for susceptibility
Nq = 200
qxs = np.linspace(-b1*1.5, b1*1.5, Nq, endpoint=False)
qys = np.linspace(-b2*1.5, b2*1.5, Nq, endpoint=False)

# Define k-mesh for band structure
Nx, Ny = 200, 348
kx_vals = np.linspace(-b1*1.5, b1*1.5, Nx)
ky_vals = np.linspace(-b2*1.5, b2*1.5, Ny)

def compute_chi(qx, qy, energies, kx_vals, ky_vals, mu):

    Nx, Ny, N_bands = energies.shape
    chi_val = 0.0

    for i in range(Nx):
        for j in range(Ny):
            E_k = energies[i, j, :]
            # shift (kx, ky) by (qx, qy)
            kx_ = fold_to_bz(kx_vals[i] + qx, -b1, b1)
            ky_ = fold_to_bz(ky_vals[j] + qy, -b2, b2)
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



energies = np.zeros((Nx, Ny, 6), dtype=float)

#  Compute eigenvalues at each k-point
for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        k_vec = np.array([kx, ky])
        H_k = (general.matrix_NN(k_vec) + general.matrix_NNN(k_vec) + general.matrix_0() + general.matrix_onsite_diff() + general.matrix_staggered(k_vec))
        H_k = 0.5 * (H_k + H_k.conj().T)

        eigvals, _ = np.linalg.eigh(H_k)
        energies[i, j, :] = np.sort(eigvals.real)

print("Energy calculation complete.")


# Prepare q-list and calculation of chi(q)

# Create a list of (iqx, iqy, qx, qy) to parallelise
q_list = []
for iqx, qx in enumerate(qxs):
    for iqy, qy in enumerate(qys):
        q_list.append((iqx, iqy, qx, qy))

# store results
chi_map = np.zeros((Nq, Nq), dtype=float)

def compute_one_q(args):
    iqx, iqy, qx, qy = args
    chi_val = compute_chi(qx, qy, energies, kx_vals, ky_vals, mu=-0.006)
    print(f"Computed χ(qx={qx:.3f}, qy={qy:.3f}) = {chi_val:.5f}", flush=True)
    return iqx, iqy, chi_val

results = Parallel(n_jobs=32, verbose=5)(
    delayed(compute_one_q)(arg) for arg in q_list
)

for (iqx, iqy, val) in results:
    chi_map[iqy, iqx] = val

print("Susceptibility calculation complete.")

# Save results to file

with open("chi_map_-6mev.txt", "w") as f:
    f.write("# qx qy chi(qx, qy)\n")
    for i, qx in enumerate(qxs):
        for j, qy in enumerate(qys):
            f.write(f"{qx:.6f} {qy:.6f} {chi_map[j, i]:.6e}\n")

print("Results saved to chi_map_-6mev.txt.")

