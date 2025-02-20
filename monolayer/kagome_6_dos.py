import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

r1 = 1/2 * np.array([np.sqrt(3),1])
r2 = 1/2 * np.array([np.sqrt(3),-1])
r3 = np.array([0,1])
r4 = r2-r3
r5 = r1+r3
r6 = r1+r2

class hamiltonian:
    def __init__(self, eps, t, t2, deps, dt):
        self.eps = eps
        self.t = t
        self.t2 = t2
        self.deps = deps
        self.dt = dt
        
    def matrix_NN(self,k):
        term_1 = c_function(k, r1) * gm1 - s_function(k, r1) * gm4 + c_function(k, r2) * gm2 - s_function(k, r2) * gm5 + 2 * c_function(k, r3) * gm3 
        term_2 = c_function(k, r1) * gm1 + s_function(k, r1) * gm4 + c_function(k, r2) * gm2 + s_function(k, r2) * gm5
        term_1 = np.kron(term_1, I2)
        term_2 = np.kron(term_2, px)
        return -self.t * (term_1+term_2)
    
    def matrix_NNN(self,k):
        term_1 = c_function(k, r4) * gm1 - s_function(k, r4) * gm4 + c_function(k, r5) * gm2 - s_function(k, r5) * gm5 
        term_2 = c_function(k, r4) * gm1 + s_function(k, r4) * gm4 + c_function(k, r5) * gm2 + s_function(k, r5) * gm5 + 2 * c_function(k, r6) * gm3
        term_1 = np.kron(term_1, I2)
        term_2 = np.kron(term_2, px)
        return -self.t2 * (term_1+term_2)

    def matrix_0(self):
        return self.eps * I6

    def matrix_onsite_diff(self):
        return self.deps/6 * (np.kron((2*I3+3*gm7+np.sqrt(3)*gm8), pz))
    
    def matrix_staggered(self,k):
        return -2 * self.dt * s_function(k, r3) * np.kron(gm6, pz)

eps = 0.01
t = 0.42
t2 = 0.03
deps = -0.033
dt = 0.01

general = hamiltonian(eps, t, t2, deps, dt)
Nk = 1000

kx_values = np.linspace(-4*np.pi, 4*np.pi, Nk)
ky_values = np.linspace(-4*np.pi, 4*np.pi, Nk)
kx, ky = np.meshgrid(kx_values, ky_values)
all_eigvals = []

# Diagonalize at each k
for i in range(Nk):
    for j in range(Nk):
        # Build Hamiltonian
        total_matrix = (general.matrix_NN([kx[i,j],ky[i,j]]) 
                        + general.matrix_NNN([kx[i,j],ky[i,j]]) 
                        + general.matrix_0() 
                        + general.matrix_onsite_diff() 
                        + general.matrix_staggered([kx[i,j],ky[i,j]]))
        
        total_matrix = (total_matrix + total_matrix.conj().T) / 2


        eigvals, _ = np.linalg.eigh(total_matrix)
        eigvals = np.sort(eigvals.real)
        all_eigvals.extend(eigvals)

# --- Compute DOS ---
all_energies = np.array(all_eigvals)  # Convert eigenvalues to 1D array

# Set high resolution for better divergence capture
num_bins = 100000
energy_min, energy_max = all_energies.min() - 0.1, all_energies.max() + 0.1
energy_bins = np.linspace(energy_min, energy_max, num_bins)

dos, bin_edges = np.histogram(all_energies, bins=energy_bins, density=True)

print(len(dos), len(bin_edges))

# Smooth the histogram using Gaussian KDE
#kde = gaussian_kde(all_energies, bw_method=0.05)  # Reduce bandwidth to capture sharper features
#dos_kde = kde(energy_bins)  

with open("dos_kagome_6.txt", "w") as f:
    for i in range(len(dos)):
        f.write(str(dos[i]) + " " + str(bin_edges[i]) + "\n")

# Plot using bin_edges[:-1] for positions, np.diff(bin_edges) for widths
fig, ax = plt.subplots()
ax.bar(bin_edges[:-1], dos, width=np.diff(bin_edges), align="edge", color="blue", edgecolor="black", alpha=0.7)

ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Density of States (DOS)")
ax.set_title("Unsmoothed DOS")

plt.show()
