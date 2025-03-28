"""
Find and plot berry curvature and chern numbers for band plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create 2D path between two points
def line_2d(kstart, kend, npoints):

    kstart = np.array(kstart, dtype=float)
    kend   = np.array(kend,   dtype=float)
    return [kstart + (kend - kstart)*i/(npoints-1) for i in range(npoints)]

# Create specific BZ path
def build_bz_path(corners, Nseg):

    path = line_2d(corners[0], corners[1], Nseg)

    for i in range(len(corners)-2):
        path += line_2d(corners[i+1], corners[i+2], Nseg)[1:]
    
    path += line_2d(corners[-1], corners[0], Nseg)[1:]

    return path

# Define necessary variables and vectors for Hamiltonian
N = 6
phi = np.pi/6 # TRS breaking
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

# Nearest Neighbour hoppings
    def matrix_NN(self,k):
        matrix = np.zeros((6,6), dtype=complex)
        matrix[0,2] += np.exp(1j * np.dot(k,r1)) * np.exp(-1j * phi/3)
        matrix[0,3] += np.exp(-1j * np.dot(k,r1)) * np.exp(-1j * phi/3)
        matrix[0,4] += np.exp(1j * np.dot(k,r2)) * np.exp(1j * phi/3)
        matrix[0,5] += np.exp(-1j * np.dot(k,r2)) * np.exp(1j * phi/3)

        matrix[1,2] += np.exp(-1j * np.dot(k,r1)) * np.exp(-1j * phi/3)
        matrix[1,3] += np.exp(1j * np.dot(k,r1)) * np.exp(-1j * phi/3)
        matrix[1,4] += np.exp(-1j * np.dot(k,r2)) * np.exp(1j * phi/3)
        matrix[1,5] += np.exp(1j * np.dot(k,r2)) * np.exp(1j * phi/3)
        
        matrix[2,0] += np.exp(-1j * np.dot(k,r1)) * np.exp(1j * phi/3)
        matrix[2,1] += np.exp(1j * np.dot(k,r1)) * np.exp(1j * phi/3)
        matrix[2,4] += np.exp(-1j * np.dot(k,r3)) * np.exp(-1j * phi/3)
        matrix[2,4] += np.exp(1j * np.dot(k,r3)) * np.exp(-1j * phi/3)

        matrix[3,0] += np.exp(1j * np.dot(k,r1)) * np.exp(1j * phi/3)
        matrix[3,1] += np.exp(-1j * np.dot(k,r1)) * np.exp(1j * phi/3)
        matrix[3,5] += np.exp(1j * np.dot(k,r3)) * np.exp(-1j * phi/3)
        matrix[3,5] += np.exp(-1j * np.dot(k,r3)) * np.exp(-1j * phi/3)

        matrix[4,0] += np.exp(-1j * np.dot(k,r2)) * np.exp(-1j * phi/3)
        matrix[4,1] += np.exp(1j * np.dot(k,r2)) * np.exp(-1j * phi/3)
        matrix[4,2] += np.exp(-1j * np.dot(k,r3)) * np.exp(1j * phi/3)
        matrix[4,2] += np.exp(1j * np.dot(k,r3)) * np.exp(1j * phi/3)

        matrix[5,0] += np.exp(1j * np.dot(k,r2)) * np.exp(-1j * phi/3)
        matrix[5,1] += np.exp(-1j * np.dot(k,r2)) * np.exp(-1j * phi/3)
        matrix[5,3] += np.exp(1j * np.dot(k,r3)) * np.exp(1j * phi/3)
        matrix[5,3] += np.exp(-1j * np.dot(k,r3))  * np.exp(1j * phi/3)
        
        return -self.t * matrix
    
# Next nearest neighbour hoppings
    def matrix_NNN(self,k):
        matrix = np.zeros((6,6), dtype=complex)
       
        matrix[0,2] += np.exp(1j * np.dot(k,r4))
        matrix[0,3] += np.exp(-1j * np.dot(k,r4))
        matrix[0,4] += np.exp(1j * np.dot(k,r5))
        matrix[0,5] += np.exp(-1j * np.dot(k,r5))
        
        matrix[1,2] += np.exp(-1j * np.dot(k,r4))
        matrix[1,3] += np.exp(1j * np.dot(k,r4))
        matrix[1,4] += np.exp(-1j * np.dot(k,r5))
        matrix[1,5] += np.exp(1j * np.dot(k,r5))
        
        matrix[2,0] += np.exp(-1j * np.dot(k,r4))
        matrix[2,1] += np.exp(1j * np.dot(k,r4))
        matrix[2,5] += np.exp(-1j * np.dot(k,r6))
        matrix[2,5] += np.exp(1j * np.dot(k,r6))

        matrix[3,0] += np.exp(1j * np.dot(k,r4))
        matrix[3,1] += np.exp(-1j * np.dot(k,r4))
        matrix[3,4] += np.exp(1j * np.dot(k,r6))
        matrix[3,4] += np.exp(-1j * np.dot(k,r6))

        matrix[4,0] += np.exp(-1j * np.dot(k,r5))
        matrix[4,1] += np.exp(1j * np.dot(k,r5))
        matrix[4,3] += np.exp(-1j * np.dot(k,r6))
        matrix[4,3] += np.exp(1j * np.dot(k,r6))

        matrix[5,0] += np.exp(1j * np.dot(k,r5))
        matrix[5,1] += np.exp(-1j * np.dot(k,r5))
        matrix[5,2] += np.exp(1j * np.dot(k,r6))
        matrix[5,2] += np.exp(-1j * np.dot(k,r6))

        return -self.t2 * matrix

# Self hoppings
    def matrix_0(self):
        return self.eps * I6

# A site differences
    def matrix_onsite_diff(self):
        matrix = np.zeros((6,6), dtype=complex)

        matrix[0,0] += 6
        matrix[1,1] += -6

        return self.deps/6 * matrix

# Staggered hopping between B and C    
    def matrix_staggered(self,k):
        matrix = np.zeros((6,6), dtype=complex)

        matrix[2,4] += np.exp(1j * np.dot(k,r3)) 
        matrix[2,4] += -np.exp(-1j * np.dot(k,r3)) 

        matrix[3,5] += -np.exp(1j * np.dot(k,r3)) 
        matrix[3,5] += np.exp(-1j * np.dot(k,r3)) 

        matrix[4,2] += -np.exp(1j * np.dot(k,r3)) 
        matrix[4,2] += np.exp(-1j * np.dot(k,r3))

        matrix[5,3] += np.exp(1j * np.dot(k,r3))
        matrix[5,3] += -np.exp(1j * np.dot(k,r3)) 

        return self.dt * matrix

    def build_hamiltonian(self,k):
        return self.matrix_NN(k) + self.matrix_NNN(k) + self.matrix_0() + self.matrix_onsite_diff() + self.matrix_staggered(k)

# Define variables
eps = 0.01
t = 0.42
t2 = 0.03
deps = -0.033
dt = 0.01

general = hamiltonian(eps, t, t2, deps, dt)

def dHdkx(k, delta=1e-5):
    kx, ky = k
    H_plus  = general.build_hamiltonian([kx+delta, ky])
    H_minus = general.build_hamiltonian([kx-delta, ky])
    return (H_plus - H_minus)/(2*delta)

def dHdky(k, delta=1e-5):
    kx, ky = k
    H_plus  = general.build_hamiltonian([kx, ky+delta])
    H_minus = general.build_hamiltonian([kx, ky-delta])
    return (H_plus - H_minus)/(2*delta)

# Calculate berry curvature at a given point
def berry_curvature_at_k(Hk, dHdkx, dHdky):

    eigvals, eigvecs = np.linalg.eigh(Hk)

    N = len(eigvals)
    omega = np.zeros(N, dtype=float)

    # For each band n
    for n in range(N):
        En = eigvals[n]
        u_n = eigvecs[:, n]

        # Sum over m != n
        sum_val = 0.0
        for m in range(N):
            if m == n:
                continue
            Em = eigvals[m]
            u_m = eigvecs[:, m]

            denom = (En - Em)**2
            if abs(denom) < 1e-14:
                continue

            term_x = np.vdot(u_n, dHdkx @ u_m)

            term_y = np.vdot(u_m, dHdky @ u_n)
            val = np.imag(term_x * term_y) / denom
            sum_val += val

        omega[n] = -2 * sum_val

    return omega

# Define High symmetry points
b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2

Gamma = [0.0, 0.0]
X = [b1, 0.0]  
K = [0.0, b2*2/3]
M = [b1, b2] 
Y = [0, b2]

# Define k point path
points_path = [X, Gamma, K, M]
bz_path = build_bz_path(points_path, Nseg=150)

def find_high_symmetry_indices(bz_path, high_symmetry_points):
    indices = []
    for hs_point in high_symmetry_points:
        distances = [np.linalg.norm(np.array(k) - np.array(hs_point)) for k in bz_path]
        indices.append(np.argmin(distances)) 
    
    return indices

# Loop over k values
energies = []
berry_curvs = []
for k_point in bz_path:
    Hk = general.build_hamiltonian(k_point)
    Hk = (Hk + Hk.conj().T) / 2

    eigvals, eigvecs = np.linalg.eigh(Hk)
    energies.append(eigvals)


    dH_x = dHdkx(k_point)
    dH_y = dHdky(k_point)
    omega_n = berry_curvature_at_k(Hk, dH_x, dH_y) 
    berry_curvs.append(omega_n)

energies = np.array(energies)       
berry_curvs = np.array(berry_curvs) 

# Set colour scale
vmin, vmax = berry_curvs.min(), berry_curvs.max()
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # Adjust scaling

# Find high symmetry points
k_index = np.arange(len(bz_path))

high_symmetry_indices = find_high_symmetry_indices(bz_path, points_path)
high_symmetry_indices.append(k_index[-1])  # Include final point

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlim([k_index[0], k_index[-1]])
ax.set_ylim([-4.1, 2.1])

for d in high_symmetry_indices:
    ax.axvline(x=d, color='k', linestyle='solid', linewidth=0.3)

# Horizontal reference lines
for l in [-2, -1, 0]:
    ax.axhline(y=l, color='k', linestyle='--', linewidth=0.5)

# Plot band
for band in range(energies.shape[1]):
    ax.plot(k_index, energies[:, band], 'k-', lw=1)

# Plot berry curvature
scatter = ax.scatter(
    np.tile(k_index, (N, 1)).T,  
    energies,  
    c=berry_curvs, 
    cmap='RdBu_r',
    norm=divnorm,
    s=20,
    alpha=1
)

ax.set_xticks(high_symmetry_indices)
ax.set_xticklabels(['X', 'Γ', 'K', 'M', 'X']) 
ax.set_ylim([-1,1])
cbar = fig.colorbar(scatter, ax=ax, label="Berry curvature")

plt.tight_layout()

# Lattice vectors
a1 = np.array([np.sqrt(3), 0])
a2 = np.array([0, 1])

# Area of real space cell
A_cell = np.abs(np.cross(a1, a2))

# Area of the reciprocal cell
A_bz = (2*np.pi)**2 / A_cell 

# Obtain Chern numbers
def integrate_berry_curvature(Nx=100, Ny=100):

    berry_sum = np.zeros(N, dtype=float)

    dA = A_bz / (Nx * Ny)

    for i in range(Nx):
        for j in range(Ny):
            frac_i = i / float(Nx)
            frac_j = j / float(Ny)

            kx, ky = frac_i*b1 + frac_j*b2

            Hk = general.build_hamiltonian([kx, ky])
            Hk = 0.5 * (Hk + Hk.conjugate().T)

            dH_x = dHdkx([kx, ky])
            dH_y = dHdky([kx, ky])

            omega_n = berry_curvature_at_k(Hk, dH_x, dH_y)
            berry_sum += omega_n * dA

    chern_numbers = berry_sum / (2*np.pi)
    return chern_numbers

chern = integrate_berry_curvature(Nx=200, Ny=200) 
for band_index, C in enumerate(chern):
    print(f"   Band {band_index}: Chern = {C:.4f}")
print()
plt.show()