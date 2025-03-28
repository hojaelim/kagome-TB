"""
Script for calculating berry curvature and Chern number for bands along the high-symmetry path
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

N = 3

def bloch(kx, ky, R, matrix):
    matrix = (1 + np.exp(-1j * (kx * R[0] + ky * R[1]))) * matrix
    return matrix

def line_2d(kstart, kend, npoints):

    kstart = np.array(kstart, dtype=float)
    kend   = np.array(kend,   dtype=float)
    return [kstart + (kend - kstart)*i/(npoints-1) for i in range(npoints)]

def build_bz_path(corners, Nseg):

    path = line_2d(corners[0], corners[1], Nseg)

    for i in range(len(corners)-2):
        path += line_2d(corners[i+1], corners[i+2], Nseg)[1:]
    
    path += line_2d(corners[-1], corners[0], Nseg)[1:]

    return path

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
    
    def matrix_AB_t2(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][1] = self.t2
        return matrix

    def matrix_BC_t2(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[1][2] = self.t2
        return matrix

    def matrix_CA_t2(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[2][0] = self.t2
        return matrix
    
    def matrix_BA_t2(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[1][0] = self.t2
        return matrix

    def matrix_CB_t2(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[2][1] = self.t2
        return matrix

    def matrix_AC_t2(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][2] = self.t2
        return matrix
    
    def build_hamiltonian(self,k,onsite_energy,phi):
        kx = k[0]
        ky = k[1]
        Hk = general.matrix_0()
        Hk += np.diag(onsite_energy) # Inversion symmetry breaking
        Hk += bloch(kx, ky, -a2, general.matrix_AB())*np.exp(-1j*phi/3) + bloch(kx, ky, a1, general.matrix_AC())*np.exp(1j*phi/3) + bloch(kx, ky, -a3, general.matrix_BC())*np.exp(-1j*phi/3)
        Hk += bloch(kx, ky, a2, general.matrix_BA())*np.exp(1j*phi/3) + bloch(kx, ky, -a1, general.matrix_CA())*np.exp(-1j*phi/3) + bloch(kx, ky, a3, general.matrix_CB())*np.exp(1j*phi/3)
        return Hk


a1 = np.array([1, 0])
a2 = np.array([-1/2, np.sqrt(3)/2])
a3 = np.array([-1/2, -np.sqrt(3)/2])

general = hamiltonian(0,a1,a2,a3,1)

def dHdkx(k, onsite_energy, phi, delta=1e-5):
    kx, ky = k
    H_plus  = general.build_hamiltonian([kx+delta, ky], onsite_energy, phi)
    H_minus = general.build_hamiltonian([kx-delta, ky], onsite_energy, phi)
    return (H_plus - H_minus)/(2*delta)

def dHdky(k, onsite_energy, phi, delta=1e-5):
    kx, ky = k
    H_plus  = general.build_hamiltonian([kx, ky+delta], onsite_energy, phi)
    H_minus = general.build_hamiltonian([kx, ky-delta], onsite_energy, phi)
    return (H_plus - H_minus)/(2*delta)

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

            # <u_n|dHdkx|u_m>
            term_x = np.vdot(u_n, dHdkx @ u_m) 
            # <u_m|dHdky|u_n>
            term_y = np.vdot(u_m, dHdky @ u_n) 


            val = np.imag(term_x * term_y) / denom
            sum_val += val

        omega[n] = -2 * sum_val

    return omega

# Define high-symmetry points
Gamma = (0.0, 0.0)
K     = (4*np.pi/3, 0.0)
M     = (np.pi, np.pi/np.sqrt(3))

points_path = [Gamma, K, M]
bz_path = build_bz_path(points_path, Nseg=150)

def find_high_symmetry_indices(bz_path, high_symmetry_points):
    indices = []
    for hs_point in high_symmetry_points:
        distances = [np.linalg.norm(np.array(k) - np.array(hs_point)) for k in bz_path]
        indices.append(np.argmin(distances))
    return indices



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 6), sharey=True)

# Compute to find colour scale
all_berry_curvs = []  

for onsite_energy, phi in zip([[0, 0, 0], [0.1, 0, -0.1], [0, 0, 0]], 
                               [0, 0, np.pi/6]):
    berry_curvs = []
    for k_point in bz_path:
        Hk = general.build_hamiltonian(k_point, onsite_energy=onsite_energy, phi=phi)
        Hk = (Hk + Hk.conj().T) / 2

        dH_x = dHdkx(k_point, onsite_energy=onsite_energy, phi=phi)
        dH_y = dHdky(k_point, onsite_energy=onsite_energy, phi=phi)
        omega_n = berry_curvature_at_k(Hk, dH_x, dH_y)
        berry_curvs.append(omega_n)

    all_berry_curvs.append(np.array(berry_curvs)) 

all_berry_curvs = np.concatenate(all_berry_curvs) 
vmin, vmax = all_berry_curvs.min(), all_berry_curvs.max()
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


scatter = ax1.scatter([], [], c=[], cmap='RdBu_r', norm=divnorm)
for ax, onsite_energy, phi in zip([ax1, ax2, ax3], 
                                   [[0, 0, 0], [0.2, 0, -0.2], [0, 0, 0]], 
                                   [0, 0, np.pi/6]):

    energies = []
    berry_curvs = []

    for k_point in bz_path:
        Hk = general.build_hamiltonian(k_point, onsite_energy=onsite_energy, phi=phi)
        Hk = (Hk + Hk.conj().T) / 2

        eigvals, eigvecs = np.linalg.eigh(Hk)
        energies.append(eigvals)

        dH_x = dHdkx(k_point, onsite_energy=onsite_energy, phi=phi)
        dH_y = dHdky(k_point, onsite_energy=onsite_energy, phi=phi)
        omega_n = berry_curvature_at_k(Hk, dH_x, dH_y)
        berry_curvs.append(omega_n)

    energies = np.array(energies)       
    berry_curvs = np.array(berry_curvs) 
    vmin, vmax = berry_curvs.min(), berry_curvs.max()
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # Adjust scaling

    k_index = np.arange(len(bz_path))

    high_symmetry_indices = find_high_symmetry_indices(bz_path, points_path)
    high_symmetry_indices.append(k_index[-1])

    ax.set_xlim([k_index[0], k_index[-1]])
    ax.set_ylim([-4.1,2.1])

    for d in high_symmetry_indices:
        ax.axvline(x=d, color='k', linestyle='solid', linewidth=0.3)

    for l in [-2,-1,0]:
        ax.axhline(y=l, color='k', linestyle='--', linewidth=0.5)

    for band in range(energies.shape[1]):
        ax.plot(k_index, energies[:, band], 'k-', lw=1)

    for i in range(N):
        ax.scatter(
            k_index,
            energies[:, i],
            c=berry_curvs[:, i],
            cmap='RdBu_r',
            norm=divnorm,
            s=20,
            alpha=1
        )

    ax.set_xticks(high_symmetry_indices, ['Γ', 'K', 'M', 'Γ'])

K_index = np.argmin([np.linalg.norm(np.array(k) - np.array(K)) for k in bz_path])
ax1.scatter(
    k_index[K_index],
    -1,  
    color='white',  
    s=20, 
    zorder=1  
)

ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.tick_params(axis='y', length=0)

ax3.set_yticks([])
ax3.set_yticklabels([])
ax3.tick_params(axis='y', length=0)

yticks = [-4, -3, -2, -1, 0, 1, 2]
ax1.set_yticks(yticks)
ax1.set_yticklabels([str(y) for y in yticks]) 

import matplotlib.ticker as ticker

cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6]) 
cbar=fig.colorbar(scatter, cax=cbar_ax)
cbar.ax.set_title(r'$\Omega_n(k)$', fontsize=14, pad=10)
cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))


import numpy as np


def get_reciprocal_lattice(a1, a2):

    # Form the 2×2 matrix whose columns are a1 and a2
    A = np.column_stack((a1, a2))  # shape = (2, 2)
    # Then B^T = 2π A^{-1}. So B = (2π (A^-1)).T
    A_inv = np.linalg.inv(A)
    B_T   = 2*np.pi * A_inv
    B     = B_T.T
    # b1, b2 are the columns of B
    b1 = B[:, 0]
    b2 = B[:, 1]
    return b1, b2

b1, b2 = get_reciprocal_lattice(a1, a2)
# Area of real-space cell
A_cell = np.abs(np.cross(a1, a2))  # = sqrt(3)/2
# Area of the reciprocal cell
A_bz = (2*np.pi)**2 / A_cell    

def integrate_berry_curvature(onsite_energy, phi, Nx=100, Ny=100):

    berry_sum = np.zeros(N, dtype=float) 

    dA = A_bz / (Nx * Ny)  # area per mesh point

    for i in range(Nx):
        for j in range(Ny):
            frac_i = i / float(Nx)
            frac_j = j / float(Ny)
            kx, ky = frac_i*b1 + frac_j*b2

            Hk = general.build_hamiltonian([kx, ky],
                                           onsite_energy=onsite_energy, 
                                           phi=phi)
            Hk = 0.5 * (Hk + Hk.conjugate().T)

            dH_x = dHdkx([kx, ky], onsite_energy=onsite_energy, phi=phi)
            dH_y = dHdky([kx, ky], onsite_energy=onsite_energy, phi=phi)

            # Berry curvature for each band
            omega_n = berry_curvature_at_k(Hk, dH_x, dH_y)
            berry_sum += omega_n * dA


    chern_numbers = berry_sum / (2*np.pi)
    return chern_numbers



parameter_sets = [
    {"onsite_energy": [0,   0,   0   ], "phi": 0,        "label":"Case 1"},
    {"onsite_energy": [0.2, 0,  -0.2 ], "phi": 0,        "label":"Case 2"},
    {"onsite_energy": [0,   0,   0   ], "phi": np.pi/6,  "label":"Case 3"},
]

chern_numbers = np.zeros((3,3))
for i, params in enumerate(parameter_sets):
    chern = integrate_berry_curvature(params["onsite_energy"], params["phi"],
                                      Nx=100, Ny=100)  
    print(f"{params['label']}: onsite={params['onsite_energy']}, phi={params['phi']}")
    for band_index, C in enumerate(chern):
        chern_numbers[i,band_index] = C
        print(f"   Band {band_index}: Chern = {C:.4f}")
    print()


avg_energy_0 = np.mean(energies[:, 0])  
avg_energy_1 = np.mean(energies[:, 1])
avg_energy_2 = np.mean(energies[:, 2])
ax3.text(
    len(bz_path)/2, avg_energy_2*0.8,
    f"C = {int(round(chern_numbers[2][2]))}", 
    fontsize=10,
    color='black',
    verticalalignment='center',
    horizontalalignment='left',
)

ax3.text(
    len(bz_path)/2, avg_energy_1*0.9,
    f"C = {int(round(chern_numbers[2][1]))}",  
    fontsize=10,
    color='black',
    verticalalignment='center',
    horizontalalignment='left',
)

ax3.text(
    len(bz_path)/2 + 0.2, avg_energy_0,
    f"C = {int(round(chern_numbers[2][0]))}", 
    fontsize=10,
    color='black',
    verticalalignment='center',
    horizontalalignment='left',
)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("band_curvature.png", dpi=600)
plt.show()
