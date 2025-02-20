import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
N = 3

def bloch(kx, ky, R, matrix):
    matrix = (1 + np.exp(-1j * (kx * R[0] + ky * R[1]))) * matrix
    return matrix

class hamiltonian:
    def __init__(self, E, a1, a2, a3, t):
        self.E = E
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.t = -t
        
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

Nk = 1000

general = hamiltonian(0,a1,a2,a3,1)

kx_values = np.linspace(-4*np.pi, 4*np.pi, Nk)
ky_values = np.linspace(-4*np.pi, 4*np.pi, Nk)
kx, ky = np.meshgrid(kx_values, ky_values)

all_eigvals = []

# Loop over k-points
for i in range(Nk):
    for j in range(Nk):
        H = general.matrix_0()
        H += bloch(kx[i, j], ky[i, j], -a2, general.matrix_AB()) + bloch(kx[i, j], ky[i, j], a1, general.matrix_AC()) 
        H += bloch(kx[i, j], ky[i, j], -a3, general.matrix_BC()) + bloch(kx[i, j], ky[i, j], a2, general.matrix_BA()) 
        H += bloch(kx[i, j], ky[i, j], -a1, general.matrix_CA()) + bloch(kx[i, j], ky[i, j], a3, general.matrix_CB())

        # Compute eigenvalues (band structure)
        eigenvalues = np.linalg.eigvalsh(H)

        all_eigvals.extend(eigenvalues)


# --- Compute DOS ---
all_energies = np.array(all_eigvals)  # Convert eigenvalues to 1D array

# Set high resolution for better divergence capture
num_bins = 100000
energy_min, energy_max = all_energies.min() - 0.1, all_energies.max() + 0.1
energy_bins = np.linspace(energy_min, energy_max, num_bins)

dos, bin_edges = np.histogram(all_energies, bins=energy_bins, density=True)

with open("dos.txt", "w") as f:
    for i in range(len(dos)):
        f.write(str(dos[i]) + " " + str(bin_edges[i]) + "\n")
