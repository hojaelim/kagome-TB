"""
Script for calculating DOS for the monolayer.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Necessary elements for the Hamiltonian.
N = 6

I6 = np.eye(6)

r1 = 1/2 * np.array([np.sqrt(3),1])
r2 = 1/2 * np.array([np.sqrt(3),-1])
r3 = np.array([0,1])
r4 = r2-r3
r5 = r1+r3
r6 = r1+r2

phi = np.pi/6 # TRS breaking



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

# Next Nearest Neighbour Hoppings    
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

# Self Hopping
    def matrix_0(self):
        return self.eps * I6

# A sites onsite difference
    def matrix_onsite_diff(self):
        matrix = np.zeros((6,6), dtype=complex)

        matrix[0,0] += 6
        matrix[1,1] += -6

        return self.deps/6 * matrix

# B and C staggered hopping
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

# Define k space
Nk = 100
kx_values = np.linspace(-2*np.pi, 2*np.pi, Nk)
ky_values = np.linspace(-2*np.pi, 2*np.pi, Nk)
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

# Compute DOS
all_energies = np.array(all_eigvals)  # Convert eigenvalues to 1D array

num_bins = 10000 # Set high resolution for better divergence capture
energy_min, energy_max = all_energies.min(), all_energies.max()
energy_bins = np.linspace(energy_min, energy_max, num_bins)

dos, bin_edges = np.histogram(all_energies, bins=energy_bins, density=True)

print(len(dos), len(bin_edges))

# Save data
with open("dos_kagome_6_TRS.txt", "w") as f:
    for i in range(len(dos)):
        f.write(str(dos[i]) + " " + str(bin_edges[i]) + "\n")

# Plot using bin_edges[:-1] for positions, np.diff(bin_edges) for widths
fig, ax = plt.subplots()
ax.bar(bin_edges[:-1], dos, width=np.diff(bin_edges), align="edge", color="blue", edgecolor="black", alpha=0.7)

ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Density of States (DOS)")
ax.set_title("Unsmoothed DOS")

plt.show()
