"""
For plotting the monolayer band plot, with setting for TRS.
"""

import numpy as np
import matplotlib.pyplot as plt

N = 6

# Required vectors and matrices
phi = np.pi/6 # For applying flux current; set to zero if TRS conserved.
I6 = np.eye(6)
r1 = 1/2 * np.array([np.sqrt(3),1])
r2 = 1/2 * np.array([np.sqrt(3),-1])
r3 = np.array([0,1])
r4 = r2-r3
r5 = r1+r3
r6 = r1+r2

# Define Hamiltonian
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

# Next Nearest-Neighbour hoppings
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

# Self hopping energies
    def matrix_0(self):
        return self.eps * I6

# Onsite difference for A sites
    def matrix_onsite_diff(self):
        matrix = np.zeros((6,6), dtype=complex)

        matrix[0,0] += 6
        matrix[1,1] += -6

        return self.deps/6 * matrix

# Staggered hoppings for B and C sites    
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

# Variables
eps = 0.01
t = 0.42
t2 = 0.03
deps = -0.033
dt = 0.01

general = hamiltonian(eps, t, t2, deps, dt)

# Generate the k point path along the given points
def generate_k_path(points, n_points=1000):

    k_vecs = []
    k_dist = [0.0]  

    for i in range(len(points) - 1):
        start = points[i]
        end   = points[i+1]
        # interpolate from start -> end
        for j in range(n_points):
            alpha = j / float(n_points)
            kx = (1 - alpha)*start[0] + alpha*end[0]
            ky = (1 - alpha)*start[1] + alpha*end[1]
            k_vecs.append((kx, ky))

            if (i > 0 or j > 0):
                # measure distance from previous point
                dx = kx - k_vecs[-2][0]
                dy = ky - k_vecs[-2][1]
                k_dist.append(k_dist[-1] + np.sqrt(dx**2 + dy**2))

    # Append the final end point
    k_vecs.append(points[-1])
    dx = points[-1][0] - k_vecs[-2][0]
    dy = points[-1][1] - k_vecs[-2][1]
    k_dist.append(k_dist[-1] + np.sqrt(dx**2 + dy**2))

    return k_vecs, k_dist

# Define High symmetry points
b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2

Gamma = [0.0, 0.0]
X = [b1, 0.0]  
K = [0.0, b2*2/3]
M = [b1, b2] 
Y = [0, b2]

points_path = [X, Gamma, K, M, X]
print("Path taken: X -> Γ -> K -> M -> X")
print(X, "->", Gamma, "->", K, "->", M, "->", X)

# Generate a path
k_vecs, k_dist = generate_k_path(points_path, n_points=10000)

all_eigvals = np.zeros((len(k_vecs), N))

# Diagonalize at each k
for i, (kx, ky) in enumerate(k_vecs):
    k = np.array([kx, ky])
    # Build Hamiltonian
    total_matrix = (general.matrix_NN(k) + general.matrix_NNN(k) + general.matrix_0() + general.matrix_onsite_diff() + general.matrix_staggered(k))
    total_matrix = (total_matrix + total_matrix.conj().T) / 2

    eigvals, _ = np.linalg.eigh(total_matrix)
    eigvals = np.sort(eigvals.real)
    all_eigvals[i, :] = eigvals

# Save data for band
with open('kagome_6_band_TRS.txt', 'w') as f:
    for i, (kx, ky) in enumerate(k_vecs):
        f.write(f"{kx} {ky} {all_eigvals[i][0]} {all_eigvals[i][1]} {all_eigvals[i][2]} {all_eigvals[i][3]} {all_eigvals[i][4]} {all_eigvals[i][5]}\n")

        

# Plot each band vs. k-dist
plt.figure(figsize=(5,5))
for band_idx in range(N):
    plt.plot(k_dist, all_eigvals[:, band_idx], color='blue')

plt.ylabel(r"$\epsilon$")

# Find distance between high symmetry points for plotting
def find_kdist_endpoints(points):
    _, kdist_end = generate_k_path(points, n_points=1)
    return kdist_end

kdist_end = find_kdist_endpoints(points_path)

for d in kdist_end:
    plt.axvline(x=d, color='k', linestyle='--', linewidth=0.5)

plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

#plt.ylim(-1,1)
plt.xlim(np.min(k_dist), np.max(k_dist))
labels = ["X", "G", "K", "M", "X"]

plt.xticks(kdist_end, labels)

plt.tight_layout()
plt.show()
#plt.savefig("kagome_6_bandplot.png")
