"""
Bandplot for bulk
"""

import numpy as np
import matplotlib.pyplot as plt

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

general = hamiltonian(0,a1,a2,a3,1)

def generate_k_path(points, n_points=50):
    k_vecs = []
    k_dist = [0.0]  

    for i in range(len(points) - 1):
        start = points[i]
        end   = points[i+1]

        for j in range(n_points):
            alpha = j / float(n_points)
            kx = (1 - alpha)*start[0] + alpha*end[0]
            ky = (1 - alpha)*start[1] + alpha*end[1]
            k_vecs.append((kx, ky))

            if (i > 0 or j > 0):

                dx = kx - k_vecs[-2][0]
                dy = ky - k_vecs[-2][1]
                k_dist.append(k_dist[-1] + np.sqrt(dx*dx + dy*dy))

    k_vecs.append(points[-1])
    dx = points[-1][0] - k_vecs[-2][0]
    dy = points[-1][1] - k_vecs[-2][1]
    k_dist.append(k_dist[-1] + np.sqrt(dx*dx + dy*dy))

    return k_vecs, k_dist


Gamma = (0.0, 0.0)
K     = (4*np.pi/3, 0.0)
M     = (np.pi, np.pi/np.sqrt(3))

points_path = [Gamma, K, M, Gamma]

k_vecs, k_dist = generate_k_path(points_path, n_points=100)

all_eigvals = np.zeros((len(k_vecs), N))
onsite_energy = np.array([0.2, 0, -0.2])
phi =  0

for i, (kx, ky) in enumerate(k_vecs):
    k = np.array([kx, ky])

    total_matrix = general.matrix_0()
    total_matrix += np.diag(onsite_energy) # Inversion symmetry breaking
    total_matrix += bloch(kx, ky, -a2, general.matrix_AB())*np.exp(-1j*phi/3) + bloch(kx, ky, a1, general.matrix_AC())*np.exp(1j*phi/3) + bloch(kx, ky, -a3, general.matrix_BC())*np.exp(-1j*phi/3)
    total_matrix += bloch(kx, ky, a2, general.matrix_BA())*np.exp(1j*phi/3) + bloch(kx, ky, -a1, general.matrix_CA())*np.exp(-1j*phi/3) + bloch(kx, ky, a3, general.matrix_CB())*np.exp(1j*phi/3)
   

    total_matrix = (total_matrix + total_matrix.conj().T) / 2

    eigvals, _ = np.linalg.eig(total_matrix)
    
    eigvals = np.sort(eigvals.real)
    all_eigvals[i, :] = eigvals


with open('kagome_band_is.txt', 'w') as f:
    for i, (kx, ky) in enumerate(k_vecs):
        f.write(f"{kx} {ky} {all_eigvals[i][0]} {all_eigvals[i][1]} {all_eigvals[i][2]}\n")


def find_kdist_endpoints(points):
    kvecs_end, kdist_end = generate_k_path(points, n_points=1)
    return kdist_end

kdist_end = find_kdist_endpoints(points_path)

fig, ax = plt.subplots(figsize=(5, 5))
for band_idx in range(3):
    ax.plot(k_dist, all_eigvals[:, band_idx], color='blue')

for d in kdist_end:
    ax.axvline(x=d, color='k', linestyle='--', linewidth=0.5)

ax.set_xticks([]) 
ax.set_xlabel("") 


labels = ["Γ", "K", "M", "Γ"]
for d, label in zip(kdist_end, labels):
    ax.text(d, ax.get_ylim()[0] - 0.1, label, ha='center', va='top', fontsize=12)


y_min, y_max = -4.1, 2.1  
ax.set_xlim(k_dist[0], k_dist[-1])
#ax.set_ylim(y_min, y_max)
plt.savefig("band_is.png")
plt.show()