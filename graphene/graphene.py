import numpy as np
import matplotlib.pyplot as plt

N = 2

def reciprocal(kx, ky, R, matrix):
    matrix = np.exp(1j * (kx * R[0] + ky * R[1])) * matrix
    return matrix

class hamiltonian:
    def __init__(self, E_A, E_B, t):
        self.E_A = E_A
        self.E_B = E_B
        self.t = t
        
    def matrix_0(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][0] = self.E_A
        matrix[1][1] = self.E_B
        return matrix
    
    def matrix_d1_plus(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][1] = self.t
        return matrix
    
    def matrix_d1_minus(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[1][0] = self.t
        return matrix

    def matrix_d2_plus(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][1] = self.t
        return matrix
    
    def matrix_d2_minus(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[1][0] = self.t
        return matrix

    def matrix_d3_plus(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[0][1] = self.t
        return matrix
    
    def matrix_d3_minus(self):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[1][0] = self.t
        return matrix
   
delta1 = [1, 0]
delta2 = [-0.5, -np.sqrt(3)/2]
delta3 = [-0.5, np.sqrt(3)/2]

general = hamiltonian(0, 0, 1)

matrix_0 = general.matrix_0()
matrix_d1_plus = general.matrix_d1_plus()
matrix_d1_minus = general.matrix_d1_minus()
matrix_d2_plus = general.matrix_d2_plus()
matrix_d2_minus = general.matrix_d2_minus()
matrix_d3_plus = general.matrix_d3_plus()
matrix_d3_minus = general.matrix_d3_minus()

k_values = np.linspace(-np.pi, np.pi, 200)
kx_values, ky_values = np.meshgrid(k_values, k_values)

eigenvalues_max = np.zeros(kx_values.shape)
eigenvalues_min = np.zeros(kx_values.shape)

for i in range(len(k_values)):
    for j in range(len(k_values)):
        kx = kx_values[i, j]
        ky = ky_values[i, j]

        reciprocal_matrix = -reciprocal(kx, ky, [0, 0], matrix_0)
        reciprocal_matrix -= reciprocal(kx, ky, delta1, matrix_d1_plus)
        reciprocal_matrix -= reciprocal(kx, ky, [-1*delta1[0], -1*delta1[1]], matrix_d1_minus)
        reciprocal_matrix -= reciprocal(kx, ky, delta2, matrix_d2_plus)
        reciprocal_matrix -= reciprocal(kx, ky, [-1*delta2[0], -1*delta2[1]], matrix_d2_minus)
        reciprocal_matrix -= reciprocal(kx, ky, delta3, matrix_d3_plus)
        reciprocal_matrix -= reciprocal(kx, ky, [-1*delta3[0], -1*delta3[1]], matrix_d3_minus)

        eigvals, _ = np.linalg.eig(reciprocal_matrix)
        eigenvalues_max[i, j] = np.max(eigvals.real)
        eigenvalues_min[i, j] = np.min(eigvals.real)

with open("bandplot.dat", "w") as f:
    for i in range(len(kx_values)):  # Divide by symmetry lines
        for j in range(len(ky_values)):
            f.write(f"{kx_values[i][j]} {ky_values[i][j]} {eigenvalues_min[i][j]} {eigenvalues_max[i][j]}\n")

print("kx range:", np.min(kx_values), np.max(kx_values))
print("ky range:", np.min(ky_values), np.max(ky_values))
print("Eigenvalues range:", np.min(eigenvalues_min), np.max(eigenvalues_max))

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(kx_values, ky_values, eigenvalues_max, cmap='viridis')

ax1.set_xlabel("kx")
ax1.set_ylabel("ky")
ax1.set_zlabel("Eigenvalue")

ax1.plot_surface(kx_values, ky_values, eigenvalues_min, cmap='plasma')
ax1.view_init(elev=20, azim=45)  # Adjust elevation and azimuth
plt.show()
