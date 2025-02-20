import numpy as np

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


general = hamiltonian(0.01,a1,a2,a3,1)

def generate_k_path(points, n_points=50):
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
                k_dist.append(k_dist[-1] + np.sqrt(dx*dx + dy*dy))

    # Append the final end point
    k_vecs.append(points[-1])
    dx = points[-1][0] - k_vecs[-2][0]
    dy = points[-1][1] - k_vecs[-2][1]
    k_dist.append(k_dist[-1] + np.sqrt(dx*dx + dy*dy))

    return k_vecs, k_dist

# Define your high-symmetry points
Gamma = (0.0, 0.0)
K     = (4*np.pi/3, 0.0)
M     = (np.pi, np.pi/np.sqrt(3))

# We want to go Γ -> K -> M -> Γ
points_path = [Gamma, K, M, Gamma]

# Generate a path
k_vecs, k_dist = generate_k_path(points_path, n_points=100)

# Prepare arrays to store the 6 eigenvalues at each k
N_BANDS = N  # = 6 in your code
all_eigvals = np.zeros((len(k_vecs), N_BANDS))
delta_epsilon = 0.01  # Adjust this value based on the paper
onsite_energy = np.array([delta_epsilon, -delta_epsilon, delta_epsilon])  # A sublattices affected
delta_t = 0.01  # Adjust this value to control gap size

for i, (kx, ky) in enumerate(k_vecs):
    k = np.array([kx, ky])

    total_matrix = general.matrix_0()
    total_matrix += np.diag(onsite_energy)
    total_matrix += bloch(kx, ky, -a2, general.matrix_AB()) + bloch(kx, ky, a1, general.matrix_AC()) + bloch(kx, ky, -a3, general.matrix_BC())
    total_matrix += bloch(kx, ky, a2, general.matrix_BA()) + bloch(kx, ky, -a1, general.matrix_CA()) + bloch(kx, ky, a3, general.matrix_CB())
    total_matrix += delta_t * bloch(kx, ky, a2, general.matrix_BC()) - delta_t * bloch(kx, ky, -a2, general.matrix_CB())

    total_matrix = (total_matrix + total_matrix.conj().T) / 2

    eigvals, _ = np.linalg.eig(total_matrix)
    
    eigvals = np.sort(eigvals.real)
    all_eigvals[i, :] = eigvals

with open('kagome_band.txt', 'w') as f:
    for i, (kx, ky) in enumerate(k_vecs):
        f.write(f"{kx} {ky} {all_eigvals[i][0]} {all_eigvals[i][1]} {all_eigvals[i][2]}\n")