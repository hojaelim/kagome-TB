import numpy as np
import matplotlib.pyplot as plt

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


def generate_k_path(points, n_points=1000):

    k_vecs = []
    k_dist = [0.0]  # distance array

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
k_vecs, k_dist = generate_k_path(points_path, n_points=1000)

all_eigvals = np.zeros((len(k_vecs), N))

# Diagonalize at each k
for i, (kx, ky) in enumerate(k_vecs):
    k = np.array([kx, ky])
    # Build Hamiltonian
    total_matrix = (general.matrix_NN(k) 
                    + general.matrix_NNN(k) 
                    + general.matrix_0() 
                    + general.matrix_onsite_diff() 
                    + general.matrix_staggered(k))
    
    total_matrix = (total_matrix + total_matrix.conj().T) / 2


    eigvals, _ = np.linalg.eigh(total_matrix)
    eigvals = np.sort(eigvals.real)
    all_eigvals[i, :] = eigvals

with open('kagome_6_band.txt', 'w') as f:
    for i, (kx, ky) in enumerate(k_vecs):
        f.write(f"{kx} {ky} {all_eigvals[i][0]} {all_eigvals[i][1]} {all_eigvals[i][2]} {all_eigvals[i][3]} {all_eigvals[i][4]} {all_eigvals[i][5]}\n")

# Plot each band vs. k-dist
plt.figure(figsize=(5,5))
for band_idx in range(N):
    plt.plot(k_dist, all_eigvals[:, band_idx], color='blue')

plt.ylabel(r"$\epsilon$")


def find_kdist_endpoints(points):
    # Use the same path generation but n_points=1
    kvecs_end, kdist_end = generate_k_path(points, n_points=1)
    # kdist_end[0] = 0, kdist_end[1] = distance(Γ->K), etc.
    return kdist_end

kdist_end = find_kdist_endpoints(points_path)

for d in kdist_end:
    plt.axvline(x=d, color='k', linestyle='--', linewidth=0.5)

plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.ylim(-1,1)
plt.xlim(np.min(k_dist), np.max(k_dist))
labels = ["X", "G", "K", "M", "X"]

plt.xticks(kdist_end, labels)

plt.tight_layout()
plt.savefig("kagome_6_bandplot.png")
