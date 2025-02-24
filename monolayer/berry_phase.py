import numpy as np
import matplotlib.pyplot as plt

def berry_phase_calculation(eigenvectors):
    products=[]  
    for i in range(len(eigenvectors)-1):
        products.append(np.dot(eigenvectors[i].conj().T, eigenvectors[i+1]))
    berry_phase = (-np.log(np.prod(products))).imag
    return berry_phase

def c_function(k_vec, r_vec):
    return np.cos(np.dot(k_vec, r_vec))

def s_function(k_vec, r_vec):
    return np.sin(np.dot(k_vec, r_vec))

def line_2d(kstart, kend, npoints):
    kstart = np.array(kstart, dtype=float)
    kend   = np.array(kend,   dtype=float)
    return [kstart + (kend - kstart)*i/(npoints-1) for i in range(npoints)]

def build_bz_path(corners, Nseg=20):
    path = line_2d(corners[0], corners[1], Nseg)

    for i in range(len(corners)-2):
        path += line_2d(corners[i+1], corners[i+2], Nseg)[1:]
    
    path += line_2d(corners[-1], corners[0], Nseg)[1:]

    return path


N = 6

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

Nk = 100

b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2

Gamma = [0.0, 0.0]
X = [b1, 0.0]  
K = [0.0, b2*2/3]
M = [b1, b2] 
Y = [0, b2]

#corners = [X, Gamma, K, M]
corners = [[-b1,-b2], [-b1,b2], [b1,b2], [b1,-b2]]

general = hamiltonian(eps, t, t2, deps, dt)

kpath = build_bz_path(corners, Nk)

all_eigvals = []
eigvecs_bands = [[] for _ in range(6)]
vectors_band0 = []

for k in kpath:
    total_matrix = (general.matrix_NN(k) 
                    + general.matrix_NNN(k) 
                    + general.matrix_0() 
                    + general.matrix_onsite_diff() 
                    + general.matrix_staggered(k))
    
    total_matrix = (total_matrix + total_matrix.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(total_matrix)

    sorted_indices = np.argsort(eigvals.real)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    band0_vec = eigvecs[:, 0]
    for i in range(6):
        eigvecs_bands[i].append(eigvecs[:,i])

eigvecs_bands = np.array(eigvecs_bands)
berry_phases = np.zeros(6)

for i in range(6):
    berry_phases[i] = np.abs(berry_phase_calculation(eigvecs_bands[i]))
    
print(berry_phases)