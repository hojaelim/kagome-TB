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

    def build_hamiltonian(self,k):
        return self.matrix_NN(k) + self.matrix_NNN(k) + self.matrix_0() + self.matrix_onsite_diff() + self.matrix_staggered(k)


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

def berry_curvature_at_k(Hk, dHdkx, dHdky):
    """
    Hk     : N×N Hamiltonian at k
    dHdkx  : ∂H/∂kx at the same k
    dHdky  : ∂H/∂ky at the same k
    Returns: array of shape (N,) with Omega_n(k) for n=0..N-1
    """
    eigvals, eigvecs = np.linalg.eigh(Hk)
    # Extract energies and vectors
    # eigvals.shape == (N,)
    # eigvecs.shape == (N, N); eigvecs[:, n] is the wavefunction of band n

    N = len(eigvals)
    omega = np.zeros(N, dtype=float)

    # For each band n
    for n in range(N):
        En = eigvals[n]
        u_n = eigvecs[:, n]  # shape (N,)

        # Sum over m != n
        sum_val = 0.0
        for m in range(N):
            if m == n:
                continue
            Em = eigvals[m]
            u_m = eigvecs[:, m]  # shape (N,)

            denom = (En - Em)**2
            if abs(denom) < 1e-14:
                # If degenerate, might skip or handle carefully
                continue

            # <u_n|dHdkx|u_m>
            print(dHdkx @ u_m, dHdky @ u_n)
            term_x = np.vdot(u_n, dHdkx @ u_m)  # complex
            # <u_m|dHdky|u_n>
            term_y = np.vdot(u_m, dHdky @ u_n)  # complex

            # product = term_x * term_y
            val = np.imag(term_x * term_y) / denom
            sum_val += val

        omega[n] = -2 * sum_val

    return omega

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

Nkx, Nky = 50, 50
kx_vals = np.linspace(-b1, b1, Nkx, endpoint=False)
ky_vals = np.linspace(-b2, b2, Nky, endpoint=False)

# We'll accumulate Omega_n(k) on a grid:
Berry_curv = np.zeros((Nkx, Nky, 6), dtype=float)

for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        # 1) Build H(k)
        Hk = general.build_hamiltonian([kx, ky])

        # 2) Build or get dHdkx, dHdky
        #    either analytically or numerically
        dH_x = dHdkx([kx, ky])
        dH_y = dHdky([kx, ky])

        # 3) Compute Omega_n(k)
        omega_n = berry_curvature_at_k(Hk, dH_x, dH_y)
        Berry_curv[i, j, :] = omega_n

# Now integrate over k-space for each band n
delta_kx = (kx_vals[-1] - kx_vals[0]) / Nkx  # or 2π/Nkx if your range is -π..π
delta_ky = (ky_vals[-1] - ky_vals[0]) / Nky
factor = delta_kx * delta_ky / (2 * np.pi)

Chern_numbers = np.zeros(6, dtype=float)
for n in range(6):
    # sum over the 2D grid
    Cn = factor * Berry_curv[:, :, n].sum()
    Chern_numbers[n] = Cn

print("Chern numbers = ", Chern_numbers)
