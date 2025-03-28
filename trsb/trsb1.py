"""
Python script for the TRSB-1 model calculation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
N = 12

# Create 2D path between two points
def line_2d(kstart, kend, npoints):

    kstart = np.array(kstart, dtype=float)
    kend   = np.array(kend,   dtype=float)
    return [kstart + (kend - kstart)*i/(npoints-1) for i in range(npoints)]

# Create specific BZ path
def build_bz_path(corners, Nseg):

    path = line_2d(corners[0], corners[1], Nseg)

    for i in range(len(corners)-2):
        path += line_2d(corners[i+1], corners[i+2], Nseg)[1:]
    
    path += line_2d(corners[-1], corners[0], Nseg)[1:]

    return path

# Required matrices
gm7 = np.array([[1,0,0], [0,-1,0], [0,0,0]], dtype=complex)
gm8 = (1/np.sqrt(3)) * np.array([[1,0,0], [0,1,0], [0,0,-2]], dtype=complex)
pz = np.array([[1,0], [0,-1]], dtype=complex)
I3 = np.eye(3)
I12 = np.eye(12)

# Hopping vectors
r1 = 1/2 * np.array([np.sqrt(3),1])
r2 = 1/2 * np.array([np.sqrt(3),-1])
r3 = np.array([0,1])
r4 = r2-r3
r5 = r1+r3
r6 = r1+r2

# Define Hamiltonian class
class hamiltonian:
    def __init__(self, eps, t, t2, deps, dt):
        self.eps = eps
        self.t = t
        self.t2 = t2
        self.deps = deps
        self.dt = dt

# Nearest Neighbours        
    def matrix_NN(self,k):

        matrix = np.zeros((N, N), dtype=complex)
        matrix += I12 * self.eps
        
#r1 hoppings
        matrix[0,4] += -self.t[2] * np.exp(-1j * np.dot(k,r1)) #A1 to B1
        matrix[4,0] += np.conj(matrix[0,4]) #B1 to A1

        matrix[0,6] += -np.conj(self.t[1]) * np.exp(1j * np.dot(k,r1)) # A1 to B3
        matrix[6,0] += np.conj(matrix[0,6]) # B3 to A1

        matrix[1,5] += -np.conj(self.t[1]) * np.exp(-1j * np.dot(k,r1)) # A2 to B2
        matrix[5,1] += np.conj(matrix[1,5]) # B2 to A2

        matrix[1,7] += -self.t[2] * np.exp(1j * np.dot(k,r1)) # A2 to B4
        matrix[7,1] += np.conj(matrix[1,7]) # B4 to A2

        matrix[2,4] += -np.conj(self.t[1]) * np.exp(1j * np.dot(k,r1)) # A3 to B1
        matrix[4,2] += np.conj(matrix[2,4]) # B1 to A3
        
        matrix[2,6] += -self.t[2] * np.exp(-1j * np.dot(k,r1)) # A3 to B3
        matrix[6,2] += np.conj(matrix[2,6]) # B3 to A3

        matrix[3,5] += -self.t[2] * np.exp(1j * np.dot(k,r1)) # A4 to B2
        matrix[5,3] += np.conj(matrix[3,5]) # B2 to A4
        
        matrix[3,7] += -np.conj(self.t[1]) * np.exp(-1j * np.dot(k,r1)) # A4 to B4
        matrix[7,3] += np.conj(matrix[3,7]) # B4 to A4

#r2 hoppings
        matrix[0,8] += -np.conj(self.t[0]) * np.exp(-1j * np.dot(k,r2)) #A1 to C1
        matrix[8,0] +=  np.conj(matrix[0,8]) #C1 to A1
        
        matrix[0,9] += -self.t[0] * np.exp(1j * np.dot(k,r2)) #A1 to C2
        matrix[9,0] += np.conj(matrix[0,9]) #C2 to A1

        matrix[1,8] += -self.t[0] * np.exp(1j * np.dot(k,r2)) #A2 to C1
        matrix[8,1] += np.conj(matrix[1,8]) #C1 to A2
        
        matrix[1,9] += -np.conj(self.t[0]) * np.exp(-1j * np.dot(k,r2)) #A2 to C2
        matrix[9,1] += np.conj(matrix[1,9]) #C2 to A2

        matrix[2,10] += -self.t[0] * np.exp(-1j * np.dot(k,r2)) #A3 to C3
        matrix[10,2] += np.conj(matrix[2,10]) #C3 to A3
        
        matrix[2,11] += -np.conj(self.t[0]) * np.exp(1j * np.dot(k,r2)) #A3 to C4
        matrix[11,2] += np.conj(matrix[2,11]) #C4 to A3

        matrix[3,10] += -np.conj(self.t[0]) * np.exp(1j * np.dot(k,r2)) #A4 to C3
        matrix[10,3] += np.conj(matrix[3,10]) #C3 to A4 
        
        matrix[3,11] += -self.t[0] * np.exp(-1j * np.dot(k,r2)) #A4 to C4
        matrix[11,3] += np.conj(matrix[3,11]) #C4 to A4

#r3 hoppings
        matrix[4,8] += -self.t[0] * np.exp(1j * np.dot(k,r3)) #B1 to C1
        matrix[8,4] += np.conj(matrix[4,8]) #C1 to B1
        
        matrix[4,11] += -np.conj(self.t[0]) * np.exp(-1j * np.dot(k,r3)) #B1 to C4
        matrix[11,4] += np.conj(matrix[4,11]) #C4 to B1

        matrix[5,9] += -np.conj(self.t[0]) * np.exp(1j * np.dot(k,r3)) #B2 to C2
        matrix[9,5] += np.conj(matrix[5,9]) #C2 to B2  
        
        matrix[5,10] += -self.t[0] * np.exp(-1j * np.dot(k,r3)) #B2 to C3
        matrix[10,5] += np.conj(matrix[5,10]) #C3 to B2

        matrix[6,9] += -self.t[0] * np.exp(-1j * np.dot(k,r3)) #B3 to C2
        matrix[9,6] += np.conj(matrix[6,9]) #C2 to B3   

        matrix[6,10] += -np.conj(self.t[0]) * np.exp(1j * np.dot(k,r3)) #B3 to C3
        matrix[10,6] += np.conj(matrix[6,10]) #C3 to B3

        matrix[7,8] += -np.conj(self.t[0]) * np.exp(-1j * np.dot(k,r3)) #B4 to C1
        matrix[8,7] += np.conj(matrix[7,8]) #C1 to B4
        
        matrix[7,11] += -self.t[0] * np.exp(1j * np.dot(k,r3)) #B4 to C4
        matrix[11,7] += np.conj(matrix[7,11]) #C4 to B4
        

        return matrix
    
# Next-Nearest Neighbours
    def matrix_NNN(self,k):

        matrix = np.zeros((N, N), dtype=complex)

#r4 hoppings
        matrix[0,5] += -self.t2 * np.exp(1j * np.dot(k,r4)) #A1 to B2
        matrix[5,0] += np.conj(matrix[0,5]) #B2 to A1
        matrix[0,7] += -self.t2 * np.exp(-1j * np.dot(k,r4)) #A1 to B4
        matrix[7,0] += np.conj(matrix[0,7]) #B4 to A1

        matrix[1,4] += -self.t2 * np.exp(1j * np.dot(k,r4)) #A2 to B1
        matrix[4,1] += np.conj(matrix[1,4]) #B1 to A2
        matrix[1,6] += -self.t2 * np.exp(-1j * np.dot(k,r4)) #A2 to B3
        matrix[6,1] += np.conj(matrix[1,6]) #B3 to A2

        matrix[2,5] += -self.t2 * np.exp(-1j * np.dot(k,r4)) #A3 to B2
        matrix[5,2] += np.conj(matrix[2,5]) #B2 to A3
        matrix[2,7] += -self.t2 * np.exp(1j * np.dot(k,r4)) #A3 to B4
        matrix[7,2] += np.conj(matrix[2,7]) #B4 to A3

        matrix[3,4] += -self.t2 * np.exp(-1j * np.dot(k,r4)) #A4 to B1
        matrix[4,3] += np.conj(matrix[3,4]) #B1 to A4
        matrix[3,6] += -self.t2 * np.exp(1j * np.dot(k,r4)) #A4 to B3
        matrix[6,3] += np.conj(matrix[3,6]) #B3 to A4

#r5 hoppings
        matrix[0,10] += -self.t2 * np.exp(1j * np.dot(k,r5)) #A1 to C3
        matrix[10,0] += np.conj(matrix[0,10]) #C3 to A1
        matrix[0,11] += -self.t2 * np.exp(-1j * np.dot(k,r5)) #A1 to C4
        matrix[11,0] += np.conj(matrix[0,11]) #C4 to A1

        matrix[1,10] += -self.t2 * np.exp(-1j * np.dot(k,r5)) #A2 to C3
        matrix[10,1] += np.conj(matrix[1,10]) #C3 to A2
        matrix[1,11] += -self.t2 * np.exp(1j * np.dot(k,r5)) #A2 to C4
        matrix[11,1] += np.conj(matrix[1,11]) #C4 to A2

        matrix[2,8] += -self.t2 * np.exp(1j * np.dot(k,r5)) #A3 to C1
        matrix[8,2] += np.conj(matrix[2,8]) #C1 to A3
        matrix[2,9] += -self.t2 * np.exp(-1j * np.dot(k,r5)) #A3 to C2
        matrix[9,2] += np.conj(matrix[2,9]) #C2 to A3

        matrix[3,8] += -self.t2 * np.exp(-1j * np.dot(k,r5)) #A4 to C1
        matrix[8,3] += np.conj(matrix[3,8])
        matrix[3,9] += -self.t2 * np.exp(1j * np.dot(k,r5)) #A4 to C2
        matrix[9,3] += np.conj(matrix[3,9])

#r6 hoppings
        matrix[4,9] += -self.t2 * np.exp(1j * np.dot(k,r6)) #B1 to C2
        matrix[9,4] += np.conj(matrix[4,9]) #C2 to B1
        matrix[4,10] += -self.t2 * np.exp(-1j * np.dot(k,r6)) #B1 to C3
        matrix[10,4] += np.conj(matrix[4,10])

        matrix[5,8] += -self.t2 * np.exp(1j * np.dot(k,r6)) #B2 to C1
        matrix[8,5] += np.conj(matrix[5,8])
        matrix[5,11] += -self.t2 * np.exp(-1j * np.dot(k,r6)) #B2 to C4
        matrix[11,5] += np.conj(matrix[5,11])

        matrix[6,8] += -self.t2 * np.exp(-1j * np.dot(k,r6)) #B3 to C1
        matrix[8,6] += np.conj(matrix[6,8])
        matrix[6,11] += -self.t2 * np.exp(1j * np.dot(k,r6)) #B3 to C4
        matrix[11,6] += np.conj(matrix[6,11])

        matrix[7,9] += -self.t2 * np.exp(-1j * np.dot(k,r6)) #B4 to C2
        matrix[9,7] += np.conj(matrix[7,9])
        matrix[7,10] += -self.t2 * np.exp(1j * np.dot(k,r6)) #B4 to C3
        matrix[10,7] += np.conj(matrix[7,10])

        return matrix

# Staggered hoppings between A sites
    def matrix_1(self):
        return self.deps/6 * np.kron((np.kron((2*I3+3*gm7+np.sqrt(3)*gm8), pz)), pz)
    
    
    def build_hamiltonian(self, k):
        total_matrix = self.matrix_NN(k) + self.matrix_NNN(k) + self.matrix_1()
        return total_matrix.conj()

def dHdkx(k, delta=1e-9):
    kx, ky = k
    H_plus  = general.build_hamiltonian([kx+delta, ky])
    H_minus = general.build_hamiltonian([kx-delta, ky])
    return (H_plus - H_minus)/(2*delta)

def dHdky(k, delta=1e-9):
    kx, ky = k
    H_plus  = general.build_hamiltonian([kx, ky+delta])
    H_minus = general.build_hamiltonian([kx, ky-delta])
    return (H_plus - H_minus)/(2*delta)

# Calculate berry curvature at a given point
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

            term_x = np.vdot(u_n, dHdkx @ u_m)

            term_y = np.vdot(u_m, dHdky @ u_n)
            val = np.imag(term_x * term_y) / denom
            sum_val += val

        omega[n] = -2 * sum_val

    return omega

# Define variables
eps = 0.035
t =  [(0.42 + 0.07j), (0.41 + 0.07j), (0.43 + 0.07j)]    
t2 = 0.03
deps = -0.033
dt = 0

# Build Hamiltonian
general = hamiltonian(eps, t, t2, deps, dt)

#Define High Symmetry Points
b2 = np.pi/6 
b1 = np.pi/(2*np.sqrt(3)) 

Gamma = [0.0, 0.0]
M1 = [b1, 0] 
M2 = [1/2* b1,3/2 * b2]
K1 = [b1, b2]
K2= [0,2*b2]

# Construct Path
points_path = [Gamma, M1, K1, M2, K2]
print("Path taken: Gamma -> M1 -> K1 -> M2 -> K2 -> Gamma")
print(Gamma, "->", M1, "->", K1, "->", M2, "->", K2, "->", Gamma)
bz_path = build_bz_path(points_path, Nseg=300)

# Execute Calculation for energies and berry curvature
energies = []
berry_curvs = []
for k_point in bz_path:
    Hk = general.build_hamiltonian(k_point)
    Hk = 0.5 * (Hk + Hk.conjugate().T)
    eigvals, eigvecs = np.linalg.eigh(Hk)
    energies.append(eigvals)

    dH_x = dHdkx(k_point)
    dH_y = dHdky(k_point)
    omega_n = berry_curvature_at_k(Hk, dH_x, dH_y)
    berry_curvs.append(omega_n)

energies = np.array(energies)   
berry_curvs = np.array(berry_curvs) 


# Compute high-symmetry indices
k_index = np.arange(len(bz_path))
def find_high_symmetry_indices(bz_path, high_symmetry_points):
    """Find the k_index values corresponding to high-symmetry points."""
    indices = []
    for hs_point in high_symmetry_points:
        # Find the closest index in bz_path to the given high-symmetry point
        distances = [np.linalg.norm(np.array(k) - np.array(hs_point)) for k in bz_path]
        indices.append(np.argmin(distances))  # Get the index of the closest point
    return indices

high_symmetry_indices = find_high_symmetry_indices(bz_path, points_path)
high_symmetry_indices.append(k_index[-1])
print(high_symmetry_indices[1:4])
berry_curvs_high_sym = berry_curvs[high_symmetry_indices, :]
energies_high_sym = energies[high_symmetry_indices, :]

# Calculate reciprocal lattice $b_i \cdot a_j = 2\pi \delta_{ij}$ 
def get_reciprocal_lattice(a1, a2):

    A = np.column_stack((a1, a2))  # shape = (2, 2)
    A_inv = np.linalg.inv(A)
    B   = (2*np.pi * A_inv).T

    b1 = B[:, 0]
    b2 = B[:, 1]
    return b1, b2

a1 = 2*np.array([np.sqrt(3), 1])
a2 = 2*np.array([np.sqrt(3), -1])
b1, b2 = get_reciprocal_lattice(a1, a2)

# Area of real space cell
A_cell = np.abs(np.cross(a1, a2))

# Area of the reciprocal cell
A_bz = (2*np.pi)**2 / A_cell 

# Integrate berry curvature along band to get the Chern number
def integrate_berry_curvature(Nx=200, Ny=200):

    berry_sum = np.zeros(N, dtype=float)

    dA = A_bz / (Nx * Ny)

    for i in range(Nx):
        for j in range(Ny):
            frac_i = i / float(Nx)
            frac_j = j / float(Ny)

            kvec = frac_i*b1 + frac_j*b2

            Hk = general.build_hamiltonian(kvec)
            Hk = 0.5 * (Hk + Hk.conjugate().T)

            dH_x = dHdkx(kvec)
            dH_y = dHdky(kvec)

            omega_n = berry_curvature_at_k(Hk, dH_x, dH_y)
            berry_sum += omega_n * dA

    chern_numbers = berry_sum / (2*np.pi)
    return chern_numbers

chern = integrate_berry_curvature() 
for band_index, C in enumerate(chern):
    print(f"   Band {band_index}: Chern = {C:.4f}")
print()
chern = np.rint(chern).astype(int)

# Save data to npz file
np.savez(
    "trsb1_data.npz",  
    energies=energies,
    berry_curvs=berry_curvs,
    chern=chern,
    k_index=k_index,
    high_symmetry_indices=high_symmetry_indices
)
print("Data saved to trsb1_data.npz")
