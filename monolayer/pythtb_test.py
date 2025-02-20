
from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat=[[np.sqrt(3),0.0],[0.0,1.0]]
# define coordinates of orbitals
orb=[[np.sqrt(3)/2, 1.0],[0.0, 1/2],[(3*np.sqrt(3))/4,(1-1/np.sqrt(3))/2],[(np.sqrt(3))/4,(1+1/np.sqrt(3))/2], [(3*np.sqrt(3))/4,(1+1/np.sqrt(3))/2], [(np.sqrt(3))/4,(1-1/np.sqrt(3))/2]]

# make two dimensional tight-binding graphene model
my_model=tb_model(2,2,lat,orb, nspin=1)

# set model parameters
delta = 0.0
t = -0.42
t2 = -0.03
de = -0.033
dt = 0.01

onsite_list = [0.01, 0.01 + 2*de, 0.01, 0.01, 0.01, 0.01]
my_model.set_onsite(onsite_list)

# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 3, [ 0, 0])
my_model.set_hop(t, 0, 4, [ 0, 0])
my_model.set_hop(t, 1, 3, [ 0, 0])
my_model.set_hop(t, 1, 5, [ 0, 0])
my_model.set_hop(t+dt, 3, 5, [ 0, 0], allow_conjugate_pair=True)
my_model.set_hop(t-dt, 5, 3, [ 0, 0], allow_conjugate_pair=True)
my_model.set_hop(t+dt, 4, 2, [ 0, 0], allow_conjugate_pair=True)
my_model.set_hop(t-dt, 2, 4, [ 0, 0], allow_conjugate_pair=True)

my_model.set_hop(t, 2, 1, [ 1, 0])
my_model.set_hop(t, 4, 1, [ 1, 0])

my_model.set_hop(t, 2, 0, [ 0, -1])
my_model.set_hop(t, 5, 0, [ 0, -1])

#set next-nearest neighbor hoppings (0,0)
my_model.set_hop(t2, 0, 1, [ 0, 0])
my_model.set_hop(t2, 0, 2, [ 0, 0])
my_model.set_hop(t2, 0, 5, [ 0, 0])
my_model.set_hop(t2+dt, 2, 5, [ 0, 0], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 5, 2, [ 0, 0], allow_conjugate_pair=True)
my_model.set_hop(t2+dt, 3, 4, [ 0, 0], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 4, 3, [ 0, 0], allow_conjugate_pair=True)

#(1,0)
my_model.set_hop(t2, 0, 1, [ 1, 0])
my_model.set_hop(t2+dt, 2, 3, [ 1, 0], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 3, 2, [ -1, 0], allow_conjugate_pair=True)
my_model.set_hop(t2, 2, 5, [ 1, 0])
my_model.set_hop(t2+dt, 4, 5, [ 1, 0], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 5, 4, [ -1, 0], allow_conjugate_pair=True)
my_model.set_hop(t2+dt, 4, 3, [ 1, 0], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 3, 4, [ -1, 0], allow_conjugate_pair=True)

#(0,1)
my_model.set_hop(t2, 0, 1, [ 0, 1])
my_model.set_hop(t2, 0, 3, [ 0, 1])
my_model.set_hop(t2, 0, 4, [ 0, 1])
my_model.set_hop(t2+dt, 3, 2, [ 0, 1], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 2, 3, [ 0, -1], allow_conjugate_pair=True)
my_model.set_hop(t2+dt, 3, 5, [ 0, 1], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 5, 3, [ 0, -1], allow_conjugate_pair=True)
my_model.set_hop(t2+dt, 4, 2, [ 0, 1], allow_conjugate_pair=True)
my_model.set_hop(t2-dt, 2, 4, [ 0, -1], allow_conjugate_pair=True)
my_model.set_hop(t2, 4, 5, [ 0, 1])

#(-1,-1)
my_model.set_hop(t2, 1, 0, [ -1, -1])


"""
cut_x = my_model.cut_piece(4, 0, glue_edgs=False)
cut_2D = cut_x.cut_piece(4, 1, glue_edgs=False)
cut_2D.display()

(fig, ax) = cut_2D.visualize(0,1)
plt.show()
"""
# define the path in fractional coords (kx, ky)
path_kpts = [
  [0.0, 1.0],  # X
  [0.0, 0.0],  # Gamma
  [0.0, 2/3],  # K 
  [1.0, 1.0],  # M
  [0.0, 1.0],  # back to X
]

# choose how many points per line segment
n_k = 100  

# get the list of k-points along this path
(k_vec, k_dist, k_node) = my_model.k_path(path_kpts, n_k)

evals_array = my_model.solve_all(k_vec)
# evals_array has shape (n_orbitals, Ntotal)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

norb = my_model.get_num_orbitals()  # should be 6
for iband in range(norb):
    plt.plot(k_dist, evals_array[iband,:], 'b-')

# Draw vertical lines at the high-symmetry points
for xline in k_node:
    plt.axvline(xline, color='k', linewidth=0.5)

# Label them
labels = ["X", "G", "K", "M", "X"]
plt.xticks(k_node, labels)

plt.xlabel("Wave vector k-path")
plt.ylabel("Energy (eV)")
plt.title("Band Structure for 1x√3 Rectangular Kagome")
plt.tight_layout()
plt.show()

