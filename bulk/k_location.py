"""
Plot the BZ and high symmetry points in k-space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def reciprocal(vector1, vector2):

    rot = np.array([[0,-1],[1,0]])

    reciprocal_vector1 = 2*np.pi * (rot @ vector2)/(vector1 @ (rot @ vector2))
    reciprocal_vector2 = 2*np.pi * (rot @ vector1)/(vector2 @ (rot @ vector1))

    return reciprocal_vector1, reciprocal_vector2

# Define the hexagonal BZ corners.
u1 = np.array([-1/2, np.sqrt(3)/2])
u2 = np.array([1,0])

b1, b2 = reciprocal(u1, u2)


hex_b1 = 2*np.pi/3 * np.array([1, np.sqrt(3)])
hex_b2 = 2*np.pi/3 * np.array([1, -np.sqrt(3)])
kx_hex = np.pi/np.sqrt(3)
ky_hex = np.pi/3
hex_corners = [
    [kx_hex,ky_hex],
    [kx_hex,-ky_hex],
    [0,-2*ky_hex],
    [-kx_hex,-ky_hex],
    [-kx_hex,ky_hex],
    [0,2*ky_hex],
    [kx_hex,ky_hex]
]

# Define symmetry points.
Gamma = (0.0, 0.0)
M = (kx_hex, 0.0)  
K = (kx_hex, ky_hex)   

sym = [Gamma, K, M, Gamma]
sym_label = ["Γ", "K", "M", "Γ"]

hex_x = [pt[0] for pt in hex_corners]
hex_y = [pt[1] for pt in hex_corners]

fig, ax = plt.subplots(figsize=(5,5))

# Plot hexagonal BZ
ax.plot(hex_x, hex_y, color='black', zorder=2, label="Hexagonal BZ", linestyle='solid')


# Create reciprocal lattice vector arrows (scaled down for display).
arrow1 = FancyArrowPatch(posA=(0, 0), posB=(b1[0]/3, b1[1]/3),
                         arrowstyle='->', mutation_scale=20, linestyle='dashed',
                         linewidth=1, color='b', zorder=4)
arrow2 = FancyArrowPatch(posA=(0, 0), posB=(b2[0]/3, b2[1]/3),
                         arrowstyle='->', mutation_scale=20, linestyle='dashed',
                         linewidth=1, color='b', zorder=4)

ax.add_patch(arrow1)
ax.add_patch(arrow2)


# Arrow labels
ax.text(b1[0]*0.33, b1[1]*0.33, r'$\mathbf{b}_1$', color='b', fontsize=12, zorder=4)
ax.text(b2[0]*0.33, b2[1]*0.33, r'$\mathbf{b}_2$', color='b', fontsize=12, zorder=4)

# Plot symmetry points and connect them with lines (foreground).
for i in range(len(sym)-1):
    ax.scatter(sym[i][0], sym[i][1], color='r', zorder=3)
    ax.plot([sym[i][0], sym[i+1][0]], [sym[i][1], sym[i+1][1]], 'r-', linewidth=2, zorder=3)

ax.text(sym[0][0] - 0.15, sym[0][1] + 0.1, sym_label[0], fontsize=12, zorder=5)
ax.text(sym[1][0] - 0.1, sym[1][1] + 0.1, sym_label[1], fontsize=12, zorder=5)
ax.text(sym[2][0] - 0.2, sym[2][1] + 0.1, sym_label[2], fontsize=12, zorder=5)

# Set plot limits.
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
#plt.xlabel('$k_x$')
#plt.ylabel('$k_y$')
plt.axis('off')
plt.tight_layout()
plt.savefig("hexagonal_BZ.png", dpi = 1000)
