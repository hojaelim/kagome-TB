import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


kx = np.pi/(2*np.sqrt(3))
ky = np.pi/2

# Define the rectangular BZ corners.
corners = [
    ( kx,  ky),
    ( kx,  -ky),
    ( -kx, -ky),
    ( -kx,  ky),
    ( kx,  ky),
]

x = [pt[0] for pt in corners]
y = [pt[1] for pt in corners]

# Define symmetry points.
Gamma = (0.0, 0.0)
X = (kx, 0.0)
K = (0.0, np.pi/(3))  
M = (kx, ky)   

sym = [X, Gamma, K, M, X]
sym_label = ["X", "Γ", "K", "M", "X"]

a1= np.array([np.sqrt(3), 0])
a2 = np.array([0,1])

rotation = np.array([[0, -1], [1, 0]])

b1 = 2*np.pi * np.dot(rotation, a2)/(np.dot(a1, np.dot(rotation, a2)))
b2 = 2*np.pi * np.dot(rotation, a1)/(np.dot(a2, np.dot(rotation, a1)))

# Define the hexagonal BZ corners.
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


hex_x = [pt[0] for pt in hex_corners]
hex_y = [pt[1] for pt in hex_corners]

fig, ax = plt.subplots(figsize=(5,5))

# Plot hexagonal BZ
ax.plot(hex_x, hex_y, color='blue', zorder=2, label="Hexagonal BZ", linestyle='dashed')
# Plot rectangular BZ (background)
ax.plot(x, y, color='black', zorder=1, label="Rectangular BZ")


# Create reciprocal lattice vector arrows (scaled down for display).
arrow1 = FancyArrowPatch(posA=(0, 0), posB=(hex_b1[0]/1.5, hex_b1[1]/1.5),
                         arrowstyle='->', mutation_scale=20, linestyle='solid',
                         linewidth=1, color='b', zorder=3)
arrow2 = FancyArrowPatch(posA=(0, 0), posB=(hex_b2[0]/1.5, hex_b2[1]/1.5),
                         arrowstyle='->', mutation_scale=20, linestyle='solid',
                         linewidth=1, color='b', zorder=3)
arrow3 = FancyArrowPatch(posA=(0, 0), posB=(b1[0]/1.5, b1[1]/2.25),
                         arrowstyle='->', mutation_scale=20, linestyle='solid',
                         linewidth=1, color='k', zorder=3)
arrow4 = FancyArrowPatch(posA=(0, 0), posB=(b2[0]/1.5, b2[1]/2.25),
                         arrowstyle='->', mutation_scale=20, linestyle='solid',
                         linewidth=1, color='k', zorder=3)
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.add_patch(arrow3)
ax.add_patch(arrow4)


# Arrow labels
ax.text(hex_b1[0]*0.7, hex_b1[1]*0.7, r'$\mathbf{b}_1$', color='b', fontsize=12, zorder=4)
ax.text(hex_b2[0]*0.7, hex_b2[1]*0.7, r'$\mathbf{b}_2$', color='b', fontsize=12, zorder=4)
ax.text(b1[0]*0.6, b1[1] +0.25, r'$\mathbf{b}_1$', color='k', fontsize=12, zorder=4)
ax.text(b2[0]+0.25, b2[1]*0.425, r'$\mathbf{b}_2$', color='k', fontsize=12, zorder=4)

# Plot symmetry points and connect them with lines (foreground).
for i in range(len(sym)-1):
    ax.scatter(sym[i][0], sym[i][1], color='r', zorder=5)
    ax.text(sym[i][0] - 0.3, sym[i][1] + 0.2, sym_label[i], fontsize=12, zorder=5)
    ax.plot([sym[i][0], sym[i+1][0]], [sym[i][1], sym[i+1][1]], 'r-', linewidth=2, zorder=5)

ax.axhline(y=np.pi/(3), linestyle = 'dotted', color ='k')

# Set plot limits.
plt.xlim(-3., 3.)
plt.ylim(-3., 3.)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.savefig("kagome_structure.png", dpi=600)
plt.show()
