"""
For plotting both path and mesh on same figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter, zoom

# Load 1D susceptibility data
data = np.loadtxt('chi_path_results_test.txt')
q_dist = data[:, 0]
qx = data[:, 1]
qy = data[:, 2]
mu_0 = data[:, 3]

# Function to flip values near Gamma
def flip_gamma(chi, q_dist, threshold):
    gamma_indices = np.where((data[:,1] + data[:,2]) < threshold)[0]
    if gamma_indices.size > 0:
        outside_indices = np.where((q_dist >= threshold) & (q_dist < 2*threshold))[0]
        if outside_indices.size > 0:
            chi_ref = np.mean(chi[outside_indices])
            chi_flipped = chi.copy()
            chi_flipped[gamma_indices] = 2 * chi_ref - chi_flipped[gamma_indices]
            return chi_flipped
    return chi

mu_0 = flip_gamma(mu_0, q_dist, threshold=0.045)
mu_0 = gaussian_filter1d(mu_0, sigma=10)

# High-symmetry path for 1D plot
b1, b2 = 4*np.pi/3, 2*np.pi/np.sqrt(3)
Gamma = (0.0, 0.0)
M     = (np.pi, np.pi/np.sqrt(3))
K     = (b1, 0)

points_path = np.array([Gamma, K, M, Gamma])
labels = [r'$\Gamma$', 'K', 'M', r'$\Gamma$']

def generate_k_path(points, n_points=100):
    q_vecs = []
    for i in range(len(points)-1):
        start, end = points[i], points[i+1]
        for alpha in np.linspace(0, 1, n_points):
            q_vecs.append((1 - alpha)*start + alpha*end)
    return np.array(q_vecs)

n_points = 300
q_vecs = generate_k_path(points_path, n_points)

# Compute distance along the q-path:
q_dist = [0.0]
for i in range(1, len(q_vecs)):
    dq = np.linalg.norm(q_vecs[i] - q_vecs[i-1])
    q_dist.append(q_dist[-1] + dq)
q_dist = np.array(q_dist)

num_segments = len(points_path) - 1
ticks = [q_dist[0]]  
for i in range(1, num_segments+1):
    index = i * n_points
    if index >= len(q_dist):
        index = len(q_dist) - 1
    ticks.append(q_dist[index])
tick_labels = labels 

# Load 2D susceptibility data
data_temp = np.loadtxt("chi_map_0mev_fold.txt")
qx = data_temp[:, 0]
qy = data_temp[:, 1]
chi0 = data_temp[:, 2] 


qx_unique0 = np.unique(qx)
qy_unique0 = np.unique(qy)
Nq_x0 = len(qx_unique0)
Nq_y0 = len(qy_unique0)

# Reshape chi into 2D array
chi_2D_0 = chi0.reshape((Nq_y0, Nq_x0)) 
chi_2D_0 = gaussian_filter(chi_2D_0, sigma=0.5)
scale_factor = 4 
chi_2D_0 = zoom(chi_2D_0, scale_factor)

# Define hexagonal BZ
b1, b2 = 4*np.pi/3, 2*np.pi/np.sqrt(3)
kx_hex = 4*np.pi/3
ky_hex = 2*np.pi/np.sqrt(3)
R = kx_hex

hex_corners = [
    [ R, 0],
    [ R/2,  R*np.sqrt(3)/2],
    [-R/2,  R*np.sqrt(3)/2],
    [-R, 0],
    [-R/2, -R*np.sqrt(3)/2],
    [ R/2, -R*np.sqrt(3)/2],
    [ R, 0]  # close the loop
]
hex_x = [pt[0] for pt in hex_corners]
hex_y = [pt[1] for pt in hex_corners]

Gamma = (0.0, 0.0)
M     = (np.pi, np.pi/np.sqrt(3))
K     = (b1, 0)
high_symmetry_points = {
    r"$\Gamma$": Gamma,
    r"$K$": K,
    r"$M$": M
}

# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})

# 1D susceptibility plot**
ax1 = axes[0]
ax1.plot(q_dist, mu_0, label="μ = 0 eV", color='b')
ax1.set_xticks(ticks)
ax1.set_xticklabels(tick_labels)
for t in ticks:
    ax1.axvline(x=t, color='gray', linestyle='--', linewidth=0.5)

ax1.set_ylabel(r"$\chi(q)$")
ax1.set_xlim(q_dist[0],q_dist[-1])
ax1.set_ylim(1,3)

# 2D susceptibility heatmap**
ax2 = axes[1]
im = ax2.imshow(
    chi_2D_0.T,  
    origin='lower',
    extent=[qx.min(), qx.max(), qy.min(), qy.max()],
    aspect='auto',
    cmap='jet'
)
ax2.plot(hex_x, hex_y, color='black', zorder=2, linestyle='solid', label="Hexagonal BZ")
ax2.set_xlim(-1.5*np.pi,1.5*np.pi)
ax2.set_ylim(-1.5*np.pi,1.5*np.pi)

# high-symmetry points
for label, (qx_hs, qy_hs) in high_symmetry_points.items():
    ax2.scatter(qx_hs, qy_hs, color='black', marker='o', edgecolors='black', s=10, zorder=3)
    ax2.text(qx_hs + 0.1, qy_hs, label, fontsize=12, color='white', fontweight='bold')

ax2.set_xlabel(r"$q_x$")
ax2.set_ylabel(r"$q_y$")

# colourbar
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6]) 
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title(r'$\chi(\mathbf{q})$', fontsize=14)

plt.tight_layout(rect=[0, 0, 0.9, 1]) 
plt.savefig("bulk_elec_sus.png", dpi=1000)
plt.show()
