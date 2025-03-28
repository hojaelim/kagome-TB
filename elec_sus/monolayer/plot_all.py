"""
Plot of the mesh electronic susceptibilities for the different mu values.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def flip_gamma_2d(chi, qx, qy, threshold=1e-6):

    # Create a mask for grid points near Gamma
    mask = np.sqrt(qx**2 + qy**2) < threshold
    
    # Determine a reference value from points outside the Gamma region.
    if np.any(~mask):
        chi_ref = np.mean(chi[~mask])
    else:
        chi_ref = 0.0
    
    chi_new = chi.copy()
    # Mirror the values near Gamma about the reference value
    chi_new[mask] = 2 * chi_ref - chi[mask]
    
    return chi_new

# Load the data from file for each mu case
data_0 = np.loadtxt("chi_map_0mev.txt")
data_6 = np.loadtxt("chi_map_-6mev.txt")
data_9 = np.loadtxt("chi_map_9mev.txt")

# Extract values for each dataset
qx = data_0[:, 0]
qy = data_0[:, 1]
chi0 = np.abs(data_0[:, 2])
chi6 = np.abs(data_6[:, 2])
chi9 = np.abs(data_9[:, 2])

# Get unique qx, qy values (assumed common to all datasets)
qx_unique = np.unique(qx)
qy_unique = np.unique(qy)
Nq_x = len(qx_unique)
Nq_y = len(qy_unique)
qx_mesh, qy_mesh = np.meshgrid(qx_unique, qy_unique)

# Reshape chi into 2D arrays
chi_2D_0 = flip_gamma_2d(chi0.reshape((Nq_y, Nq_x)), qx_mesh, qy_mesh)
chi_2D_6 = flip_gamma_2d(chi6.reshape((Nq_y, Nq_x)), qx_mesh, qy_mesh)
chi_2D_9 = flip_gamma_2d(chi9.reshape((Nq_y, Nq_x)), qx_mesh, qy_mesh)

# Apply Gaussian smoothing
chi_2D_0 = gaussian_filter(chi_2D_0, sigma=0.5)
chi_2D_6 = gaussian_filter(chi_2D_6, sigma=0.5)
chi_2D_9 = gaussian_filter(chi_2D_9, sigma=0.5)

# Determine common color scaling using the overall min and max
vmin = min(np.min(chi_2D_0), np.min(chi_2D_6), np.min(chi_2D_9)) * 0.98
vmax = max(np.max(chi_2D_0), np.max(chi_2D_6), np.max(chi_2D_9)) * 0.98

# Define high-symmetry points for the Brillouin zone
b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2
high_symmetry_points = {
    r"$\Gamma$": (0, 0),
    r"$K$": (0, b2*2/3),
    r"$M$": (b1, b2),
    r"$X$": (b1, 0),
}

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

# Create subplots: one row with three columns
fig, axes = plt.subplots(1, 3, figsize=(10, 6))

# Plot each dataset 
im6 = axes[0].imshow(chi_2D_6.T,
                     origin='lower',
                     extent=[qx.min(), qx.max(), qy.min(), qy.max()],
                     aspect='auto',
                     cmap='jet',
                     vmin=vmin, vmax=vmax)

axes[0].set_xticks([])
axes[0].set_yticks([])

im0 = axes[1].imshow(chi_2D_0.T,
                     origin='lower',
                     extent=[qx.min(), qx.max(), qy.min(), qy.max()],
                     aspect='auto',
                     cmap='jet',
                     vmin=vmin, vmax=vmax)

axes[1].set_xticks([])
axes[1].set_yticks([])


im9 = axes[2].imshow(chi_2D_9.T,
                     origin='lower',
                     extent=[qx.min(), qx.max(), qy.min(), qy.max()],
                     aspect='auto',
                     cmap='jet',
                     vmin=vmin, vmax=vmax)

axes[2].set_xticks([])
axes[2].set_yticks([])



# high-symmetry points on each plot
for ax in axes:
    ax.plot(x, y, color='black', linestyle = 'dotted', zorder=1, label="Rectangular BZ")
    ax.set_xlim(qx.min()+0.2, qx.max()-0.2)
    ax.set_ylim(qy.min()+0.5, qy.max()-0.5)
    for label, (qx_hs, qy_hs) in high_symmetry_points.items():
        ax.scatter(qx_hs, qy_hs, color='black', marker='o', edgecolors='black', s=10, zorder=3)
        ax.text(qx_hs + 0.1, qy_hs, label, fontsize=12, color='white', fontweight='bold')

# Add one common colorbar for all subplots
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.05])
cbar = fig.colorbar(im0, ax=axes, orientation='horizontal', cax = cbar_ax)

cbar.ax.set_ylabel(r'$\chi(k)$', fontsize=14, labelpad=25, rotation=0)  # Set title


plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('monolayer_susceptibility_multi.png', dpi=1000)
plt.show()
