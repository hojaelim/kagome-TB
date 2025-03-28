"""
Plot for the zoomed monolayer band and dos plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Generate the k point path along the given points
def generate_k_path(points, n_points=1000):

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
                k_dist.append(k_dist[-1] + np.sqrt(dx**2 + dy**2))

    # Append the final end point
    k_vecs.append(points[-1])
    dx = points[-1][0] - k_vecs[-2][0]
    dy = points[-1][1] - k_vecs[-2][1]
    k_dist.append(k_dist[-1] + np.sqrt(dx**2 + dy**2))

    return k_vecs, k_dist

# Define k point path from High symmetry points
b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2

Gamma = [0.0, 0.0]
X = [b1, 0.0]  
K = [0.0, b2*2/3]
M = [b1, b2] 
Y = [0, b2]

points_path = [X, Gamma, K, M, X]
k_vecs, k_dist = generate_k_path(points_path, n_points=10000)


# Load band data
data1 = np.loadtxt("kagome_6_band.txt")

kx = data1[:, 0]
ky = data1[:, 1]
eigvals = data1[:, 2:]

# Load DOS data
data2 = np.loadtxt("dos_kagome_6.txt")

dos = data2[:, 0]
bin_edges = data2[:, 1]

# Mask for filtering value range & Gaussian smoothing
mask = (bin_edges >= -0.04) & (bin_edges <= 0.04)
filtered_bin_edges = bin_edges[mask]
filtered_dos = dos[mask]
filtered_bin_edges = np.append(filtered_bin_edges, filtered_bin_edges[-1] + (filtered_bin_edges[-1]-filtered_bin_edges[-2]))
sigma = 5 
smoothed_dos = gaussian_filter1d(filtered_dos, sigma=sigma)

# Compute bin centers from bin edges
bin_centers = (filtered_bin_edges[:-1] + filtered_bin_edges[1:]) / 2
max_dos_energy = bin_centers[np.argmax(smoothed_dos)]  


# Find distance between high symmetry points for plotting
def find_kdist_endpoints(points):
    _, kdist_end = generate_k_path(points, n_points=1)
    return kdist_end

kdist_end = find_kdist_endpoints(points_path)

# Plot bands
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5), gridspec_kw={'width_ratios': [3, 2]})
for band_idx in range(6):
    ax1.plot(k_dist, eigvals[:, band_idx], color='blue')

# Plot band v lines
for d in kdist_end:
    ax1.axvline(x=d, color='k', linestyle='--', linewidth=0.5)

# Remove x-axis ticks and labels
ax1.set_xticks([])  
ax1.set_xlabel("")  

# Label the points
labels = ["X", "Γ", "K", "M", "X"]
ax1.set_xticks(kdist_end, labels)

# Set y range
y_min, y_max = -0.04, 0.04 
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)


M_index = points_path.index(M)  # Find index of M in points_path
M_kdist = kdist_end[M_index]
x_range = 0.4  # Adjust the range width as needed
ax1.set_xlim(M_kdist - x_range, M_kdist + x_range)
ax2.set_xlim(np.min(smoothed_dos), np.max(smoothed_dos)+0.1)
ax1.set_yticks([-0.04,0.04])
ax2.set_yticks([])  # Removes y-axis ticks
ax2.set_xticks([])  # Set custom x-axis ticks
ax2.set_xlabel(r'$\rho(\xi)$', fontsize=14)

ax2.plot(smoothed_dos, bin_centers)

horizontal_lines = [-2,-1,0]  # Define where to add lines
for y in horizontal_lines:
    ax1.axhline(y=y, linestyle="--", color="black", linewidth=1)
    ax2.axhline(y=y, linestyle="--", color="black", linewidth=1)

interp_func = interp1d(bin_centers, smoothed_dos, bounds_error=False, fill_value="extrapolate")
x_max_dos = interp_func(0.0095)  # Get x where y=0.0095

ax2.hlines(y=0.0095, xmin=np.min(smoothed_dos) , xmax = x_max_dos, linestyle="dotted", color="red", linewidth=1)
ax2.hlines(y=max_dos_energy, xmin=np.min(smoothed_dos) , xmax = np.max(smoothed_dos), linestyle="dotted", color="black", linewidth=1)
# Add (a) and (b) Labels 
ax1.text(0, 1.05, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold")
ax2.text(0, 1.05, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold")
#Add text annotations for VHS points 
ax2.text(0.65, 0.01, "Type-I VHS", fontsize=10, verticalalignment="bottom", horizontalalignment="left", color="black")
ax2.text(0.8, -0.01, "Type-II VHS", fontsize=10, verticalalignment="top", horizontalalignment="left", color="black")
ax1.set_ylabel(r'$\xi(\mathbf{k})$', fontsize=14) 
ax1.yaxis.set_label_coords(-0.1, 0.5) 


plt.tight_layout()
plt.subplots_adjust(wspace=0.05) 
plt.savefig("6_band_structure_dos.png", dpi=600)
plt.show()