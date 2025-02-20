import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def generate_k_path(points, n_points=50):
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

Gamma = (0.0, 0.0)
K     = (4*np.pi/3, 0.0)
M     = (np.pi, np.pi/np.sqrt(3))

# We want to go Γ -> K -> M -> Γ
points_path = [Gamma, K, M, Gamma]
k_vecs, k_dist = generate_k_path(points_path, n_points=100)



data1 = np.loadtxt("kagome_band.txt")

kx = data1[:, 0]
ky = data1[:, 1]
eigvals = data1[:, 2:]

data2 = np.loadtxt("dos.txt")

dos = data2[:, 0]
bin_edges = data2[:, 1]

mask = (bin_edges >= -4.1) & (bin_edges <= 2.1)
filtered_bin_edges = bin_edges[mask]
filtered_dos = dos[mask]
filtered_bin_edges = np.append(filtered_bin_edges, filtered_bin_edges[-1] + (filtered_bin_edges[-1]-filtered_bin_edges[-2]))
sigma = 300  # Adjust smoothing strength (higher sigma = smoother curve)
smoothed_dos = gaussian_filter1d(filtered_dos, sigma=sigma)
# Compute bin centers from bin edges
bin_centers = (filtered_bin_edges[:-1] + filtered_bin_edges[1:]) / 2



def find_kdist_endpoints(points):
    # Use the same path generation but n_points=1
    kvecs_end, kdist_end = generate_k_path(points, n_points=1)
    # kdist_end[0] = 0, kdist_end[1] = distance(Γ->K), etc.
    return kdist_end

kdist_end = find_kdist_endpoints(points_path)
# kdist_end is length 4+1 = 5. The last is the total path distance.



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for band_idx in range(3):
    ax1.plot(k_dist, eigvals[:, band_idx], color='blue')

# Mark them
for d in kdist_end:
    ax1.axvline(x=d, color='k', linestyle='--', linewidth=0.5)

# Remove x-axis ticks and labels
ax1.set_xticks([])  # Removes x-axis ticks
ax1.set_xlabel("")  # Removes x-axis label

# Label the points
labels = ["Γ", "K", "M", "Γ"]
for d, label in zip(kdist_end, labels):
    ax1.text(d, ax1.get_ylim()[0] - 0.1, label, ha='center', va='top', fontsize=12)

# --- Set Same Y-Range for Both Plots ---
y_min, y_max = -4.1, 2.1  # Get min/max energy from band structure
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

ax2.set_xlim(0,0.4)

ax2.set_yticks([])  # Removes y-axis ticks
ax2.set_xticks([])  # Set custom x-axis ticks
ax2.set_xlabel(r'$\rho(\xi)$', fontsize=14)
ax1.set_ylabel(r'$\xi(\mathbf{k})$', fontsize=14)  # or whatever units

ax2.plot(smoothed_dos, bin_centers)

horizontal_lines = [-2,-1,0]  # Define where to add lines
for y in horizontal_lines:
    ax1.axhline(y=y, linestyle="--", color="black", linewidth=1)
    ax2.axhline(y=y, linestyle="--", color="black", linewidth=1)

# --- Add "(a)" and "(b)" Labels ---
ax1.text(0, 1.05, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold")
ax2.text(0, 1.05, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold")

plt.subplots_adjust(wspace=0.05)  # Reduce horizontal spacing
plt.savefig("band_structure_dos.png", dpi=600)
plt.show()