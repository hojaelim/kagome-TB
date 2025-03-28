"""
Plot of the path electronic susceptibilities for the different mu values.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Load data 
data = np.loadtxt("chi_path_results_test.txt")
q_dist = data[:, 0] 

# Extract susceptibility for each mu value
chi_minus_6 = data[:, 3]
chi_0       = data[:, 4]
chi_6       = data[:, 5]
chi_9       = data[:, 6]
chi_16      = data[:, 7]
chi_26      = data[:, 8]

mus = [-6, 0, 6, 9, 16, 26]


def flip_gamma(chi, q_dist, threshold=0.001):

    # Identify indices near Gamma
    gamma_indices = np.where((data[:,1] + data[:,2]) < threshold)[0]
    if gamma_indices.size > 0:
        # Choose reference from data in the interval
        outside_indices = np.where((q_dist >= threshold) & (q_dist < 2*threshold))[0]
        if outside_indices.size > 0:
            chi_ref = np.mean(chi[outside_indices])
            chi_flipped = chi.copy()
            chi_flipped[gamma_indices] = 2 * chi_ref - chi_flipped[gamma_indices]
            return chi_flipped
    return chi

# Apply flipping
chi_minus_6_flipped = flip_gamma(chi_minus_6, q_dist, threshold=0.05)
chi_0_flipped       = flip_gamma(chi_0,       q_dist, threshold=0.05)
chi_6_flipped       = flip_gamma(chi_6,       q_dist, threshold=0.05)
chi_9_flipped       = flip_gamma(chi_9,       q_dist, threshold=0.05)
chi_16_flipped      = flip_gamma(chi_16,      q_dist, threshold=0.05)
chi_26_flipped      = flip_gamma(chi_26,      q_dist, threshold=0.05)

chi_arrays = [
    chi_minus_6_flipped, 
    chi_0_flipped, 
    chi_6_flipped, 
    chi_9_flipped, 
    chi_16_flipped, 
    chi_26_flipped
]

plt.figure(figsize=(8, 5))


offset_step = 3.0  
base_color = 'red'

# Plot from top to bottom
for i, (mu_val, chi_vals) in enumerate(zip(mus, chi_arrays)):
    offset = (len(mus) - 1 - i) * offset_step
    alpha = 0.3 + 0.7 * (i / (len(mus) - 1))
    
    # Plot the curve offset
    plt.plot(q_dist, chi_vals + offset, color=base_color, alpha=alpha)

b1 = np.pi/3 * np.sqrt(3)/2
b2 = np.pi/np.sqrt(3) * np.sqrt(3)/2
Gamma = np.array([0.0, 0.0])
X = np.array([b1, 0.0])
M = np.array([b1, b2])
Y = np.array([0.0, b2])

points_path = [Gamma, X, M, Y, Gamma]
labels = [r'$\Gamma$', 'X', 'M', 'Y', r'$\Gamma$']

def generate_k_path(points, n_points=500):
    q_vecs = []
    for i in range(len(points)-1):
        start, end = points[i], points[i+1]
        for alpha in np.linspace(0, 1, n_points):
            q_vecs.append((1 - alpha)*start + alpha*end)
    return np.array(q_vecs)

n_points = 50
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

plt.xlim(0, q_dist[-1])
plt.ylim(14,36)
plt.xticks(ticks=ticks, labels=tick_labels)
for t in ticks:
    plt.axvline(x=t, color='gray', linestyle='--', linewidth=0.5)

plt.ylabel(r"$\chi(q)$")
plt.yticks([])

plt.tight_layout()
plt.savefig("chi_path.png")
plt.show()

