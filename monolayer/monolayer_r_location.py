"""
Plots the real space representation of the monolayer unit cel
"""

import numpy as np
import matplotlib.pyplot as plt

# Define real space vectors
r1 = 1/2 * np.array([np.sqrt(3),1])
r2 = 1/2 * np.array([np.sqrt(3),-1])
r3 = np.array([0,1])
r4 = r2-r3
r5 = r1+r3
r6 = r1+r2

# Define point coordinates
origin = [0,0]
A1 = origin+r3
A2 = origin-r6
B1 = origin+r2
B2 = origin-r2
C1 = origin+r1
C2 = origin-r1

points = np.array([A1, A2, B1, B2, C1, C2, origin-r3, origin+r6])
point_labels = np.array(["A1", "A2", "B1", "B2", "C1", "C2", "", ""])
point_colors = {
    "A": "red",
    "B": "blue",
    "C": "green"
}

# Connect hoppings
connections = [
    (0, 3),
    (3, 1),
    (1, 5),
    (5, 6),
    (2, 6),
    (2, 7),
    (4, 7),
    (4, 0)
]

# Define corners
max_y = points[np.argmax(points[:, 1])][1]
max_x = points[np.argmax(points[:, 0])][0]
corners = np.array([[max_x,-max_y],[max_x, max_y],[-max_x, max_y],[-max_x, -max_y],[max_x,-max_y]])

plt.figure(figsize=(np.sqrt(3)*2, 2), dpi=300)

# Plot box
for i in range(len(corners)-1):
    plt.plot([corners[i][0], corners[i+1][0]], [corners[i][1], corners[i+1][1]], color='r', zorder=2)

# Plot Origin
plt.scatter(origin[0], origin[1], marker='x', color='k')

# Plot points & labels
for i in range(len(points)):
    label = point_labels[i]
    colour = point_colors.get(label[0], "red") if label else "gray"  # Check if label is not empty
    plt.scatter(points[i][0], points[i][1], color = colour, zorder=3)
    plt.text(points[i][0]+0.1, points[i][1] +0.1, point_labels[i])

# Plot out of bounds points
plt.text(points[6][0]+0.1, points[6][1] -0.2, "A1", color = "grey")
plt.text(points[7][0]+0.1, points[7][1] +0.1, "A2", color = "grey")

# Plot connections between points
for idx1, idx2 in connections:
    plt.plot([points[idx1][0], points[idx2][0]], [points[idx1][1], points[idx2][1]], linestyle="solid", color="gray", alpha = 0.8, zorder=1)

# Plot vertical lines (also connections)
plt.vlines(x=B2[0], ymin=-max_y, ymax=max_y, color='gray', linestyle='solid', alpha=0.8, zorder=1)
plt.vlines(x=B1[0], ymin=-max_y, ymax=max_y, color='gray', linestyle='solid', alpha=0.8, zorder=1)
plt.axis("off")
plt.tight_layout()
plt.show()
#plt.savefig("r_location.png", dpi=600)