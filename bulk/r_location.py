import numpy as np
import matplotlib.pyplot as plt

r1 = np.array([0, 1])
r2 = np.array([np.sqrt(3)/2, -1/2])
r3 = np.array([-np.sqrt(3)/2, -1/2])

origin = [0,0]
A = origin
B = origin-r2
C = origin-r3

points = np.array([A, B, C, origin+r2, origin+r3])
point_labels = np.array(["A", "B", "C", "", ""])
point_colors = {
    "A": "red",
    "B": "blue",
    "C": "green"
}

pseudo = np.array([origin+r1, origin-r1,[np.sqrt(3),0], [-np.sqrt(3),0]])
max_y = pseudo[np.argmax(pseudo[:, 1])][1]
max_x = pseudo[np.argmax(pseudo[:, 0])][0]


connections = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4)
]

corners = np.array([pseudo[np.argmax(pseudo[:, 0])],  pseudo[np.argmax(pseudo[:, 1])], pseudo[np.argmin(pseudo[:, 0])], pseudo[np.argmin(pseudo[:, 1])], pseudo[np.argmax(pseudo[:, 0])]])

plt.figure(figsize=(2*np.sqrt(3), 2), dpi=300)

for i in range(len(corners)-1):
    plt.plot([corners[i][0], corners[i+1][0]], [corners[i][1], corners[i+1][1]], color='r', zorder=2)

for i in range(len(points)):
    label = point_labels[i]
    colour = point_colors.get(label[0], "red") if label else "gray"  # Check if label is not empty
    plt.scatter(points[i][0], points[i][1], color = colour, zorder=3)

plt.text(points[0][0], points[0][1] +0.1, point_labels[0])
plt.text(points[1][0]-0.1, points[1][1] +0.1, point_labels[1])
plt.text(points[2][0], points[2][1] +0.1, point_labels[2])
plt.text(points[4][0]-0.1, points[4][1] -0.25, "C", color = "grey")
plt.text(points[3][0], points[3][1] -0.25, "B", color = "grey")

for idx1, idx2 in connections:
    plt.plot([points[idx1][0], points[idx2][0]], [points[idx1][1], points[idx2][1]], linestyle="solid", color="gray", alpha = 0.8, zorder=1)

plt.axis("off")
plt.tight_layout()

plt.show()
#plt.savefig("r_location.png", dpi=600)