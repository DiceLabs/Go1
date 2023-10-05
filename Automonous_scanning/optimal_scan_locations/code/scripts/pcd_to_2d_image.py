import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import random

mesh = o3d.io.read_triangle_mesh('perpendicular_faces.obj')
pcd = mesh.sample_points_poisson_disk(10000) # Sample points from the mesh
print(len(pcd.points))

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=16),
    fast_normal_computation=True
)

o3d.visualization.draw_geometries([pcd])  # Works only outside Jupyter/Colab

# Apply statistical outlier removal filter
pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries([pcd_filtered])

# Project points to XY plane
points = np.asarray(pcd_filtered.points)
x_coords = points[:, 0]
y_coords = points[:, 1]

# Create grid
grid_size = 0.5
x_min, x_max = np.min(x_coords), np.max(x_coords)
y_min, y_max = np.min(y_coords), np.max(y_coords)
# Project points to XY plane
points = np.asarray(pcd_filtered.points)
x_coords = points[:, 0]
y_coords = points[:, 1]

x_range = np.arange(x_min, x_max + grid_size, grid_size)
y_range = np.arange(y_min, y_max + grid_size, grid_size)
xx, yy = np.meshgrid(x_range, y_range)

# # Visualize the grid and points
# fig, ax = plt.subplots()
# ax.scatter(x_coords, y_coords, color='black', s=1)
# ax.grid(True)
# ax.set_aspect('equal')

# # Plot grid lines
# for x in x_range:
#     ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.2)
# for y in y_range:
#     ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.2)

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()

# Save the figure
# fig.savefig('test.png')

# Visualize the grid and points
fig, ax = plt.subplots()
ax.scatter(x_coords, y_coords, color='black', s=1)
ax.axis('off')  # Remove the axis lines and labels

# Save the figure
fig.savefig('test.png', bbox_inches='tight', pad_inches=0)
plt.show()


# =====================

# =====================================

# from sklearn.cluster import DBSCAN
# from scipy.spatial import ConvexHull

# # Set the minimum length threshold for the intersecting wall
# minimum_wall_length = 1.0  # meters

# # Convert points to NumPy array
# points_array = np.asarray(pcd_filtered.points)

# # Perform DBSCAN clustering on the point cloud data
# eps = 0.3  # DBSCAN neighborhood distance
# min_samples = 10  # Minimum number of points to form a cluster
# dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# labels = dbscan.fit_predict(points_array)

# # Get the unique cluster labels
# unique_labels = np.unique(labels)

# # Initialize a list to store the corner points
# corner_points = []

# # Iterate over each unique cluster label
# for label in unique_labels:
#     # Skip noise points (label = -1)
#     if label == -1:
#         continue

#     # Get the points belonging to the current cluster
#     cluster_points = points_array[labels == label]

#     # Check if the cluster has enough points to form a wall segment
#     if len(cluster_points) >= minimum_wall_length / grid_size:
#         # Calculate the convex hull of the cluster points
#         hull = ConvexHull(cluster_points[:, :2])  # Consider only X and Y coordinates

#         # Find the indices of the vertices with the largest angles
#         max_angle_indices = np.argmax(hull.simplices, axis=1)

#         # Append the corner points to the list
#         corner_points.extend(hull.points[max_angle_indices])

# corner_points = np.asarray(corner_points)

# # Visualize the corners
# fig, ax = plt.subplots()
# ax.scatter(x_coords, y_coords, color='black', s=1)
# ax.scatter(corner_points[:, 0], corner_points[:, 1], color='red', s=10)
# ax.grid(True)
# ax.set_aspect('equal')

# # Plot grid lines
# for x in x_range:
#     ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.2)
# for y in y_range:
#     ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.2)

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()

# ==============================
