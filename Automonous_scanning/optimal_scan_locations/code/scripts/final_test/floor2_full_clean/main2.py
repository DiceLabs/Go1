import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import open3d as o3d
import random

# def calculate_camera_distance(width_captured_area, fov_angle):
#     # Convert the FOV angle from degrees to radians
#     fov_angle_rad = math.radians(fov_angle)

#     # Calculate the camera distance using the formula
#     camera_distance = (width_captured_area / 2) / math.tan(fov_angle_rad / 2)
#     camera_distance = math.floor(camera_distance * 10) / 10

#     return camera_distance

# def calculate_horizontal_distance(camera_distance, width_captured_area, fov_angle, overlap_percentage):
#     # Convert the FOV angle from degrees to radians
#     fov_angle_rad = math.radians(fov_angle)

#     # Calculate the horizontal distance using the formula
#     horizontal_distance = (width_captured_area / 2) / math.tan(fov_angle_rad / 2) * (1 - overlap_percentage)
#     horizontal_distance = math.floor(horizontal_distance * 10) / 10

#     return horizontal_distance

# # Example usage
# width_captured_area = 0.5  # Width of the captured area (in unit m)
# fov_angle = 60.0  # Field of view angle of the camera (in degrees)
# overlap_percentage = 0.4  # Overlap percentage between images(between 0 and 1)

# camera_distance = calculate_camera_distance(width_captured_area, fov_angle)
# print("Camera Distance:", camera_distance)

# horizontal_distance = calculate_horizontal_distance(camera_distance, width_captured_area, fov_angle, overlap_percentage)
# print("Horizontal Distance:", horizontal_distance)

# ======================================================
# PCD
# ======================================================

mesh = o3d.io.read_triangle_mesh('nonpink.obj')
# mesh = o3d.io.read_triangle_mesh('perpendicular_faces.obj')
pcd = mesh.sample_points_poisson_disk(20000) # Sample points from the mesh

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=16),
    fast_normal_computation=True
)

o3d.visualization.draw_geometries([pcd])  # Works only outside Jupyter/Colab

plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# Remove points below the plane equation
points = np.asarray(pcd.points)
distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
threshold = 0.01 # Adjust this threshold if needed
filtered_indices = np.where(distances > threshold)[0]
filtered_pcd = pcd.select_by_index(filtered_indices)

filtered_pcd.paint_uniform_color([0, 0, 1])  # Paint filtered points blue
o3d.visualization.draw_geometries([filtered_pcd])

# Apply statistical outlier removal filter
pcd_filtered, _ = filtered_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries([pcd_filtered])

# ==============================================

# Extract x and y coordinates from pcd_filtered
points = np.asarray(pcd_filtered.points)
x_coords = points[:, 0]
y_coords = points[:, 1]

# Define the grid boundaries
x_min = math.floor(np.min(x_coords*10)/10)
x_max = math.ceil(np.max(x_coords))
y_min = math.floor(np.min(y_coords*10)/10)
y_max = math.ceil(np.max(y_coords))

print(x_min)

# Define grid parameters
grid_size_x = math.ceil(x_max) - math.floor(x_min)  # Size of the grid in the x-axis
grid_size_y = math.ceil(y_max) - math.floor(y_min)  # Size of the grid in the y-axis
cell_size = 0.1  # Size of each cell

# Calculate the number of cells in each dimension
num_cells_x = int(np.ceil(grid_size_x / cell_size))
num_cells_y = int(np.ceil(grid_size_y / cell_size))

# Create an empty grid
grid = np.zeros((num_cells_y, num_cells_x), dtype=np.uint8)

# Iterate over the points and fill the corresponding cells in the grid
for point in zip(x_coords, y_coords):
    cell_indices = (
        int((point[0] - x_min) / cell_size),
        int((point[1] - y_min) / cell_size)
    )
    grid[cell_indices[1], cell_indices[0]] = 1

# Create a figure and axes
fig, ax = plt.subplots()

# Set the background color to white
ax.set_facecolor('white')

# Plot the grid with black cells and visible grid lines
ax.imshow(grid, cmap='binary', origin='lower', extent=[x_min, x_max, y_min, y_max])
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Grid', fontsize=14)

# Set the axis interval and naming
ax.set_xticks(np.arange(x_min, x_max + cell_size, cell_size))
ax.set_yticks(np.arange(y_min, y_max + cell_size, cell_size))

# Show the grid lines for each cell
ax.grid(color='black', linewidth=0.5)

# Show the plot
plt.show()
