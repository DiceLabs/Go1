import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import random

# def extract_mesh_info(file_path):
#     vertices = []
#     faces = []

#     with open(file_path, 'r') as obj_file:
#         for line in obj_file:
#             line = line.strip()
#             if line.startswith(('v ', 'f ')):
#                 elements = line.split()
#                 if line.startswith('v '):
#                     vertices.append([float(elements[1]), float(elements[2]), float(elements[3])])
#                 elif line.startswith('f '):
#                     face_vertices = []
#                     for element in elements[1:]:
#                         vertex_indices = element.split('/')
#                         face_vertices.append(int(vertex_indices[0]) - 1)
#                     faces.append(face_vertices)

#     return np.array(vertices), np.array(faces)

# def detect_walls_perpendicular_to_xy(vertices, faces):
#     extracted_vertices = []
#     extracted_faces = []

#     for face in faces:
#         # Retrieve the vertices for the face
#         face_vertices = vertices[face]

#         # Calculate the normal vector of the face using cross product
#         vector1 = face_vertices[1] - face_vertices[0]
#         vector2 = face_vertices[2] - face_vertices[0]
#         normal = np.cross(vector1, vector2)

#         # Check if the normal is perpendicular to the x-y plane
#         if np.allclose(normal[2], 0):  # Adjust the threshold as needed
#             extracted_vertices.extend(face_vertices)
#             extracted_faces.append(np.arange(len(face_vertices)) + len(extracted_vertices) - len(face_vertices))

#     return np.array(extracted_vertices), np.array(extracted_faces)

# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# extracted_vertices, extracted_faces = detect_walls_perpendicular_to_xy(vertices, faces)

# # Save the extracted perpendicular faces as a new .obj file
# extracted_mesh = o3d.geometry.TriangleMesh()
# extracted_mesh.vertices = o3d.utility.Vector3dVector(extracted_vertices)
# extracted_mesh.triangles = o3d.utility.Vector3iVector(extracted_faces)

# o3d.io.write_triangle_mesh("perpendicular_faces.obj", extracted_mesh)


# ======================================================
# PCD
# ======================================================

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

# Compute distance threshold for DBSCAN clustering
distance_threshold = 1.0  # Minimum dimension of a side

# Apply DBSCAN clustering
labels = np.array(pcd_filtered.cluster_dbscan(eps=distance_threshold, min_points=50, print_progress=True))

# Get the number of clusters
num_clusters = labels.max() + 1

print("Number of closed loops:", num_clusters)

# Generate random colors for each cluster
cluster_colors = [[random.random(), random.random(), random.random()] for _ in range(num_clusters)]

# Assign colors to each point based on cluster labels
colors = [cluster_colors[label] for label in labels]

# Assign colors to the filtered point cloud
pcd_filtered.colors = o3d.utility.Vector3dVector(colors)

# Visualize the filtered point cloud with colored clusters
o3d.visualization.draw_geometries([pcd_filtered])

# # ============================================

# # Generate a random point inside the detected loop
# loop_points = np.asarray(pcd_filtered.points)
# random_point = np.mean(loop_points, axis=0)  # Compute the centroid of the loop points

# # Calculate the z-coordinate range
# z_min = np.min(loop_points[:, 2])
# z_max = np.max(loop_points[:, 2])
# z_mid = (z_min + z_max) / 2.0

# # Set the z-coordinate of the random point to the midpoint of the z-limits
# random_point[2] = z_mid

# # Calculate the minimum distance between the red point and any of the points in the point cloud
# min_distance = float('inf')
# for point in pcd_filtered.points:
#     distance = np.linalg.norm(point - random_point)
#     min_distance = min(min_distance, distance)

# # Create a big red-colored point
# big_red_point = o3d.geometry.PointCloud()
# big_red_point.points = o3d.utility.Vector3dVector([random_point])
# big_red_point.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color

# # Combine the filtered point cloud and the big red point
# combined_point_cloud = pcd_filtered + big_red_point

# # Visualize the combined point cloud with colored clusters and the big red point
# o3d.visualization.draw_geometries([combined_point_cloud])

# print("Coordinates of the red point:")
# print(random_point)

# print("Minimum distance to the point cloud:", min_distance)

# =====================================================
# Generate a random point inside the detected loop
loop_points = np.asarray(pcd_filtered.points)
random_point = np.mean(loop_points, axis=0)  # Compute the centroid of the loop points

# Calculate the z-coordinate range
z_min = np.min(loop_points[:, 2])
z_max = np.max(loop_points[:, 2])
z_mid = (z_min + z_max) / 2.0

# Set the z-coordinate of the random point to the midpoint of the z-limits
random_point[2] = z_mid

# Find the closest point in the x-direction
x_dist = 0.4
x_indices = np.where(np.abs(loop_points[:, 0] - random_point[0]) <= x_dist)[0]
x_filtered_points = loop_points[x_indices]

# Find the closest point in the y-direction
y_dist = 0.4
y_indices = np.where(np.abs(x_filtered_points[:, 1] - random_point[1]) <= y_dist)[0]
y_filtered_points = x_filtered_points[y_indices]

# If there are any points that satisfy both x and y distances
if len(y_filtered_points) > 0:
    random_point = y_filtered_points[0]  # Choose the first point
else:
    print("No points found within the specified x and y distances.")

# Calculate the minimum distance between the red point and any of the points in the point cloud
min_distance = float('inf')
for point in pcd_filtered.points:
    distance = np.linalg.norm(point - random_point)
    min_distance = min(min_distance, distance)

# Create a big red-colored point
big_red_point = o3d.geometry.PointCloud()
big_red_point.points = o3d.utility.Vector3dVector([random_point])
big_red_point.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color

# Combine the filtered point cloud and the big red point
combined_point_cloud = pcd_filtered + big_red_point

# Visualize the combined point cloud with colored clusters and the big red point
o3d.visualization.draw_geometries([combined_point_cloud])

print("Coordinates of the red point:")
print(random_point)

print("Minimum distance to the point cloud:", min_distance)

# ==========================================

# # Generate a random point inside the detected loop
# loop_points = np.asarray(pcd_filtered.points)
# random_point = np.mean(loop_points, axis=0)  # Compute the centroid of the loop points

# # Calculate the z-coordinate range
# z_min = np.min(loop_points[:, 2])
# z_max = np.max(loop_points[:, 2])
# z_mid = (z_min + z_max) / 2.0

# # Set the z-coordinate of the random point to the midpoint of the z-limits
# random_point[2] = z_mid

# # Calculate the unit direction vector for the random point
# unit_direction = np.random.uniform(-1, 1, size=3)
# unit_direction /= np.linalg.norm(unit_direction)

# # Set the distance of the random point from the loop points
# distance = 0.4

# # Compute the coordinates of the red point
# red_point = random_point + distance * unit_direction

# # Calculate the minimum distance between the red point and any of the points in the point cloud
# min_distance = float('inf')
# for point in pcd_filtered.points:
#     distance = np.linalg.norm(point - red_point)
#     min_distance = min(min_distance, distance)

# # Create a big red-colored point
# big_red_point = o3d.geometry.PointCloud()
# big_red_point.points = o3d.utility.Vector3dVector([red_point])
# big_red_point.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color

# # Combine the filtered point cloud and the big red point
# combined_point_cloud = pcd_filtered + big_red_point

# # Visualize the combined point cloud with colored clusters and the big red point
# o3d.visualization.draw_geometries([combined_point_cloud])

# # Print the coordinates of the red point
# print("Coordinates of the red point:")
# print(red_point)

# print("Minimum distance to the point cloud:", min_distance)


# ======================================================

# # Generate a random point inside the detected loop
# loop_points = np.asarray(pcd_filtered.points)
# random_index = random.randint(0, len(loop_points) - 1)
# random_point = loop_points[random_index]

# # Find the nearest neighbor to the random point
# search_tree = o3d.geometry.KDTreeFlann(pcd_filtered)
# distances, _ = search_tree.search_radius_vector_3d(random_point, 1)
# nearest_distance = distances[0]

# # Calculate the scaling factor to achieve the desired shortest distance
# scaling_factor = 0.4 / nearest_distance

# # Scale the random point to achieve the desired shortest distance
# random_point_scaled = random_point * scaling_factor

# # Create a big red-colored point
# big_red_point = o3d.geometry.PointCloud()
# big_red_point.points = o3d.utility.Vector3dVector([random_point_scaled])
# big_red_point.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color

# # Combine the filtered point cloud and the big red point
# combined_point_cloud = pcd_filtered + big_red_point

# # Visualize the combined point cloud with colored clusters and the big red point
# o3d.visualization.draw_geometries([combined_point_cloud])

# # Print the coordinates of the red point
# print("Coordinates of the red point:")
# print(random_point_scaled)

# print("Desired shortest distance:", 0.4)

# =====================================================

# # Check if the filtered point cloud is empty
# if pcd_filtered.is_empty():
#     print("Filtered point cloud is empty. No closed loops.")
# else:
#     # Compute the convex hull
#     hull = pcd_filtered.compute_convex_hull()

#     # Count the number of connected components in the convex hull
#     labels = hull.cluster_connected_triangles()
#     num_loops = len(set(labels))

#     print("Number of closed loops:", num_loops)

#     # Visualize the closed loops
#     hull.paint_uniform_color([1, 0, 0])  # Color the convex hull

#     o3d.visualization.draw_geometries([pcd_filtered, hull])


# # Project points to XY plane
# points = np.asarray(pcd_filtered.points)
# x_coords = points[:, 0]
# y_coords = points[:, 1]

# # Create grid
# grid_size = 0.5
# x_min, x_max = np.min(x_coords), np.max(x_coords)
# y_min, y_max = np.min(y_coords), np.max(y_coords)
# # Project points to XY plane
# points = np.asarray(pcd_filtered.points)
# x_coords = points[:, 0]
# y_coords = points[:, 1]

# # Create grid
# grid_size = 0.5
# x_min, x_max = np.min(x_coords), np.max(x_coords)
# y_min, y_max = np.min(y_coords), np.max(y_coords)
# x_range = np.arange(x_min, x_max + grid_size, grid_size)
# y_range = np.arange(y_min, y_max + grid_size, grid_size)
# xx, yy = np.meshgrid(x_range, y_range)

# # Visualize the grid and points
# fig, ax = plt.subplots()
# ax.scatter(x_coords, y_coords, color='blue', s=5)
# ax.grid(True)
# ax.set_aspect('equal')

# # Plot grid lines
# for x in x_range:
#     ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
# for y in y_range:
#     ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()
# x_range = np.arange(x_min, x_max + grid_size, grid_size)
# y_range = np.arange(y_min, y_max + grid_size, grid_size)
# xx, yy = np.meshgrid(x_range, y_range)

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