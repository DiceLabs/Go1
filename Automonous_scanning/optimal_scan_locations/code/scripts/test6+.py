# import numpy as np
# import open3d as o3d
# import random

# # Load the mesh and sample points from it
# mesh = o3d.io.read_triangle_mesh('perpendicular_faces.obj')
# pcd = mesh.sample_points_poisson_disk(10000)

# # Estimate normals
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=16),
#     fast_normal_computation=True
# )

# # Apply statistical outlier removal filter
# pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)


# # Generate a random point inside the detected loop
# loop_points = np.asarray(pcd_filtered.points)

# # Calculate the z-coordinate range
# z_min = np.min(loop_points[:, 2])
# z_max = np.max(loop_points[:, 2])
# z_mid = (z_min + z_max) / 2.0
# # Calculate the point to check distance from
# point_to_check = np.array([116.9, 64.66, z_mid])

# # Find the shortest normal distance
# min_distance = float('inf')
# for point in pcd_filtered.points:
#     distance = np.linalg.norm(point - point_to_check)
#     min_distance = min(min_distance, distance)

# print("Shortest normal distance to the point cloud:", min_distance)

# # Print the coordinates of the points in the pcd point cloud
# # for point in pcd.points:
# #     print("Point coordinates:", point)

# ----------------------------------------------


import numpy as np
import open3d as o3d
import random

# Load the mesh and sample points from it
mesh = o3d.io.read_triangle_mesh('perpendicular_faces.obj')
pcd = mesh.sample_points_poisson_disk(10000)

# Estimate normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=16),
    fast_normal_computation=True
)

# Calculate the point to check distance from
point_to_check = np.array([65, 117, 0])

# Check if the point lies inside the closed loop
is_inside = False
ray_direction = np.array([1, 0, 0])  # Ray direction in x-axis

for i in range(len(mesh.triangles)):
    triangle_vertices = mesh.vertices[mesh.triangles[i]]
    intersect_count = 0

    for j in range(3):
        v1 = triangle_vertices[j]
        v2 = triangle_vertices[(j + 1) % 3]

        # Check if the ray intersects with the triangle edge
        if (v1[1] > point_to_check[1]) != (v2[1] > point_to_check[1]):
            t = (point_to_check[1] - v1[1]) / (v2[1] - v1[1])
            intersect_x = v1[0] + t * (v2[0] - v1[0])
            
            # Check if the intersection point is to the right of the point_to_check
            if intersect_x > point_to_check[0]:
                intersect_count += 1

    # Check if the ray crosses the triangle an odd number of times
    if intersect_count % 2 == 1:
        is_inside = not is_inside

if is_inside:
    print("The point lies inside the closed loop.")
else:
    print("The point lies outside the closed loop.")




