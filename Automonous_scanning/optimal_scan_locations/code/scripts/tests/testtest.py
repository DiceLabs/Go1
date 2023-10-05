# import numpy as np
# import matplotlib.pyplot as plt
# # import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Create vertices
# vertices = np.array([
#     [0, 0, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 1],
#     [1, 1, 1],
#     [0, 1, 1]
# ])

# # Create faces
# faces = np.array([
#     [0, 1, 2],
#     [0, 2, 3],
#     [4, 5, 6],
#     [4, 6, 7],
#     [0, 1, 5],
#     [0, 4, 5],
#     [1, 2, 6],
#     [1, 5, 6],
#     [2, 3, 7],
#     [2, 6, 7],
#     [3, 0, 4],
#     [3, 4, 7]
# ])

# # Plot the mesh
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_trisurf(
#     vertices[:, 0],
#     vertices[:, 1],
#     vertices[:, 2],
#     triangles=faces,
#     # color="darkgray"
# )

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

# # Define the dimensions of the occupancy grid
# grid_width = 5
# grid_height = 5

# def ray_casting(vertices, faces, grid_width, grid_height):
#     # Create an empty occupancy grid map
#     occupancy_grid = np.zeros((grid_height, grid_width), dtype=bool)

#     # Iterate over each grid cell
#     for i in range(grid_height):
#         for j in range(grid_width):
#             # Cast a ray from the grid cell to check for intersections
#             ray = np.array([j + 0.5, i + 0.5, 1])
#             intersection_count = 0

#             # Iterate over each face of the mesh
#             for face in faces:
#                 face_vertices = vertices[face]

#                 # Check if the ray intersects the face
#                 intersect = False
#                 for k in range(len(face_vertices)):
#                     p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
#                     if ((p1[1] > ray[1]) != (p2[1] > ray[1])) and (ray[0] < (p2[0] - p1[0]) * (ray[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]):
#                         intersect = not intersect

#                 if intersect:
#                     intersection_count += 1

#             # Check if the number of intersections is odd (inside the mesh)
#             if intersection_count % 2 != 0:
#                 occupancy_grid[i, j] = True

#     return occupancy_grid

# # Perform ray casting to generate the occupancy grid map
# occupancy_grid = ray_casting(vertices, faces, grid_width, grid_height)

# # Visualize the occupancy grid map
# plt.imshow(occupancy_grid, cmap='binary')
# plt.gca().invert_yaxis()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create vertices
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# Create faces
faces = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7]
])

# Define the dimensions of the occupancy grid map
grid_width = 5
grid_height = 5

# Define the source of light position
light_source_z = 2

# Define the starting position of the rays (in the x-y plane)
ray_start_z = -1

# Create an empty grid for storing the ray tracing result
ray_trace_grid = np.zeros((grid_height, grid_width))

# Iterate over each grid cell
for i in range(grid_height):
    for j in range(grid_width):
        # Calculate the direction of the ray
        ray_direction = np.array([j + 0.5, i + 0.5, light_source_z]) - np.array([j + 0.5, i + 0.5, ray_start_z])
        ray_direction /= np.linalg.norm(ray_direction)

        # Initialize the intersection flag
        intersected = False

        # Perform ray casting
        for face in faces:
            face_vertices = vertices[face]

            # Calculate the normal vector of the face
            face_normal = np.cross(face_vertices[-1] - face_vertices[0], face_vertices[1] - face_vertices[0])
            face_normal /= np.linalg.norm(face_normal)

            # Calculate the distance from the ray start point to the face
            t = np.dot(face_vertices[0] - np.array([j + 0.5, i + 0.5, ray_start_z]), face_normal) / np.dot(ray_direction, face_normal)

            # Check if the ray intersects the face within the valid range
            if 0 < t < light_source_z - ray_start_z:
                intersected = True
                break

        # Set the corresponding grid cell value based on intersection
        if intersected:
            ray_trace_grid[i, j] = 1

# Visualize the ray tracing result
plt.imshow(ray_trace_grid, cmap='binary')
plt.gca().invert_yaxis()
plt.show()