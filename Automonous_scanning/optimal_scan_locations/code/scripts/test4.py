# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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

# def calculate_face_normals(vertices, faces):
#     face_normals = []

#     for face in faces:
#         # Retrieve the vertices for the face
#         face_vertices = vertices[face]

#         # Calculate the normal vector of the face using cross product
#         vector1 = face_vertices[1] - face_vertices[0]
#         vector2 = face_vertices[2] - face_vertices[0]
#         normal = np.cross(vector1, vector2)

#         face_normals.append(normal)

#     return np.array(face_normals)

# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# face_normals = calculate_face_normals(vertices, faces)

# # Plot the mesh
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for face in faces:
#     face_vertices = vertices[face]
#     x_coords = face_vertices[:, 0]
#     y_coords = face_vertices[:, 1]
#     z_coords = face_vertices[:, 2]

#     ax.plot(x_coords, y_coords, z_coords, color='blue')

# # Plot the face normals
# for face, normal in zip(faces, face_normals):
#     face_vertices = vertices[face]
#     face_center = np.mean(face_vertices, axis=0)
#     normal_endpoint = face_center + normal

#     x_coords = [face_center[0], normal_endpoint[0]]
#     y_coords = [face_center[1], normal_endpoint[1]]
#     z_coords = [face_center[2], normal_endpoint[2]]

#     ax.plot(x_coords, y_coords, z_coords, color='red')

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# ========================================================================================================================
# TOP-VIEW IMAGE OF THE WALLS PERPENDICULAR TO X-Y PLANE
# ========================================================================================================================

# import numpy as np
# import trimesh
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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


# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)

# # Create a trimesh object
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# # Compute face normals
# face_normals = mesh.face_normals

# # Plot the mesh
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
#                 triangles=mesh.faces, color='lightgray', edgecolor='k')

# # Plot the face normals
# for face_normal, face in zip(face_normals, faces):
#     face_vertices = vertices[face]
#     face_center = np.mean(face_vertices, axis=0)
#     normal_endpoint = face_center + face_normal

#     x_coords = [face_center[0], normal_endpoint[0]]
#     y_coords = [face_center[1], normal_endpoint[1]]
#     z_coords = [face_center[2], normal_endpoint[2]]

#     ax.plot(x_coords, y_coords, z_coords, color='red')

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# ========================================================================================================================
# TEST 
# ========================================================================================================================

# import numpy as np
# import trimesh
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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


# def detect_walls_perpendicular_to_xz(vertices, faces):
#     walls = []

#     for face in faces:
#         # Retrieve the vertices for the face
#         face_vertices = vertices[face]

#         # Calculate the normal vector of the face using cross product
#         vector1 = face_vertices[1] - face_vertices[0]
#         vector2 = face_vertices[2] - face_vertices[0]
#         normal = np.cross(vector1, vector2)

#         # Check if the normal is perpendicular to the x-z plane and points towards +x axis
#         if np.allclose(normal[1], 0) and normal[0] > 0:  # Adjust the threshold as needed
#             walls.append(face)

#     return walls


# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# wall_faces = detect_walls_perpendicular_to_xz(vertices, faces)

# # Create a trimesh object
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# # Compute face normals
# face_normals = mesh.face_normals

# # Plot the mesh
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
#                 triangles=mesh.faces, color='lightgray', edgecolor='k')

# # Plot the face normals
# for face_normal, face in zip(face_normals, faces):
#     if any(np.array_equal(face, wall_face) for wall_face in wall_faces):
#         face_vertices = vertices[face]
#         face_center = np.mean(face_vertices, axis=0)
#         normal_endpoint = face_center + face_normal

#         x_coords = [face_center[0], normal_endpoint[0]]
#         y_coords = [face_center[1], normal_endpoint[1]]
#         z_coords = [face_center[2], normal_endpoint[2]]

#         ax.plot(x_coords, y_coords, z_coords, color='red')

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()


# ========================================================================================================================
# ISOLATE THE FACES THAT HAVE NORMALS IN +X AXIS
# ========================================================================================================================

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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
#     walls = []

#     for face in faces:
#         # Retrieve the vertices for the face
#         face_vertices = vertices[face]

#         # Calculate the normal vector of the face using cross product
#         vector1 = face_vertices[1] - face_vertices[0]
#         vector2 = face_vertices[2] - face_vertices[0]
#         normal = np.cross(vector1, vector2)

#         # Check if the normal is perpendicular to the x-y plane and pointing towards +x axis
#         if np.allclose(normal[2], 0) and np.all(normal[:2] >= 0):  # Adjust the threshold as needed
#             walls.append(face)

#     return walls

# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# wall_faces = detect_walls_perpendicular_to_xy(vertices, faces)

# # Plot the walls
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for face in wall_faces:
#     face_vertices = vertices[face]
#     x_coords = face_vertices[:, 0]
#     y_coords = face_vertices[:, 1]
#     z_coords = face_vertices[:, 2]

#     ax.plot(x_coords, y_coords, z_coords, color='blue')

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()


# ========================================================================================================================
# 
# ========================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_mesh_info(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            line = line.strip()
            if line.startswith(('v ', 'f ')):
                elements = line.split()
                if line.startswith('v '):
                    vertices.append([float(elements[1]), float(elements[2]), float(elements[3])])
                elif line.startswith('f '):
                    face_vertices = []
                    for element in elements[1:]:
                        vertex_indices = element.split('/')
                        face_vertices.append(int(vertex_indices[0]) - 1)
                    faces.append(face_vertices)

    return np.array(vertices), np.array(faces)

# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)

# # Plot all the vertices
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x_coords = vertices[:, 0]
# y_coords = vertices[:, 1]
# z_coords = vertices[:, 2]

# ax.scatter(x_coords, y_coords, z_coords, color='red')

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# # ===================================================================================================================================
# # WORKING CODE TO ISOLATE THE VERTICAL WALLS
# # ===================================================================================================================================

def detect_walls_perpendicular_to_xy(vertices, faces):
    walls = []

    for face in faces:
        # Retrieve the vertices for the face
        face_vertices = vertices[face]

        # Calculate the normal vector of the face using cross product
        vector1 = face_vertices[1] - face_vertices[0]
        vector2 = face_vertices[2] - face_vertices[0]
        normal = np.cross(vector1, vector2)

        # Check if the normal is perpendicular to the x-y plane
        if np.allclose(normal[2], 0):  # Adjust the threshold as needed
            walls.append(face)

    return walls

obj_file_path = './nonpink.obj'
vertices, faces = extract_mesh_info(obj_file_path)
wall_faces = detect_walls_perpendicular_to_xy(vertices, faces)

# Plot the walls
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for face in wall_faces:
    face_vertices = vertices[face]
    x_coords = face_vertices[:, 0]
    y_coords = face_vertices[:, 1]
    z_coords = face_vertices[:, 2]

    ax.plot(x_coords, y_coords, z_coords, color='blue')

# Set labels and show the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# ===================================================================================================================================
# WORKING CODE TO ISOLATE THE VERTICAL WALLS-PART 2 - TO PLOT THE TOP-DOWN IMAGE
# ===================================================================================================================================

# Create a top-down image
fig, ax = plt.subplots()

for face in wall_faces:
    face_vertices = vertices[face]
    x_coords = face_vertices[:, 0]
    y_coords = face_vertices[:, 1]

    ax.plot(x_coords, y_coords, color='blue')

# Set labels and show the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
plt.show()


# ========================================================================================================================
# 
# ========================================================================================================================

# import numpy as np
# import matplotlib.pyplot as plt

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

# # ==============================================================================
# # WORKING CODE TO ISOLATE THE VERTICAL WALLS - PART 1
# # ==============================================================================

# def detect_walls_perpendicular_to_xy(vertices, faces):
#     walls = []

#     for face in faces:
#         # Retrieve the vertices for the face
#         face_vertices = vertices[face]

#         # Calculate the normal vector of the face using cross product
#         vector1 = face_vertices[1] - face_vertices[0]
#         vector2 = face_vertices[2] - face_vertices[0]
#         normal = np.cross(vector1, vector2)

#         # Check if the normal is perpendicular to the x-y plane
#         if np.allclose(normal[2], 0):  # Adjust the threshold as needed
#             walls.append(face)

#     return walls

# def calculate_distance(point, line_start, line_end):
#     # Calculate the perpendicular distance between a point and a line segment
#     line_vector = line_end - line_start
#     point_vector = point - line_start
#     line_length = np.linalg.norm(line_vector)

#     if line_length == 0:
#         return np.inf  # Assign a large distance if the line length is zero

#     line_direction = line_vector / line_length
#     projection = np.dot(point_vector, line_direction)
#     projection = np.clip(projection, 0, line_length)
#     closest_point = line_start + projection * line_direction
#     distance = np.linalg.norm(point - closest_point)
#     return distance

# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# wall_faces = detect_walls_perpendicular_to_xy(vertices, faces)

# # ==============================================================================
# # WORKING CODE TO ISOLATE THE VERTICAL WALLS - PART 2 - TO PLOT THE TOP-DOWN IMAGE
# # ==============================================================================

# # Create a top-down image
# fig, ax = plt.subplots()

# for face in wall_faces:
#     face_vertices = vertices[face]
#     x_coords = face_vertices[:, 0]
#     y_coords = face_vertices[:, 1]

#     ax.plot(x_coords, y_coords, color='blue')

#     # Find the points that are exactly 0.4 units away from the faces
#     for i in range(len(x_coords)):
#         point = np.array([x_coords[i], y_coords[i]])
#         line_start = np.array([x_coords[i - 1], y_coords[i - 1]])
#         line_end = np.array([x_coords[i], y_coords[i]])
#         distance = calculate_distance(point, line_start, line_end)
#         if np.isclose(distance, 0.4):  # Adjust the threshold as needed
#             ax.plot(point[0], point[1], 'ro')

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_aspect('equal')
# plt.show()

# ========================================================================================================================
# 
# ========================================================================================================================

# import numpy as np
# import matplotlib.pyplot as plt

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

# # ==============================================================================
# # WORKING CODE TO ISOLATE THE VERTICAL WALLS - PART 1
# # ==============================================================================

# def detect_walls_perpendicular_to_xy(vertices, faces):
#     walls = []

#     for face in faces:
#         # Retrieve the vertices for the face
#         face_vertices = vertices[face]

#         # Calculate the normal vector of the face using cross product
#         vector1 = face_vertices[1] - face_vertices[0]
#         vector2 = face_vertices[2] - face_vertices[0]
#         normal = np.cross(vector1, vector2)

#         # Check if the normal is perpendicular to the x-y plane
#         if np.allclose(normal[2], 0):  # Adjust the threshold as needed
#             walls.append(face)

#     return walls

# def calculate_distance(point, line_start, line_end):
#     # Calculate the perpendicular distance between a point and a line segment
#     line_vector = line_end - line_start
#     point_vector = point - line_start
#     line_length = np.linalg.norm(line_vector)

#     if line_length == 0:
#         return np.inf  # Assign a large distance if the line length is zero

#     line_direction = line_vector / line_length
#     projection = np.dot(point_vector, line_direction)
#     projection = np.clip(projection, 0, line_length)
#     closest_point = line_start + projection * line_direction
#     distance = np.linalg.norm(point - closest_point)
#     return distance

# obj_file_path = './nonpink.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# wall_faces = detect_walls_perpendicular_to_xy(vertices, faces)

# # ==============================================================================
# # WORKING CODE TO ISOLATE THE VERTICAL WALLS - PART 2 - TO PLOT THE TOP-DOWN IMAGE
# # ==============================================================================

# # Create a top-down image
# fig, ax = plt.subplots()

# # Create a 2D grid of 0.1 units
# x_grid = np.arange(0, 10, 0.1)
# y_grid = np.arange(0, 10, 0.1)
# occupancy_grid = np.zeros((len(y_grid), len(x_grid)))  # Initialize with all zeros

# for face in wall_faces:
#     face_vertices = vertices[face]
#     x_coords = face_vertices[:, 0]
#     y_coords = face_vertices[:, 1]

#     ax.plot(x_coords, y_coords, color='blue')

#     # Find the points that are exactly 0.4 units away from the faces
#     for i in range(len(x_coords)):
#         point = np.array([x_coords[i], y_coords[i]])
#         line_start = np.array([x_coords[i - 1], y_coords[i - 1]])
#         line_end = np.array([x_coords[i], y_coords[i]])
#         distance = calculate_distance(point, line_start, line_end)
#         if np.isclose(distance, 0.4):  # Adjust the threshold as needed
#             ax.plot(point[0], point[1], 'ro')

#             # Calculate the indices in the occupancy grid for marking occupied lines
#             x_index = int(point[0] * 10)
#             y_index = int(point[1] * 10)
#             occupancy_grid[y_index, x_index] = 1

# # Set labels and show the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_aspect('equal')

# # Add grid lines to the plot
# ax.set_xticks(np.arange(0, 10, 1))
# ax.set_yticks(np.arange(0, 10, 1))
# ax.grid(True, linestyle='--', color='gray', linewidth=0.5)

# # Save the figure
# plt.savefig('test.png')

# # Display the occupancy grid
# print("Occupancy Grid:")
# print(occupancy_grid)




