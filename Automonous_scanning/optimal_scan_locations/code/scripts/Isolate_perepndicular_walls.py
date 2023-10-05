
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

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



