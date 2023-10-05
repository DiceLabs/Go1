# import pywavefront
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ========================================================
# Parse the .obj file:
# ========================================================

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


obj_file_path = './data/Wolf.obj'
# obj_file_path = './data/floor2.obj'
vertices, faces = extract_mesh_info(obj_file_path)
print(faces)

m_x = min(vertices[:, 0])
m_y = min(vertices[:, 1])
m_z = min(vertices[:, 2])

for count in range(len(vertices)):
    vertices[count][0] -= m_x
    vertices[count][1] -= m_y
    vertices[count][2] -= m_z

mesh = trimesh.load(obj_file_path)

mesh.show()

# ========================================================
# Visualize it MATPLOTLIB for  x,y,z limits:
# ========================================================

def visualize_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot faces
    for face in faces:
        polygon = Poly3DCollection([vertices[face]])
        polygon.set_alpha(0.5)
        ax.add_collection3d(polygon)

    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])

    plt.show()

# visualize_mesh(vertices, faces)

# ========================================================
# Check if the faces in a mesh are already triangulated:
# ========================================================

def check_triangulation(faces):
    is_triangulated = all(len(face) == 3 for face in faces)
    return is_triangulated

is_triangulated = check_triangulation(faces)
print("Are faces Triangulated:", is_triangulated)

# ========================================================
# Ray casting algorithm:
# ========================================================

# def ray_casting(vertices, faces, grid_width, grid_height):
#     # Create an empty occupancy grid map
#     occupancy_grid = np.zeros((grid_height, grid_width), dtype=bool)

#     # Iterate over each grid cell
#     for i in range(grid_height):
#         for j in range(grid_width):
#             # Cast a ray from the grid cell to check for intersections
#             ray = np.array([j + 0.5, i + 0.5, -100])  # Extend the ray below the mesh
#             intersection_count = 0

#             # Iterate over each face of the mesh
#             for face in faces:
#                 face_vertices = vertices[face]

#                 # Check if the ray intersects the face
#                 intersections = []
#                 for k in range(len(face_vertices)):
#                     p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
#                     if (p1[1] > ray[1]) != (p2[1] > ray[1]):
#                         t = (ray[1] - p1[1]) / (p2[1] - p1[1])
#                         intersection_x = p1[0] + t * (p2[0] - p1[0])
#                         if intersection_x > ray[0]:
#                             intersections.append(intersection_x)

#                 if len(intersections) > 0:
#                     intersection_count += 1

#             # Check if the number of intersections is odd (inside the mesh)
#             if intersection_count % 2 != 0:
#                 occupancy_grid[i, j] = True

#     return occupancy_grid

start = time.time()

def ray_casting(vertices, faces, grid_width, grid_height, cell_size_x, cell_size_y):
    # Calculate the number of cells in each dimension
    num_cells_x = int(grid_width / cell_size_x)
    num_cells_y = int(grid_height / cell_size_y)

    # Create an empty occupancy grid map
    occupancy_grid = np.zeros((num_cells_y, num_cells_x), dtype=bool)

    # Iterate over each grid cell
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Calculate the ray origin for the current grid cell
            ray_origin_x = j * cell_size_x + cell_size_x / 2
            ray_origin_y = i * cell_size_y + cell_size_y / 2
            ray_origin = np.array([ray_origin_x, ray_origin_y, -10])  # Extend the ray below the mesh

            intersection_count = 0

            # Iterate over each face of the mesh
            for face in faces:
                face_vertices = vertices[face]

                # Check if the ray intersects the face
                intersections = []
                for k in range(len(face_vertices)):
                    p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
                    if np.logical_xor(p1[1] > ray_origin[1], p2[1] > ray_origin[1]):
                        t = (ray_origin[1] - p1[1]) / (p2[1] - p1[1])
                        intersection_x = p1[0] + t * (p2[0] - p1[0])
                        if intersection_x > ray_origin[0]:
                            intersections.append(intersection_x)

                if len(intersections) > 0:
                    intersection_count += 1

            # Check if the number of intersections is odd (inside the mesh)
            if intersection_count % 2 != 0:
                occupancy_grid[i, j] = True

    return occupancy_grid

grid_width, grid_height = max(vertices[:, 0]), max(vertices[:, 1])
cell_size_x = 1
cell_size_y = 1

occupancy_grid = ray_casting(vertices, faces, grid_width, grid_height, cell_size_x, cell_size_y)
# print(occupancy_grid)
end = time.time()
print(end-start)

def visualize_occupancy_grid(occupancy_grid):
    plt.imshow(occupancy_grid.astype(int), cmap='binary', origin='lower')

    # Add grid lines
    plt.grid(True, color='black', linewidth=0.5)

    # Set the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Set the axis labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show the plot
    plt.show()

visualize_occupancy_grid(occupancy_grid)