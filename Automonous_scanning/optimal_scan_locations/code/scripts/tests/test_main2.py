import pywavefront
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

obj_file_path = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor2.obj'
vertices, faces = extract_mesh_info(obj_file_path)
m_x = min(vertices[:, 0])
m_y = min(vertices[:, 1])
m_z = min(vertices[:, 2])

for count in range(len(vertices)):
    vertices[count][0] -= m_x
    vertices[count][1] -= m_y
    vertices[count][2] -= m_z

mesh = trimesh.load(obj_file_path)

mesh.show()

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

visualize_mesh(vertices, faces)

# ========================================================
# Check if the faces in a mesh are already triangulated:
# ========================================================

def check_triangulation(faces):
    is_triangulated = all(len(face) == 3 for face in faces)
    return is_triangulated

is_triangulated = check_triangulation(faces)
print("Are faces Triangulated:", is_triangulated)


# ========================================================
# Create an empty grid:
# ========================================================

# def create_occupancy_grid(size, resolution):
#     grid_size = int(size / resolution)
#     occupancy_grid = np.zeros((grid_size, grid_size), dtype=bool)
#     return occupancy_grid

# ========================================================
# Ray casting algorithm:
# ========================================================

# def perform_ray_casting(vertices, faces, occupancy_grid, resolution, max_height):
#     mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
#     tri_mesh = Delaunay(vertices[:, :2])

#     # Iterate through each cell in the occupancy grid
#     for i in range(occupancy_grid.shape[0]):
#         for j in range(occupancy_grid.shape[1]):
#             # Compute the 3D position of the center of the cell
#             x = (i + 0.5) * resolution
#             y = (j + 0.5) * resolution
#             z = reference_plane_height + max_height

#             # Perform a ray casting algorithm
#             direction = np.array([0, 0, -1])  # Direction of the ray (Downward)
#             ray_origin = np.array([x, y, z])  # Starting point of the ray
#             intersections = mesh.ray.intersects_location(ray_origins=[ray_origin], ray_directions=[direction])

#             # Check if any intersections occurred and if the mesh is within the height range
#             if len(intersections) > 0 and len(intersections[0]) > 0 and intersections[0][0][2] <= max_height:
#                 occupancy_grid[i, j] = True  # Mark the cell as occupied

#     return occupancy_grid

def ray_casting(vertices, faces, grid_width, grid_height):
    # Create an empty occupancy grid map
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=bool)

    # Iterate over each grid cell
    for i in range(grid_height):
        for j in range(grid_width):
            # Cast a ray from the grid cell to check for intersections
            ray = np.array([j + 0.5, i + 0.5, -100])  # Extend the ray below the mesh
            intersection_count = 0

            # Iterate over each face of the mesh
            for face in faces:
                face_vertices = vertices[face]

                # Check if the ray intersects the face
                intersections = []
                for k in range(len(face_vertices)):
                    p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
                    if (p1[1] > ray[1]) != (p2[1] > ray[1]):
                        t = (ray[1] - p1[1]) / (p2[1] - p1[1])
                        intersection_x = p1[0] + t * (p2[0] - p1[0])
                        if intersection_x > ray[0]:
                            intersections.append(intersection_x)

                if len(intersections) > 0:
                    intersection_count += 1

            # Check if the number of intersections is odd (inside the mesh)
            if intersection_count % 2 != 0:
                occupancy_grid[i, j] = True

    return occupancy_grid

# grid_size = 10.0  # Size of the grid in units (e.g., meters)
# resolution = 0.1  # Resolution of each grid cell in units (e.g., meters)
# max_height =  0.8 # Maximum height for considering occupancy in units (e.g., meters) # 0.8 meters
# reference_plane_height = np.min(vertices[:, 2])
# print(reference_plane_height)
grid_width, grid_height = 5, 10

# occupancy_grid = create_occupancy_grid(grid_size, resolution)
# occupancy_grid = perform_ray_casting(vertices, faces, occupancy_grid, resolution, max_height)
occupancy_grid = ray_casting(vertices, faces, grid_width, grid_height)
print(occupancy_grid)

def visualize_occupancy_grid(occupancy_grid):
    plt.imshow(occupancy_grid.astype(int), cmap='binary')
    plt.colorbar()    
    plt.show()

visualize_occupancy_grid(occupancy_grid)