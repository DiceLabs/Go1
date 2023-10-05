# ================================================================================================
# Data Preprocessing:
# ================================================================================================

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

pink_faces = []

def parse_obj_file(obj_file_path):
    vertices = []
    faces = []
    material_groups = {}

    with open(obj_file_path, 'r') as file:
        current_material = None

        for line in file:
            line = line.strip()

            if line.startswith('v '):
                vertex = list(map(float, line[2:].split()))
                vertices.append(vertex)
            elif line.startswith('f '):
                face_line = line[2:]
                face_data = []
                for face_vertex in face_line.split():
                    vertex_indices = list(map(int, face_vertex.split('//')))
                    face_data.append(vertex_indices)
                faces.append(face_data)
                if current_material not in material_groups:
                    material_groups[current_material] = []
                material_groups[current_material].append(len(faces) - 1)
            elif line.startswith('usemtl '):
                current_material = line[7:]

    return vertices, faces, material_groups

def parse_mtl_file(mtl_file_path):
    materials = {}

    with open(mtl_file_path, 'r') as file:
        current_material = None

        for line in file:
            line = line.strip()

            if line.startswith('newmtl '):
                current_material = line[7:]
                materials[current_material] = {}
            elif line.startswith(('Ka ', 'Kd ', 'Ks ', 'Tr ', 'illum ', 'Ns ')):
                key, values = line.split(' ', 1)
                materials[current_material][key] = values

    return materials

def print_material_faces(material_properties, material_groups, faces):
    for material_id, properties in material_properties.items():
        print('Material ID:', material_id)
        print('Material Properties:', properties)
        if material_id in material_groups:
            face_groups = material_groups[material_id]
            for face_group in face_groups:
                face_indices = faces[face_group]
                # print(face_indices)
                # print('Faces:', face_indices)
                pink_faces.append(face_indices)

def remove_faces_from_obj_file(obj_file_path, faces_to_remove):
    updated_lines = []

    with open(obj_file_path, 'r') as file:
        for line in file:
            if line.startswith('f '):
                face_line = line[2:]
                face_data = []
                for face_vertex in face_line.split():
                    vertex_indices = list(map(int, face_vertex.split('//')))
                    face_data.append(vertex_indices)
                if face_data not in faces_to_remove:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

    with open('nonpink.obj', 'w') as file:
        file.writelines(updated_lines)

def remove_nonpink_faces_from_obj_file(obj_file_path, faces_to_remove):
    updated_lines = []

    with open(obj_file_path, 'r') as file:
        for line in file:
            if line.startswith('f '):
                face_line = line[2:]
                face_data = []
                for face_vertex in face_line.split():
                    vertex_indices = list(map(int, face_vertex.split('//')))
                    face_data.append(vertex_indices)
                if face_data in faces_to_remove:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

    with open('pink.obj', 'w') as file:
        file.writelines(updated_lines)

# Provide the paths to the .obj and .mtl files
obj_file_path = './data/floor2.obj'
mtl_file_path = './data/floor2.mtl'

# Parse the .obj file
vertices, faces, material_groups = parse_obj_file(obj_file_path)

# Parse the .mtl file
materials = parse_mtl_file(mtl_file_path)

# Define the specific material properties
specific_material_properties_pink = {
    'Ka': '0.2 0 0.100392',
    'Kd': '1 0 0.501961'
}


# Check if the specific material properties match any materials
matching_materials = {}
for material_id, properties in materials.items():
    match = True
    for key, value in specific_material_properties_pink.items():
        if key not in properties or properties[key] != value:
            match = False
            break
    if match:
        matching_materials[material_id] = properties

# Get the face indices to remove
faces_to_remove = []
print_material_faces(matching_materials, material_groups, faces)
for face_indices in pink_faces:
    faces_to_remove.append(face_indices)

# Remove the pink_faces from the obj file and save as test.obj

remove_nonpink_faces_from_obj_file(obj_file_path, faces_to_remove)
print("pink.obj file created with the pink_faces.")

remove_faces_from_obj_file(obj_file_path, faces_to_remove)
print("non_pink.obj file created without the pink_faces.")

def remove_floor_from_obj_file(obj_file_path, faces_to_remove):
    updated_lines = []

    with open(obj_file_path, 'r') as file:
        for line in file:
            if line.startswith('f '):
                face_line = line[2:]
                face_data = []
                for face_vertex in face_line.split():
                    vertex_indices = list(map(int, face_vertex.split('//')))
                    face_data.append(vertex_indices)
                if face_data not in faces_to_remove:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

    with open('nonfloor.obj', 'w') as file:
        file.writelines(updated_lines)

# Provide the paths to the .obj and .mtl files
obj_file_path = './nonpink.obj'
mtl_file_path = './data/floor2.mtl'

# Parse the .obj file
vertices, faces, material_groups = parse_obj_file(obj_file_path)

# Parse the .mtl file
materials = parse_mtl_file(mtl_file_path)


# Define the specific material properties
specific_material_properties_floor = {
    'Ka': '0.0705882 0.0486275 0.0376471',
    'Kd': '0.352941 0.243137 0.188235'
}

#  Check if the specific material properties match any materials
matching_materials = {}
for material_id, properties in materials.items():
    match = True
    for key, value in specific_material_properties_floor.items():
        if key not in properties or properties[key] != value:
            match = False
            break
    if match:
        matching_materials[material_id] = properties

# Get the face indices to remove
faces_to_remove = []
print_material_faces(matching_materials, material_groups, faces)
for face_indices in pink_faces:
    faces_to_remove.append(face_indices)


remove_floor_from_obj_file(obj_file_path, faces_to_remove)
print("nonfloor.obj file created with the brown_faces.")

# =============================================================================================

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


obj_file_path = './pink.obj'
vertices, faces = extract_mesh_info(obj_file_path)

mesh1 = trimesh.load( './floor2.obj')
mesh1.show()

mesh2 = trimesh.load( './nonpink.obj')
mesh2.show()

mesh3 = trimesh.load( './pink.obj')
mesh3.show()

mesh3 = trimesh.load( './nonfloor.obj')
mesh3.show()

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

# ===============================================================================
# MODIFY THE CODE BELOW:
# ================================================================================

def load_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = line[2:].strip().split(' ')
                vertices.append((float(vertex[0]), float(vertex[1]), float(vertex[2])))
            elif line.startswith('f '):
                face = line[2:].strip().split(' ')
                faces.append([int(idx.split('/')[0]) - 1 for idx in face])

    return vertices, faces

def create_occupancy_grid(vertices, faces, resolution):
    # Calculate the grid size based on the resolution
    x_min = min(v[0] for v in vertices)
    y_min = min(v[1] for v in vertices)
    x_max = max(v[0] for v in vertices)
    y_max = max(v[1] for v in vertices)

    grid_width = int(np.ceil((x_max - x_min) / resolution))
    grid_height = int(np.ceil((y_max - y_min) / resolution))

    # Create an empty occupancy grid
    occupancy_grid = np.zeros((grid_height, grid_width))

    # Mark occupied cells for each face
    for face in faces:
        face_vertices = [vertices[i][:2] for i in face]
        x_coords = [v[0] for v in face_vertices]
        y_coords = [v[1] for v in face_vertices]
        min_x = min(x_coords) -0.5
        max_x = max(x_coords) +0.5
        min_y = min(y_coords) -0.5
        max_y = max(y_coords) +0.5

        # Calculate the cell indices covered by the face
        for x in np.arange(min_x, max_x, resolution):
            for y in np.arange(min_y, max_y, resolution):
                if point_in_polygon(x, y, face_vertices):
                    cell_x = int((x - x_min) / resolution)
                    cell_y = int((y - y_min) / resolution)
                    occupancy_grid[cell_y, cell_x] = 1

    return occupancy_grid, (x_min, y_min), resolution

def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def plot_occupancy_grid(occupancy_grid, origin, resolution):
    fig, ax = plt.subplots()

    # Plot the occupancy grid
    ax.imshow(occupancy_grid, cmap='Greys', origin='lower')

    # Set the ticks and tick labels
    x_ticks = np.arange(0, occupancy_grid.shape[1] + 1, 1)
    y_ticks = np.arange(0, occupancy_grid.shape[0] + 1, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Round off the tick labels to one decimal place
    x_ticklabels = np.arange(origin[0], origin[0] + (occupancy_grid.shape[1] + 1) * resolution, resolution)
    x_ticklabels = [round(x, 1) for x in x_ticklabels[:occupancy_grid.shape[1]+1]]
    y_ticklabels = np.arange(origin[1], origin[1] + (occupancy_grid.shape[0] + 1) * resolution, resolution)
    y_ticklabels = [round(y, 1) for y in y_ticklabels[:occupancy_grid.shape[0]+1]]

    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.grid(True)
    plt.show()

# Define the file path for the OBJ file
file_path = 'nonfloor.obj'

# Load the OBJ file
vertices, faces = load_obj(file_path)

# Set the resolution of the occupancy grid
resolution = 0.5

# Create the occupancy grid
occupancy_grid, origin, resolution = create_occupancy_grid(vertices, faces, resolution)

# Plot the occupancy grid
plot_occupancy_grid(occupancy_grid, origin, resolution)



# start = time.time()

# def ray_casting(vertices, faces, grid_width, grid_height, cell_size_x, cell_size_y):
#     # Calculate the number of cells in each dimension
#     num_cells_x = int(grid_width / cell_size_x)
#     num_cells_y = int(grid_height / cell_size_y)

#     # Create an empty occupancy grid map
#     occupancy_grid = np.zeros((num_cells_y, num_cells_x), dtype=bool)

#     # Iterate over each grid cell
#     for i in range(num_cells_y):
#         for j in range(num_cells_x):
#             # Calculate the ray origin for the current grid cell
#             ray_origin_x = j * cell_size_x + cell_size_x / 2
#             ray_origin_y = i * cell_size_y + cell_size_y / 2
#             ray_origin = np.array([ray_origin_x, ray_origin_y, -100])  # Extend the ray below the mesh

#             intersection_count = 0

#             # Iterate over each face of the mesh
#             for face in faces:
#                 face_vertices = vertices[face]

#                 # Check if the ray intersects the face
#                 intersections = []
#                 for k in range(len(face_vertices)):
#                     p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
#                     if np.logical_xor(p1[1] > ray_origin[1], p2[1] > ray_origin[1]):
#                         t = (ray_origin[1] - p1[1]) / (p2[1] - p1[1])
#                         intersection_x = p1[0] + t * (p2[0] - p1[0])
#                         if intersection_x > ray_origin[0]:
#                             intersections.append(intersection_x)

#                 if len(intersections) > 0:
#                     intersection_count += 1

#             # Check if the number of intersections is odd (inside the mesh)
#             if intersection_count % 2 != 0:
#                 occupancy_grid[i, j] = True

#     return occupancy_grid

# grid_width, grid_height = max(vertices[:, 0]), max(vertices[:, 1])
# cell_size_x = 0.2
# cell_size_y = 0.2

# occupancy_grid = ray_casting(vertices, faces, grid_width, grid_height, cell_size_x, cell_size_y)
# # print(occupancy_grid)
# end = time.time()
# print(end-start)

# # def visualize_occupancy_grid(occupancy_grid):
# #     plt.imshow(occupancy_grid.astype(int), cmap='binary', origin='lower')

# #     # Add grid lines
# #     plt.grid(True, color='black', linewidth=0.5)

# #     # # Set the aspect ratio
# #     # plt.gca().set_aspect('equal', adjustable='box')

# #     # # Set the axis labels
# #     # plt.xlabel('X')
# #     # plt.ylabel('Y')

# #     # Show the plot
# #     plt.show()

# def visualize_occupancy_grid(occupancy_grid, xmin= min(vertices[:, 0]), xmax = max(vertices[:, 0]), ymin= min(vertices[:, 1]), ymax = max(vertices[:, 1])):
#     plt.imshow(occupancy_grid.astype(int), cmap='binary', extent=[xmin, xmax, ymin, ymax])
#     plt.colorbar()
#     plt.show()

# visualize_occupancy_grid(occupancy_grid)

# ===============================================================================
# Calculate Camera distance from the wall and Step Distance of the Robot
# ================================================================================
# Given:
    # Desired width of the captured area: W
    # FOV angle: θ (degrees)
    # Overlap percentage: P_overlap
# ================================================================================

def calculate_camera_distance(width_captured_area, fov_angle):
    # Convert the FOV angle from degrees to radians
    fov_angle_rad = math.radians(fov_angle)

    # Calculate the camera distance using the formula
    camera_distance = (width_captured_area / 2) / math.tan(fov_angle_rad / 2)
    # camera_distance = math.floor(camera_distance * 10) / 10

    return camera_distance

def calculate_horizontal_distance(camera_distance, width_captured_area, fov_angle, overlap_percentage):
    # Convert the FOV angle from degrees to radians
    fov_angle_rad = math.radians(fov_angle)

    # Calculate the horizontal distance using the formula
    horizontal_distance = (width_captured_area / 2) / math.tan(fov_angle_rad / 2) * (1 - overlap_percentage)
    # horizontal_distance = math.floor(horizontal_distance * 10) / 10

    return horizontal_distance

# Example usage
width_captured_area = 0.5  # Width of the captured area (in unit m)
fov_angle = 60.0  # Field of view angle of the camera (in degrees)
overlap_percentage = 0.4  # Overlap percentage between images(between 0 and 1)

camera_distance = calculate_camera_distance(width_captured_area, fov_angle)
print("Camera Distance:", camera_distance)

horizontal_distance = calculate_horizontal_distance(camera_distance, width_captured_area, fov_angle, overlap_percentage)
print("Horizontal Distance:", horizontal_distance)