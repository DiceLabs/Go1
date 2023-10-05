# =========================================================================================================================================================

# DATA PREPROCESSING

# =========================================================================================================================================================

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
obj_file_path = './floor2.obj'
mtl_file_path = './floor2.mtl'

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

# def remove_door_from_obj_file(obj_file_path, faces_to_remove):
#     updated_lines = []

#     with open(obj_file_path, 'r') as file:
#         for line in file:
#             if line.startswith('f '):
#                 face_line = line[2:]
#                 face_data = []
#                 for face_vertex in face_line.split():
#                     vertex_indices = list(map(int, face_vertex.split('//')))
#                     face_data.append(vertex_indices)
#                 if face_data not in faces_to_remove:
#                     updated_lines.append(line)
#             else:
#                 updated_lines.append(line)

#     with open('nodoor.obj', 'w') as file:
#         file.writelines(updated_lines)

# # Provide the paths to the .obj and .mtl files
# obj_file_path = './nonpink.obj'
# mtl_file_path = './floor2.mtl'

# # Parse the .obj file
# vertices, faces, material_groups = parse_obj_file(obj_file_path)

# # Parse the .mtl file
# materials = parse_mtl_file(mtl_file_path)


# # Define the specific material properties
# specific_material_properties_floor = {
#     'Ka': '0.063 0.063 0.063',
#     'Kd': '0.315 0.315 0.315'
# }

# #  Check if the specific material properties match any materials
# matching_materials = {}
# for material_id, properties in materials.items():
#     match = True
#     for key, value in specific_material_properties_floor.items():
#         if key not in properties or properties[key] != value:
#             match = False
#             break
#     if match:
#         matching_materials[material_id] = properties

# # Get the face indices to remove
# faces_to_remove = []
# print_material_faces(matching_materials, material_groups, faces)
# for face_indices in pink_faces:
#     faces_to_remove.append(face_indices)


# remove_door_from_obj_file(obj_file_path, faces_to_remove)
# print("nodoor.obj file created with the brown_faces.")


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

# mesh4 = trimesh.load( './nodoor.obj')
# mesh4.show()

# ========================================================
# Visualize it MATPLOTLIB for  x,y,z limits:
# ========================================================

# def visualize_mesh(vertices, faces):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Plot faces
#     for face in faces:
#         polygon = Poly3DCollection([vertices[face]])
#         polygon.set_alpha(0.5)
#         ax.add_collection3d(polygon)

#     ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
#     ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
#     ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])

#     plt.show()

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






# ===============================================================================
# Calculate Camera distance from the wall and Step Distance of the Robot
# ================================================================================
# Given:
    # Desired width of the captured area: W
    # FOV angle: θ (degrees)
    # Overlap percentage: P_overlap
# ================================================================================

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

# obj_file_path = './nodoor.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# extracted_vertices, extracted_faces = detect_walls_perpendicular_to_xy(vertices, faces)

# # Save the extracted perpendicular faces as a new .obj file
# extracted_mesh = o3d.geometry.TriangleMesh()
# extracted_mesh.vertices = o3d.utility.Vector3dVector(extracted_vertices)
# extracted_mesh.triangles = o3d.utility.Vector3iVector(extracted_faces)

# o3d.io.write_triangle_mesh("perpendicular_faces.obj", extracted_mesh)

# mesh3 = trimesh.load( './perpendicular_faces.obj')
# mesh3.show()

# =========================================================================================================================================================