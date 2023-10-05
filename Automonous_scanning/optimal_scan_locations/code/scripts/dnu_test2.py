import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# pink_faces = []

# def parse_obj_file(obj_file_path):
#     vertices = []
#     faces = []
#     material_groups = {}

#     with open(obj_file_path, 'r') as file:
#         current_material = None

#         for line in file:
#             line = line.strip()

#             if line.startswith('v '):
#                 vertex = list(map(float, line[2:].split()))
#                 vertices.append(vertex)
#             elif line.startswith('f '):
#                 face_line = line[2:]
#                 face_data = []
#                 for face_vertex in face_line.split():
#                     vertex_indices = list(map(int, face_vertex.split('//')))
#                     face_data.append(vertex_indices)
#                 faces.append(face_data)
#                 if current_material not in material_groups:
#                     material_groups[current_material] = []
#                 material_groups[current_material].append(len(faces) - 1)
#             elif line.startswith('usemtl '):
#                 current_material = line[7:]

#     return vertices, faces, material_groups

# def parse_mtl_file(mtl_file_path):
#     materials = {}

#     with open(mtl_file_path, 'r') as file:
#         current_material = None

#         for line in file:
#             line = line.strip()

#             if line.startswith('newmtl '):
#                 current_material = line[7:]
#                 materials[current_material] = {}
#             elif line.startswith(('Ka ', 'Kd ', 'Ks ', 'Tr ', 'illum ', 'Ns ')):
#                 key, values = line.split(' ', 1)
#                 materials[current_material][key] = values

#     return materials

# def print_material_faces(material_properties, material_groups, faces):
#     for material_id, properties in material_properties.items():
#         print('Material ID:', material_id)
#         print('Material Properties:', properties)
#         if material_id in material_groups:
#             face_groups = material_groups[material_id]
#             for face_group in face_groups:
#                 face_indices = faces[face_group]
#                 pink_faces.append(face_indices)

# def remove_faces_from_obj_file(obj_file_path, faces_to_remove):
#     updated_lines = []

#     with open(obj_file_path, 'r') as file:
#         for line in file:
#             if line.startswith('f '):
#                 face_line = line[2:]
#                 face_data = []
#                 for face_vertex in face_line.split():
#                     vertex_indices = list(map(int, face_vertex.split('//')))
#                     face_data.append(vertex_indices)
#                 if face_data in faces_to_remove:
#                     updated_lines.append(line)
#             else:
#                 updated_lines.append(line)

#     with open('pink.obj', 'w') as file:
#         file.writelines(updated_lines)

# def remove_materials_from_mtl_file(mtl_file_path, materials_to_remove):
#     updated_lines = []

#     with open(mtl_file_path, 'r') as file:
#         current_material = None

#         for line in file:
#             if line.startswith('newmtl '):
#                 updated_lines.append('\n')  # Add a new line before each "newmtl"
#                 current_material = line[7:]
#                 if current_material not in materials_to_remove:
#                     updated_lines.append(line)
#             elif line.startswith(('Ka ', 'Kd ', 'Ks ', 'Tr ', 'illum ', 'Ns ')):
#                 if current_material not in materials_to_remove:
#                     updated_lines.append(line)

#     with open('pink.mtl', 'w') as file:
#         file.writelines(updated_lines)


# # Provide the paths to the .obj and .mtl files
# obj_file_path = './data/floor2.obj'
# mtl_file_path = './data/floor2.mtl'

# # Parse the .obj file
# vertices, faces, material_groups = parse_obj_file(obj_file_path)

# # Parse the .mtl file
# materials = parse_mtl_file(mtl_file_path)

# # Define the specific material properties
# specific_material_properties = {
#     'Ka': '0.2 0 0.100392',
#     'Kd': '1 0 0.501961'
# }

# # Check if the specific material properties match any materials
# matching_materials = {}
# for material_id, properties in materials.items():
#     match = True
#     for key, value in specific_material_properties.items():
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

# # Remove the non-pink faces from the obj file and save as pink.obj
# remove_faces_from_obj_file(obj_file_path, faces_to_remove)
# print("pink.obj file created with only the pink faces.")

# # Remove the matching_materials from the mtl file and save as pink.mtl
# materials_to_remove = matching_materials.keys()
# remove_materials_from_mtl_file(mtl_file_path, materials_to_remove)
# print("pink.mtl file created without the matching_materials.")

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


obj_file_path = './test.obj'
vertices, faces = extract_mesh_info(obj_file_path)

mesh1 = trimesh.load( './data/floor2.obj')
mesh1.show()

mesh = trimesh.load(obj_file_path)
mesh.show()
