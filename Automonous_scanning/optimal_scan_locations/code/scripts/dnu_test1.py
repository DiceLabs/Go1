# # import trimesh
# # import numpy as np
# # import matplotlib.pyplot as plt

# # r = trimesh.load('data/floor2.obj', force='mesh')
# # r.show()
# # print(r.visual)
# # print(r.visual.kind)
# # print(r.visual.material)

# # # Access the texture information
# # visuals = r.visual
# # texture = visuals.material

# # texture_type = type(texture)
# # print("Texture type:", texture_type)

# # # Convert texture values to a NumPy array
# # texture_array = np.asarray(texture.image)

# # # Extract color channels
# # red_channel = texture_array[:, :, 0]
# # green_channel = texture_array[:, :, 1]
# # blue_channel = texture_array[:, :, 2]

# # print(red_channel)

# # # Combine color channels to get RGB color values
# # rgb_colors = np.stack((red_channel, green_channel, blue_channel), axis=2)

# # # Print the shape of the RGB color array
# # print("RGB color array shape:", rgb_colors)

# # plt.imshow(texture_array)
# # plt.axis('off')
# # plt.show()

# # =================

# # mesh_name = list(r.geometry.keys())[0]  # Get the name of the first mesh in the scene
# # visuals = r.geometry[mesh_name].visual
# # print(visuals)

# # m = trimesh.load('data/floor2.obj')
# # print(m.visual)

# # m.visual = m.visual.to_color()
# # print(m.visual.kind)
# # print(m.visual.vertex_colors)
# # print(m.visual.face_colors)

# # =================

# # import trimesh
# # import numpy as np
# # import matplotlib.pyplot as plt


# # r = trimesh.load('data/floor2.obj', force='mesh')
# # # r.show()
# # print(r.visual)
# # print(r.visual.kind)
# # print(r.visual.material)

# # # Access the texture information
# # visuals = r.visual
# # texture = visuals.material

# # # Convert texture values to an array
# # texture_array = np.asarray(texture.image)

# # print("Texture array:")
# # print(texture_array)

# # plt.imshow(texture_array)
# # plt.axis('off')
# # plt.show()

# # ========================================================

# import numpy as np
# import pyrender
# import trimesh
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import time
# from scipy.spatial import Delaunay
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# # ========================================================
# # Parse the .obj file:
# # ========================================================

# # def extract_mesh_info(file_path):
# #     vertices = []
# #     faces = []

# #     with open(file_path, 'r') as obj_file:
# #         for line in obj_file:
# #             line = line.strip()
# #             if line.startswith(('v ', 'f ')):
# #                 elements = line.split()
# #                 if line.startswith('v '):
# #                     vertices.append([float(elements[1]), float(elements[2]), float(elements[3])])
# #                 elif line.startswith('f '):
# #                     face_vertices = []
# #                     for element in elements[1:]:
# #                         vertex_indices = element.split('/')
# #                         face_vertices.append(int(vertex_indices[0]) - 1)
# #                     faces.append(face_vertices)

# #     return np.array(vertices), np.array(faces)

# # obj_file_path = './data/floor2.obj'
# # vertices, faces = extract_mesh_info(obj_file_path)

# def extract_mesh_info(file_path, mtl_file_path):
#     vertices = []
#     faces = []

#     with open(file_path, 'r') as obj_file:
#         current_material = None
#         for line in obj_file:
#             line = line.strip()
#             if line.startswith('mtllib'):
#                 mtl_file_path = line.split()[1]
#             elif line.startswith('usemtl'):
#                 current_material = line.split()[1]
#             elif line.startswith(('v ', 'f ')):
#                 elements = line.split()
#                 if line.startswith('v '):
#                     vertices.append([float(elements[1]), float(elements[2]), float(elements[3])])
#                 elif line.startswith('f '):
#                     if current_material is not None and is_material_to_remove(current_material, mtl_file_path):
#                         continue
#                     face_vertices = []
#                     for element in elements[1:]:
#                         vertex_indices = element.split('/')
#                         face_vertices.append(int(vertex_indices[0]) - 1)
#                     faces.append(face_vertices)

#     return np.array(vertices), np.array(faces)


# def is_material_to_remove(material_name, mtl_file_path):
#     with open(mtl_file_path, 'r') as mtl_file:
#         for line in mtl_file:
#             if line.startswith('newmtl ' + material_name):
#                 while line.strip() != '':
#                     if line.startswith(('Ka', 'Kd', 'Ks', 'Tr', 'illum', 'Ns')):
#                         return True
#                     line = next(mtl_file)
#                 break
#     return False


# obj_file_path = './data/floor2.obj'
# mtl_file_path = './data/floor2.mtl'
# material_properties = {
#     'Ka': [0.2, 0, 0.100392],
#     'Kd': [1, 0, 0.501961],
#     'Ks': [0, 0, 0],
#     'Tr': 0.8,
#     'illum': 1,
#     'Ns': 50
# }

# # Load the .mtl file and extract the material properties
# with open(mtl_file_path, 'r') as mtl_file:
#     for line in mtl_file:
#         line = line.strip()
#         if line.startswith('newmtl'):
#             current_material = line.split()[1]
#         elif line.startswith(('Ka', 'Kd', 'Ks', 'Tr', 'illum', 'Ns')):
#             elements = line.split()
#             property_name = elements[0]
#             property_values = [float(value) for value in elements[1:]]
#             if current_material in material_properties and material_properties[current_material] == property_values:
#                 material_properties[current_material] = property_name

# # Extract the vertices and faces based on the specified material properties
# vertices, faces, texture_coords = extract_mesh_info(obj_file_path, material_properties)

# mesh = trimesh.load(obj_file_path)
# mesh.show()

# print(len(vertices), len(faces))

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
                print('Faces:', face_indices)
        print()

# Provide the paths to the .obj and .mtl files
obj_file_path = './data/floor2.obj'
mtl_file_path = './data/floor2.mtl'

# Parse the .obj file
vertices, faces, material_groups = parse_obj_file(obj_file_path)

# Parse the .mtl file
materials = parse_mtl_file(mtl_file_path)

# Define the specific material properties
specific_material_properties = {
    'Ka': '0.2 0 0.100392',
    'Kd': '1 0 0.501961'
}

# Check if the specific material properties match any materials
matching_materials = {}
for material_id, properties in materials.items():
    match = True
    for key, value in specific_material_properties.items():
        if key not in properties or properties[key] != value:
            match = False
            break
    if match:
        matching_materials[material_id] = properties

# Print the matching material ID, properties, and faces
print_material_faces(matching_materials, material_groups, faces)
