import numpy as np

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

def remove_floor_faces(obj_file_path):
    vertices, faces, material_groups = parse_obj_file(obj_file_path)
    materials = parse_mtl_file('./data/floor2.mtl')

    specific_material_properties_floor = {
        'Ka': '0.0705882 0.0486275 0.0376471',
        'Kd': '0.352941 0.243137 0.188235'
    }

    matching_materials_floor = {}
    for material_id, properties in materials.items():
        match = True
        for key, value in specific_material_properties_floor.items():
            if key not in properties or properties[key] != value:
                match = False
                break
        if match:
            matching_materials_floor[material_id] = properties

    floor_faces = []
    for material_id, _ in matching_materials_floor.items():
        if material_id in material_groups:
            floor_faces.extend(material_groups[material_id])

    updated_lines = []
    with open(obj_file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('f ') and i in floor_faces:
                continue
            updated_lines.append(line)

    with open('nonpink.obj', 'w') as file:
        file.writelines(updated_lines)

# Provide the path to the .obj file
obj_file_path = './data/floor2.obj'

# Define the specific material properties for pink faces
specific_material_properties_pink = {
    'Ka': '0.2 0 0.100392',
    'Kd': '1 0 0.501961'
}

# Parse the .obj file
vertices, faces, material_groups = parse_obj_file(obj_file_path)

# Parse the .mtl file
materials = parse_mtl_file('./data/floor2.mtl')

# Check if the specific material properties match any materials for pink faces
matching_materials_pink = {}
for material_id, properties in materials.items():
    match = True
    for key, value in specific_material_properties_pink.items():
        if key not in properties or properties[key] != value:
            match = False
            break
    if match:
        matching_materials_pink[material_id] = properties

# Get the face indices to remove for pink faces
pink_faces = []
print_material_faces(matching_materials_pink, material_groups, faces)
for face_indices in pink_faces:
    face_indices = np.array(face_indices)
    pink_faces.append(face_indices[:, :-1] - 1)

# Remove the pink faces from the .obj file and save as nonpink.obj
remove_faces_from_obj_file(obj_file_path, pink_faces)
print("nonpink.obj file created without the pink faces.")

# Remove the floor faces from the .obj file and save as nonpink.obj
remove_floor_faces('nonpink.obj')
print("nonpink.obj file created without the floor faces.")
