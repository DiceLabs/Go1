# import pywavefront
# import numpy as np
# import pyrender
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

# # Example usage
# obj_file_path = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor2.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# mesh = trimesh.load(obj_file_path)
# mesh.show()

# print("No.of vertices:", len(vertices))
# print("No.of faces:", len(faces))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles=faces, Z=vertices[:, 2])

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()


# ======================
# TEST CODE 1
# ======================

# import pywavefront
# import numpy as np
# import pyrender
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

# # Example usage
# obj_file_path = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor2.obj'
# vertices, faces = extract_mesh_info(obj_file_path)
# mesh = trimesh.load(obj_file_path)

# mesh.show()

# m_x = min(vertices[:, 0])
# m_y = min(vertices[:, 1])
# m_z = min(vertices[:, 2])

# for count in range(len(vertices)):
#     vertices[count][0] -= m_x
#     vertices[count][1] -= m_y
#     vertices[count][2] -= m_z
    
# print("No.of vertices:", len(vertices))
# print("No.of faces:", len(faces))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# viewpoint = (1, 1 , 1)
# ax.scatter([viewpoint[0]], [viewpoint[1]], [viewpoint[2]], color='red', label='Viewpoint')

# ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles=faces, Z=vertices[:, 2])

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

# =========================

import pywavefront
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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

# Example usage
obj_file_path = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor2_new.obj'
vertices, faces = extract_mesh_info(obj_file_path)
mesh = trimesh.load(obj_file_path)

mesh.show()

m_x = min(vertices[:, 0])
m_y = min(vertices[:, 1])
m_z = min(vertices[:, 2])

for count in range(len(vertices)):
    vertices[count][0] -= m_x
    vertices[count][1] -= m_y
    vertices[count][2] -= m_z
    
print("No.of vertices:", len(vertices))
print("No.of faces:", len(faces))

# Define the rectangular bounds
x_min, y_min, z_min = min(vertices[:, 0]), min(vertices[:, 1]), min(vertices[:, 2])
x_max, y_max, z_max = max(vertices[:, 0]), max(vertices[:, 1]), max(vertices[:, 2])

vertex_r = []
delta = 0.01
for count in range(len(vertices)):
    if (x_min< vertices[count][0] < x_min+delta or x_max-delta< vertices[count][0] < x_max or y_min< vertices[count][1] < y_min+delta or y_max-delta< vertices[count][1] < y_max or z_min< vertices[count][2] < z_min+delta or z_max-delta< vertices[count][2] < z_max ):
        vertex_r.append(vertices[count])

print(len(vertex_r))
i = 1

start_time = time.time()
for v in vertex_r:
    print(i)
    index_to_remove = np.where((vertices == v).all(axis=1))[0][0]
    vertices = np.delete(vertices, index_to_remove, axis=0)
    faces = [tri for tri in faces if index_to_remove not in tri]
    i+=1

print("No.of vertices:", len(vertices))
print("No.of faces:", len(faces))

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))

mesh1 = trimesh.load(obj_file_path)
mesh1.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# viewpoint = (1, 1 , 1)
# ax.scatter([viewpoint[0]], [viewpoint[1]], [viewpoint[2]], color='red', label='Viewpoint')

# ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles=faces, Z=vertices[:, 2])

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()
