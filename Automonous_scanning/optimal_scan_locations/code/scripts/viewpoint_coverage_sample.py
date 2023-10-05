# ====================================================
# Import Necessary Modules/Packages
# =====================================================
import pywavefront
import numpy as np
import pyrender
import trimesh
import math
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

# Example usage
obj_file_path = './data/Wolf.obj'
vertices, faces = extract_mesh_info(obj_file_path)
mesh = trimesh.load(obj_file_path)
mesh.show()

# ====================================================
# Visualize using MatplotLib
# =====================================================

# print("Vertices:")
# for i, vertex in enumerate(vertices):
#     print(f"Vertex {i+1}: {vertex}")

# print("\nFaces:")
# print(faces)
# print(len(faces))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# viewpoint = (0, 200, 200)
# ax.scatter([viewpoint[0]], [viewpoint[1]], [viewpoint[2]], color='black', label='Point (1, 1, 1)')

# ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, Z=mesh.vertices[:, 2])

# # Set coordinate axis spacing
# ax.set_xticks(np.arange(viewpoint[0] - 100, viewpoint[0] + 101, 25))
# ax.set_yticks(np.arange(viewpoint[1] - 100, viewpoint[1] + 101, 25))
# ax.set_zticks(np.arange(viewpoint[2] - 100, viewpoint[2] + 101, 25))

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

# =============================================================================
# Defining viewpoint, adding light source, can_pose etc for Viewpoint Coverage
# ==============================================================================

# viewpoint = np.array([0, 275, 0])
viewpoint = np.array([0, 175, 175])

cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.eye(4)
cam_pose[:3, 3] = viewpoint
# cam_pose = np.array([
#     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0],
#     [1.0, 0.0,           0.0,           275],
#     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0],
#     [0.0,  0.0,           0.0,          1.0]
# ])

# Create the light source at the viewpoint
light = pyrender.PointLight(color=np.array([1,1,0]), intensity=10000000.0, range = 1000)
print(light)

# Create the scene and add the mesh and light node
scene = pyrender.Scene(ambient_light=np.array([0,0,0, 1.0]))
mesh_node = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_node)
point_l_node = scene.add(light, pose=cam_pose)

# Create the viewer with shadows enabled
viewer = pyrender.Viewer(scene, shadows=True)

# Calculate face normals
face_normals = mesh.face_normals

# ====================================================
# Algorithm
# =====================================================
# viewpoint coverage algo:
# 1. Calculate the distance of each face with the viewpoint, sort them in the list in the order of ascending distance, shortest first.
# 2. calculate the dot product of the vectors
# 3. if less than 90 degrees: check if the distance of the meshes present in the angle list with the normal from the viewpoint to the existing mesh id less than 1 unit. if yes, exclude it, else include it.
# ====================================================


# ====================================================
# Without Occlusion
# =====================================================
# # Calculate angles between viewpoint and front-facing face normals (Angle range = 0 to 180 degrees)
# angles = []
# for face_normal in face_normals:
#     dot_product = np.dot(viewpoint, face_normal)
#     if dot_product >= 0:  # Check if the face is front-facing (dot product is non-negative) # Backface culling
#         angle = math.degrees(math.acos(dot_product / (np.linalg.norm(viewpoint) * np.linalg.norm(face_normal))))
#         if (angle<90):
#             angles.append(angle)

# ====================================================
# With Occlusion
# =====================================================

# Calculate distances of each face from the viewpoint and sort them in ascending order
distances = [np.linalg.norm(vertices[face[0]] - viewpoint) for face in faces]
sorted_indices = np.argsort(distances)

# Calculate angles between viewpoint and face normals (Angle range = 0 to 180 degrees)
angles = []
for i in sorted_indices:
    face_normal = face_normals[i]
    occluded = False
    angle = math.degrees(math.acos(np.dot(viewpoint, face_normal) / (np.linalg.norm(viewpoint) * np.linalg.norm(face_normal))))
    
    if len(angles) == 0:      
        if angle < 90:
            angles.append(i)
    else:
        for j in angles:
            angle1 = math.degrees(math.acos(np.dot(face_normals[j], face_normal) / (np.linalg.norm(face_normals[j]) * np.linalg.norm(face_normal))))
            if angle < 90 and angle1 < 10:
                distance = np.linalg.norm(vertices[faces[j][0]] - viewpoint)
                if distance > 1.0:
                    occluded = True
                    break
        if not occluded:
            angle = math.degrees(math.acos(np.dot(viewpoint, face_normal) / (np.linalg.norm(viewpoint) * np.linalg.norm(face_normal))))
            if angle < 90:
                angles.append(i)

# =====================================================
# Print angles
# =====================================================
# for i, angle in enumerate(angles):
#     print(f"Angle between viewpoint and face normal {i+1}: {angle} degrees")

print("No of visible meshes: ", len(angles))
print("Total number of faces: ", len(faces))
print("Viewpoint Coverage: ", len(angles)/len(faces))
