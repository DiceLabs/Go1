import pywavefront
import numpy as np
import pyrender
import trimesh
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
obj_file_path = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/Wolf.obj'
vertices, faces = extract_mesh_info(obj_file_path)
mesh = trimesh.load(obj_file_path)

viewpoint = np.array([0, 275, 0])

cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.eye(4)
cam_pose[:3, 3] = viewpoint
# cam_pose = np.array([
#     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0],
#     [1.0, 0.0,           0.0,           125],
#     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 175],
#     [0.0,  0.0,           0.0,          1.0]
# ])

# Create the light source at the viewpoint
light = pyrender.PointLight(color=np.ones(3), intensity=100000.0, range = 1000)

# Create the scene and add the mesh and light node
scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
mesh_node = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_node)
point_l_node = scene.add(light, pose=cam_pose)

# Create the viewer with shadows enabled
viewer = pyrender.Viewer(scene, shadows=True)
