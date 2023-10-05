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

viewpoint = np.array([0, 125, 175])

cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0],
    [1.0, 0.0,           0.0,           125],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 175],
    [0.0,  0.0,           0.0,          1.0]
])

# Create the light source at the viewpoint
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1000.0)

# Create the scene and add the mesh and light node
scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
mesh_node = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_node)
point_l_node = scene.add(light, pose=cam_pose)

# Create the offscreen renderer
renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

# Render the scene
color, _ = renderer.render(scene)

# Display the rendered image using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(color)
plt.axis('off')
plt.show()
