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
obj_file_path = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/code/data/Wolf.obj'
vertices, faces = extract_mesh_info(obj_file_path)
mesh = trimesh.load(obj_file_path)

viewpoint = np.array([0, 275, 0])

# Calculate the aspect ratio manually
cam_height = 1080  # Replace with the desired height resolution
cam_width = int(cam_height * 16 / 9)  # Calculate the width based on the desired aspect ratio
aspect_ratio = cam_width / cam_height

cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.eye(4)
cam_pose[:3, 3] = viewpoint

# Create the light source at the viewpoint
light = pyrender.PointLight(color=np.ones(3), intensity=100000.0, range=1000)

# Create the scene and add the mesh and light node
scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
mesh_node = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_node)
point_l_node = scene.add(light, pose=cam_pose)

# Create the viewer with shadows enabled
viewer = pyrender.Viewer(scene, shadows=True)

# Calculate the number of meshes that are not illuminated
num_unilluminated_meshes = 0

# Calculate the rays from the camera
projection_matrix = cam.get_projection_matrix(width=cam_width, height=cam_height)
ray_origins = np.zeros((cam_height, cam_width, 3))
ray_directions = np.zeros((cam_height, cam_width, 3))

for i in range(cam_height):
    for j in range(cam_width):
        ray_origin = np.linalg.inv(cam_pose) @ np.array([j, i, 0.0, 1.0])
        ray_origin /= ray_origin[3]
        ray_direction = np.linalg.inv(projection_matrix) @ np.array([j, i, 1.0, 1.0])
        ray_direction /= np.linalg.norm(ray_direction)
        ray_origins[i, j] = ray_origin[:3]
        ray_directions[i, j] = ray_direction[:3]

for i in range(cam_height):
    for j in range(cam_width):
        ray_origin = ray_origins[i, j]
        ray_direction = ray_directions[i, j]
        ray_intersections = mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction],
        )
        if len(ray_intersections) == 0:
            num_unilluminated_meshes += 1

print("Number of meshes not illuminated:", num_unilluminated_meshes)
