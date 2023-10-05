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

# Create a Pyrender scene
scene = pyrender.Scene()

# Create the mesh node and add it to the scene
mesh_node = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_node)

# Create a point light at the viewpoint
light = pyrender.PointLight(color=[1.0, 1.0, 0.0], intensity=1.0)  # Yellow light color

# Create a node to hold both the mesh and the light
node_matrix = np.eye(4)
node_matrix[:3, 3] = viewpoint
node = pyrender.Node(mesh=mesh_node, light=light, matrix=node_matrix)
scene.add_node(node)

# Create a camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

# Create a camera node and add it to the scene
camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
scene.add_node(camera_node)

# Set the camera pose
camera_pose = np.eye(4)
camera_pose[:3, 3] = viewpoint
scene.set_pose(camera_node, camera_pose)

# Create the renderer
renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)

# Render the scene
color, _ = renderer.render(scene)
# print(color.shape[1])

# ----------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Extract the X, Y, Z coordinates from the vertices
x_coords = vertices[:, 0]
y_coords = vertices[:, 1]
z_coords = vertices[:, 2]

# # Plot the mesh with colored faces based on lighting
ax.plot_trisurf(x_coords, y_coords, z_coords, triangles=faces, edgecolor='k', cmap='viridis')

# # Plot the viewpoint
ax.scatter(viewpoint[0], viewpoint[1], viewpoint[2], color='yellow', marker='o', s=100, edgecolor='k')

# # Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# # Set the viewpoint as the initial camera position
ax.view_init(elev=30, azim=-45)

# # Show the plot
plt.show()
