import pywavefront
import numpy as np
import trimesh
import matplotlib.pyplot as plt

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

viewpoint = np.array([0, 175, 0])

# Create a visualization node for the viewpoint
viewpoint_sphere = trimesh.creation.icosphere(radius=5.0)
viewpoint_sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
viewpoint_sphere.apply_translation(viewpoint)

# Create a plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='grey', alpha=0.3)
ax.plot_trisurf(viewpoint_sphere.vertices[:, 0], viewpoint_sphere.vertices[:, 1], viewpoint_sphere.vertices[:, 2], triangles=viewpoint_sphere.faces, color='red', alpha=0.8)

# Set plot limits
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_zlim(-200, 200)

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
