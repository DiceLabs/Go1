import open3d as o3d
import numpy as np

# Load the .obj file
mesh = o3d.io.read_triangle_mesh("/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor2.obj")

# Assign a uniform color to the mesh
color = [0, 0, 0.2]  # RGB values (range: [0, 1])
mesh.paint_uniform_color(color)

# Visualize the mesh with the assigned color
o3d.visualization.draw_geometries([mesh])
print(mesh)


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def read_obj_file(file_path):
#     vertices = []
#     faces = []

#     with open(file_path, 'r') as obj_file:
#         for line in obj_file:
#             line = line.strip()
#             if line.startswith('v '):
#                 elements = line.split()
#                 vertex = [float(elements[1]), float(elements[2]), float(elements[3])]
#                 vertices.append(vertex)
#             elif line.startswith('f '):
#                 elements = line.split()
#                 face = [int(elem.split('/')[0]) for elem in elements[1:]]
#                 faces.append(face)

#     return np.array(vertices), np.array(faces)

# # Example usage
# obj_file_path = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor7.obj'
# vertices, faces = read_obj_file(obj_file_path)

# # Plot the 3D mesh
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot vertices
# ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r')

# # Plot faces
# for face in faces:
#     ax.plot(vertices[face, 0], vertices[face, 1], vertices[face, 2], 'b')

# # Set axis labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Mesh')

# # Show the plot
# plt.show()
