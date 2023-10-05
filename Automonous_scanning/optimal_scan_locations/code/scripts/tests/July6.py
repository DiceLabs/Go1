import pywavefront
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ========================================================
# Parse the .obj file:
# ========================================================

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

obj_file_path = './floor2.obj'
vertices, faces = extract_mesh_info(obj_file_path)
m_x = min(vertices[:, 0])
m_y = min(vertices[:, 1])
m_z = min(vertices[:, 2])

for count in range(len(vertices)):
    vertices[count][0] -= m_x
    vertices[count][1] -= m_y
    vertices[count][2] -= m_z

# print(min(vertices[:, 0]))
# print(max(vertices[:, 0]))

mesh = trimesh.load(obj_file_path)

mesh.show()

def visualize_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot faces
    for face in faces:
        polygon = Poly3DCollection([vertices[face]])
        polygon.set_alpha(0.5)
        ax.add_collection3d(polygon)

    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])

    plt.show()

# visualize_mesh(vertices, faces)

# ========================================================
# Check if the faces in a mesh are already triangulated:
# ========================================================

def check_triangulation(faces):
    is_triangulated = all(len(face) == 3 for face in faces)
    return is_triangulated

is_triangulated = check_triangulation(faces)
print("Are faces Triangulated:", is_triangulated)

# ========================================================
# Ray casting algorithm:
# ========================================================

# def ray_casting(vertices, faces, grid_width, grid_height):
#     # Create an empty occupancy grid map
#     occupancy_grid = np.zeros((grid_height, grid_width), dtype=bool)

#     # Iterate over each grid cell
#     for i in range(grid_height):
#         for j in range(grid_width):
#             # Cast a ray from the grid cell to check for intersections
#             ray = np.array([j + 0.5, i + 0.5, -100])  # Extend the ray below the mesh
#             intersection_count = 0

#             # Iterate over each face of the mesh
#             for face in faces:
#                 face_vertices = vertices[face]

#                 # Check if the ray intersects the face
#                 intersections = []
#                 for k in range(len(face_vertices)):
#                     p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
#                     if (p1[1] > ray[1]) != (p2[1] > ray[1]):
#                         t = (ray[1] - p1[1]) / (p2[1] - p1[1])
#                         intersection_x = p1[0] + t * (p2[0] - p1[0])
#                         if intersection_x > ray[0]:
#                             intersections.append(intersection_x)

#                 if len(intersections) > 0:
#                     intersection_count += 1

#             # Check if the number of intersections is odd (inside the mesh)
#             if intersection_count % 2 != 0:
#                 occupancy_grid[i, j] = True

#     return occupancy_grid

def ray_casting(vertices, faces, grid_width, grid_height):
    # Calculate the cell size based on the grid dimensions
    cell_size_x = 1
    cell_size_y = 1

    # Calculate the number of cells in each dimension
    num_cells_x = int(grid_width / cell_size_x)
    num_cells_y = int(grid_height / cell_size_y)

    # Create an empty occupancy grid map
    occupancy_grid = np.zeros((num_cells_y, num_cells_x), dtype=bool)

    # Iterate over each grid cell
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Calculate the ray origin for the current grid cell
            ray_origin_x = j * cell_size_x + cell_size_x / 2
            ray_origin_y = i * cell_size_y + cell_size_y / 2
            ray_origin = np.array([ray_origin_x, ray_origin_y, -10])  # Extend the ray below the mesh

            intersection_count = 0

            # Iterate over each face of the mesh
            for face in faces:
                face_vertices = vertices[face]

                # Check if the ray intersects the face
                intersections = []
                for k in range(len(face_vertices)):
                    p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
                    if (p1[1] > ray_origin[1]) != (p2[1] > ray_origin[1]):
                        t = (ray_origin[1] - p1[1]) / (p2[1] - p1[1])
                        intersection_x = p1[0] + t * (p2[0] - p1[0])
                        if intersection_x > ray_origin[0]:
                            intersections.append(intersection_x)

                if len(intersections) > 0:
                    intersection_count += 1

            # Check if the number of intersections is odd (inside the mesh)
            if intersection_count % 2 != 0:
                occupancy_grid[i, j] = True

    return occupancy_grid

grid_width, grid_height = max(vertices[:, 0]), max(vertices[:, 1])

occupancy_grid = ray_casting(vertices, faces, grid_width, grid_height)
print(occupancy_grid)

def visualize_occupancy_grid(occupancy_grid):
    plt.imshow(occupancy_grid.astype(int), origin='lower')
    plt.colorbar()    
    plt.show()

visualize_occupancy_grid(occupancy_grid)

# # =======================================
# # Scanning Positions:
# # =======================================

# def calculate_side_wall_scanning_positions(occupancy_grid, cell_size_x, cell_size_y):
#     grid_height, grid_width = occupancy_grid.shape

#     side_wall_scanning_positions = []

#     # Iterate over the top and bottom rows
#     for j in range(grid_width):
#         if occupancy_grid[0, j]:
#             side_wall_scanning_positions.append([j * cell_size_x, 0, 0])  # Top wall
#         if occupancy_grid[grid_height - 1, j]:
#             side_wall_scanning_positions.append([j * cell_size_x, (grid_height - 1) * cell_size_y, 0])  # Bottom wall

#     # Iterate over the left and right columns (excluding corners)
#     for i in range(1, grid_height - 1):
#         if occupancy_grid[i, 0]:
#             side_wall_scanning_positions.append([0, i * cell_size_y, 0])  # Left wall
#         if occupancy_grid[i, grid_width - 1]:
#             side_wall_scanning_positions.append([(grid_width - 1) * cell_size_x, i * cell_size_y, 0])  # Right wall

#     return side_wall_scanning_positions

# def capture_side_wall_images(side_wall_positions, cell_size_x, cell_size_y, camera_intrinsics):
#     images = []

#     # Create a pyrender scene with a perspective camera
#     scene = pyrender.Scene()
#     camera = pyrender.PerspectiveCamera(intrinsics=camera_intrinsics)

#     # Iterate over the side wall positions
#     for position in side_wall_positions:
#         # Calculate the position of the camera for capturing the image
#         camera_position = [position[0], position[1], position[2] + distance_from_wall]

#         # Add the camera to the scene at the calculated position
#         scene.add(camera, pose=np.eye(4, dtype=np.float32))
#         scene.set_pose(camera, pose=pyrender.Matrix44.from_translation(camera_position))

#         # Render the scene and capture the image
#         r = pyrender.OffscreenRenderer(viewport_width, viewport_height)
#         color, _ = r.render(scene)

#         # Add the captured image to the list
#         images.append(color)

#         # Remove the camera from the scene
#         scene.remove_node(camera)

#     return images

# # Set the distance from the wall at which the images will be captured
# distance_from_wall = 0.1  # Specify the distance in the unit of the occupancy grid

# # Set the camera intrinsic parameters
# focal_length_x = 500  # Focal length in pixels (along x-axis)
# focal_length_y = 500  # Focal length in pixels (along y-axis)
# principal_point_x = 320  # Principal point position in pixels (x-coordinate)
# principal_point_y = 240  # Principal point position in pixels (y-coordinate)

# # Create the camera intrinsics matrix
# camera_intrinsics = np.array([[focal_length_x, 0, principal_point_x],
#                               [0, focal_length_y, principal_point_y],
#                               [0, 0, 1]])

# # Define the viewport size for rendering the images
# viewport_width = 640  # Width of the viewport in pixels
# viewport_height = 480  # Height of the viewport in pixels
# cell_size_x = 0.1
# cell_size_y = 0.1

# side_wall_positions = calculate_side_wall_scanning_positions(occupancy_grid, cell_size_x, cell_size_y)

# # Call the function to capture the side wall images
# captured_images = capture_side_wall_images(side_wall_positions, cell_size_x, cell_size_y, camera_intrinsics)
