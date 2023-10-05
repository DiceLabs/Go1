import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def ray_casting(vertices, faces, grid_width, grid_height):
    # Create an empty occupancy grid map
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=bool)

    # Iterate over each grid cell
    for i in range(grid_height):
        for j in range(grid_width):
            # Cast a ray from the grid cell to check for intersections
            ray = np.array([j + 0.5, i + 0.5, -100])  # Extend the ray below the mesh
            intersection_count = 0

            # Iterate over each face of the mesh
            for face in faces:
                face_vertices = vertices[face]

                # Check if the ray intersects the face
                intersections = []
                for k in range(len(face_vertices)):
                    p1, p2 = face_vertices[k], face_vertices[(k + 1) % len(face_vertices)]
                    if (p1[1] > ray[1]) != (p2[1] > ray[1]):
                        t = (ray[1] - p1[1]) / (p2[1] - p1[1])
                        intersection_x = p1[0] + t * (p2[0] - p1[0])
                        if intersection_x > ray[0]:
                            intersections.append(intersection_x)

                if len(intersections) > 0:
                    intersection_count += 1

            # Check if the number of intersections is odd (inside the mesh)
            if intersection_count % 2 != 0:
                occupancy_grid[i, j] = True

    return occupancy_grid

def visualize_occupancy_grid(occupancy_grid):
    plt.imshow(occupancy_grid, cmap='binary')
    plt.gca().invert_yaxis()
    plt.show()

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

# Create vertices and faces for a cube
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

faces = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7]
])

visualize_mesh(vertices, faces)
# Define the dimensions of the grid
grid_width = 2
grid_height = 2

# Perform ray casting to generate the occupancy grid map
occupancy_grid = ray_casting(vertices, faces, grid_width, grid_height)

# Visualize the occupancy grid map
visualize_occupancy_grid(occupancy_grid)
