# ===========================================================================================
# Occupancy Grid Map for 2 Cuboids
# ===========================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon

def plot_cuboids(cuboids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cuboid in cuboids:
        vertices, faces = cuboid

        # Plot the vertices
        x, y, z = zip(*vertices)
        ax.scatter(x, y, z, c='b')

        # Plot the faces
        for face in faces:
            face_vertices = [vertices[i] for i in face]
            ax.add_collection(Poly3DCollection([face_vertices], facecolors='cyan', edgecolors='k', alpha=0.5))

    # Set the axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the plot limits
    all_x = [v[0] for cuboid in cuboids for v in cuboid[0]]
    all_y = [v[1] for cuboid in cuboids for v in cuboid[0]]
    all_z = [v[2] for cuboid in cuboids for v in cuboid[0]]
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))

    plt.show()

def create_occupancy_grid(cuboids, resolution):
    # Calculate the grid size based on the resolution
    x_min = min(v[0] for cuboid in cuboids for v in cuboid[0]) - 1
    y_min = min(v[1] for cuboid in cuboids for v in cuboid[0]) - 1
    x_max = max(v[0] for cuboid in cuboids for v in cuboid[0]) + 1
    y_max = max(v[1] for cuboid in cuboids for v in cuboid[0]) + 1

    grid_width = int(np.ceil((x_max - x_min) / resolution))
    grid_height = int(np.ceil((y_max - y_min) / resolution))

    # Create an empty occupancy grid
    occupancy_grid = np.zeros((grid_height, grid_width))

    # Mark occupied cells for each cuboid face
    for cuboid in cuboids:
        vertices = cuboid[0]
        faces = cuboid[1]

        # Iterate over each face of the cuboid
        for face in faces:
            face_vertices = [vertices[i][:2] for i in face]
            face_polygon = Polygon(face_vertices)

            # Calculate the cell indices covered by the face
            for x in np.arange(x_min, x_max, resolution):
                for y in np.arange(y_min, y_max, resolution):
                    if face_polygon.contains_point((x, y)):
                        cell_x = int((x - x_min) / resolution)
                        cell_y = int((y - y_min) / resolution)
                        occupancy_grid[cell_y, cell_x] = 1

    return occupancy_grid, (x_min, y_min), resolution

def plot_occupancy_grid(occupancy_grid, origin, resolution):
    fig, ax = plt.subplots()

    # Plot the occupancy grid
    ax.imshow(occupancy_grid, cmap='Greys', origin='lower')

    # Set the ticks and tick labels
    x_ticks = np.arange(0, occupancy_grid.shape[1], 1)
    y_ticks = np.arange(0, occupancy_grid.shape[0], 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Round off the tick labels to one decimal place
    x_ticklabels = np.arange(origin[0], origin[0] + occupancy_grid.shape[1] * resolution, resolution)
    x_ticklabels = [round(x, 1) for x in x_ticklabels]
    y_ticklabels = np.arange(origin[1], origin[1] + occupancy_grid.shape[0] * resolution, resolution)
    y_ticklabels = [round(y, 1) for y in y_ticklabels]

    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add the cuboids as polygons for visualization
    for cuboid in cuboids:
        vertices = cuboid[0]
        faces = cuboid[1]
        for face in faces:
            face_vertices = [vertices[i][:2] for i in face]
            ax.add_patch(Polygon(face_vertices, edgecolor='red', facecolor='none'))

    plt.grid(True)
    plt.show()


# Define the vertices and faces for the first cuboid
vertices1 = [
    (1, 1, 1),
    (2, 1, 1),
    (2, 2, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 1, 2),
    (2, 2, 2),
    (1, 2, 2)
]

faces1 = [
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 1, 5, 4],  # Side face
    [1, 2, 6, 5],  # Side face
    [2, 3, 7, 6],  # Side face
    [3, 0, 4, 7]   # Side face
]

# Define the vertices and faces for the second cuboid
vertices2 = [
    (2, 0, 0),
    (3, 0, 0),
    (3, 1, 0),
    (2, 1, 0),
    (2, 0, 1),
    (3, 0, 1),
    (3, 1, 1),
    (2, 1, 1)
]

faces2 = [
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 1, 5, 4],  # Side face
    [1, 2, 6, 5],  # Side face
    [2, 3, 7, 6],  # Side face
    [3, 0, 4, 7]   # Side face
]

# Combine the vertices and faces of the cuboids
cuboid1 = (vertices1, faces1)
cuboid2 = (vertices2, faces2)
cuboids = [cuboid1, cuboid2]

# # Plot the cuboids
plot_cuboids(cuboids)

# Set the resolution of the occupancy grid
resolution = 0.1

# Create the occupancy grid
occupancy_grid, origin, resolution = create_occupancy_grid(cuboids, resolution)

print(occupancy_grid)

# Plot the occupancy grid
plot_occupancy_grid(occupancy_grid, origin, resolution)