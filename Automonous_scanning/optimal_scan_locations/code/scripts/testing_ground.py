import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def load_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = line[2:].strip().split(' ')
                vertices.append((float(vertex[0]), float(vertex[1]), float(vertex[2])))
            elif line.startswith('f '):
                face = line[2:].strip().split(' ')
                faces.append([int(idx.split('/')[0]) - 1 for idx in face])

    return vertices, faces

def create_occupancy_grid(vertices, faces, resolution):
    # Calculate the grid size based on the resolution
    x_min = min(v[0] for v in vertices)
    y_min = min(v[1] for v in vertices)
    x_max = max(v[0] for v in vertices)
    y_max = max(v[1] for v in vertices)

    grid_width = int(np.ceil((x_max - x_min) / resolution))
    grid_height = int(np.ceil((y_max - y_min) / resolution))

    # Create an empty occupancy grid
    occupancy_grid = np.zeros((grid_height, grid_width))

    # Mark occupied cells for each face
    for face in faces:
        face_vertices = [vertices[i][:2] for i in face]
        x_coords = [v[0] for v in face_vertices]
        y_coords = [v[1] for v in face_vertices]
        min_x = min(x_coords) -0.5
        max_x = max(x_coords) +0.5
        min_y = min(y_coords) -0.5
        max_y = max(y_coords) +0.5

        # Calculate the cell indices covered by the face
        for x in np.arange(min_x, max_x, resolution):
            for y in np.arange(min_y, max_y, resolution):
                if point_in_polygon(x, y, face_vertices):
                    cell_x = int((x - x_min) / resolution)
                    cell_y = int((y - y_min) / resolution)
                    occupancy_grid[cell_y, cell_x] = 1

    return occupancy_grid, (x_min, y_min), resolution

def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def plot_occupancy_grid(occupancy_grid, origin, resolution):
    fig, ax = plt.subplots()

    # Plot the occupancy grid
    ax.imshow(occupancy_grid, cmap='Greys', origin='lower')

    # Set the ticks and tick labels
    x_ticks = np.arange(0, occupancy_grid.shape[1] + 1, 1)
    y_ticks = np.arange(0, occupancy_grid.shape[0] + 1, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Round off the tick labels to one decimal place
    x_ticklabels = np.arange(origin[0], origin[0] + (occupancy_grid.shape[1] + 1) * resolution, resolution)
    x_ticklabels = [round(x, 1) for x in x_ticklabels[:occupancy_grid.shape[1]+1]]
    y_ticklabels = np.arange(origin[1], origin[1] + (occupancy_grid.shape[0] + 1) * resolution, resolution)
    y_ticklabels = [round(y, 1) for y in y_ticklabels[:occupancy_grid.shape[0]+1]]

    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.grid(True)
    plt.show()

# Define the file path for the OBJ file
file_path = 'nonfloor.obj'

# Load the OBJ file
vertices, faces = load_obj(file_path)

# Set the resolution of the occupancy grid
resolution = 0.5

# Create the occupancy grid
occupancy_grid, origin, resolution = create_occupancy_grid(vertices, faces, resolution)

# Plot the occupancy grid
plot_occupancy_grid(occupancy_grid, origin, resolution)

def calculate_horizontal_distance(camera_distance, width_captured_area, fov_angle, overlap_percentage):
    # Convert the FOV angle from degrees to radians
    fov_angle_rad = math.radians(fov_angle)

    # Calculate the horizontal distance using the formula
    x = (width_captured_area / 2) / math.tan(fov_angle_rad / 2) * (1 - overlap_percentage)
    # x = math.floor(x * 10) / 10

    return x


def calculate_camera_distance(width_captured_area, fov_angle):
    # Convert the FOV angle from degrees to radians
    fov_angle_rad = math.radians(fov_angle)

    # Calculate the camera distance using the formula
    camera_distance = (width_captured_area / 2) / math.tan(fov_angle_rad / 2)
    # camera_distance = math.floor(camera_distance * 10) / 10

    return camera_distance

# Example usage
width_captured_area = 0.5  # Width of the captured area (in units)
fov_angle = 60.0  # Field of view angle of the camera (in degrees)
overlap_percentage = 0.4  # Overlap percentage between images(between 0 and 1)

camera_distance = calculate_camera_distance(width_captured_area, fov_angle)
print("Camera Distance:", camera_distance)

horizontal_distance = calculate_horizontal_distance(camera_distance, width_captured_area, fov_angle, overlap_percentage)
print("Horizontal Distance:", horizontal_distance)