import cv2
import numpy as np
import pandas as pd

def export_to_excel(scan_locations, output_path):

    # Create a DataFrame from the occupancy grid map (NumPy array) with row and column indices
    df = pd.DataFrame(scan_locations)

    # Export the DataFrame to an Excel file
    df.to_excel(output_path, index=True, header=True)

def extract_occupancy_grid_map(image_path, threshold_value=128):
    # Load the cost map image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the grayscale image to create a binary occupancy grid map
    _, occupancy_grid_map = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    return occupancy_grid_map

def perform_convolution(occupancy_grid_map):
    # Define the kernel for 10x10 convolution
    kernel = np.ones((10, 10), dtype=np.uint8)

    # Perform convolution on the occupancy grid map
    convolved_map = cv2.filter2D(occupancy_grid_map, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # Set unoccupied cells between occupied cells to be unoccupied
    reduced_map = convolved_map.copy()
    reduced_map[convolved_map < 255] = 0

    return reduced_map

def extract_corners(occupancy_grid_map):
    # Find corners in the occupancy grid map using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(occupancy_grid_map, maxCorners=10000, qualityLevel=0.01, minDistance=20)

    # Convert the corner points to integer values
    corners = np.int0(corners)

    # Extract the corner coordinates as (x, y) pairs
    corner_coordinates = [tuple(corner[0]) for corner in corners]

    # Check if there are any nearby corners and keep only one
    filtered_corners = []
    for corner1 in corner_coordinates:
        is_nearby = False
        for corner2 in filtered_corners:
            if euclidean_distance(corner1, corner2) < 10:
                is_nearby = True
                break
        if not is_nearby:
            filtered_corners.append(corner1)

    return filtered_corners

def convert_to_image(occupancy_grid_map, path=None):
    # Convert the binary occupancy grid map back to an image
    image = cv2.cvtColor(occupancy_grid_map, cv2.COLOR_GRAY2BGR)

    if path is not None:
        # Draw the path on the image (in blue)
        for node in path:
            y, x = node
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

    return image

# Function to check if a corner point is visible from a scan location
def is_visible(scan_location, corner, occupancy_grid_map):
    scan_x, scan_y = scan_location
    corner_x, corner_y = corner

    # Bresenham's line algorithm to find all points on the line segment
    dx = abs(corner_x - scan_x)
    dy = abs(corner_y - scan_y)
    x, y = scan_x, scan_y
    sx = 1 if scan_x < corner_x else -1
    sy = 1 if scan_y < corner_y else -1
    is_visible = True

    if dx > dy:
        err = dx / 2.0
        while x != corner_x:
            if occupancy_grid_map[y, x] == 255:
                is_visible = False
                break
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != corner_y:
            if occupancy_grid_map[y, x] == 255:
                is_visible = False
                break
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    return is_visible


# Art Gallery Problem - Scanning Algorithm
def find_scan_locations(corners, occupancy_grid_map):
    scan_locations = []
    for corner in corners:
        visible_from_any_scan_location = False
        for scan_location in scan_locations:
            if is_visible(scan_location, corner, occupancy_grid_map):
                visible_from_any_scan_location = True
                break
        if not visible_from_any_scan_location:
            scan_locations.append(corner)

    return scan_locations


def calculate_centroid(scan_locations, corners, occupancy_grid_map):
    centroid_coordinates = []

    for scan_location in scan_locations:
        visible_corners = []
        for corner in corners:
            if is_visible(scan_location, corner, occupancy_grid_map):
                visible_corners.append(corner)

        num_visible_corners = len(visible_corners)

        if num_visible_corners > 0:
            centroid_x, centroid_y = scan_location
            for corner in visible_corners:
                corner_x, corner_y = corner
                centroid_x += corner_x
                centroid_y += corner_y

            centroid_x /= (1 + num_visible_corners)
            centroid_y /= (1 + num_visible_corners)

            centroid_coordinates.append((int(centroid_x), int(centroid_y)))

    return centroid_coordinates

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_centroid(scan_locations, corners, occupancy_grid_map):
    centroid_coordinates = []

    for scan_location in scan_locations:
        visible_corners = []
        for corner in corners:
            if is_visible(scan_location, corner, occupancy_grid_map):
                visible_corners.append(corner)

        num_visible_corners = len(visible_corners)

        if num_visible_corners > 0:
            centroid_x, centroid_y = scan_location
            for corner in visible_corners:
                corner_x, corner_y = corner
                centroid_x += corner_x
                centroid_y += corner_y

            centroid_x /= (1 + num_visible_corners)
            centroid_y /= (1 + num_visible_corners)

            new_centroid = (int(centroid_x), int(centroid_y))

            # Check if there is any existing centroid nearby
            is_nearby = False
            for existing_centroid in centroid_coordinates:
                if euclidean_distance(new_centroid, existing_centroid) < 30:
                    is_nearby = True
                    break

            if not is_nearby:
                centroid_coordinates.append(new_centroid)

    return centroid_coordinates

# Example usage
if __name__ == "__main__":
    image_path = "map.png"
    occupancy_grid_map = extract_occupancy_grid_map(image_path)

    # Perform 10x10 convolution and reduce the size
    reduced_occupancy_grid_map = perform_convolution(occupancy_grid_map)

    # Extract corners from the occupancy grid map
    corners = extract_corners(reduced_occupancy_grid_map)

    # Find the minimum number of scan locations to cover all corners
    scan_locations = find_scan_locations(corners, reduced_occupancy_grid_map)
    # print(scan_locations)

    centroid_coordinates = calculate_centroid(scan_locations, corners, reduced_occupancy_grid_map)
    # print("Centroid Coordinates:", centroid_coordinates)

    # Define the start and goal positions (replace these with the actual coordinates)
    # start_position = (1750, 50)
    # goal_position = (1000, 1500)

    # Convert the binary occupancy grid map to an image and draw start, goal, and path positions
    occupancy_grid_image = convert_to_image(reduced_occupancy_grid_map)
    # print(occupancy_grid_image.shape)

    # Draw corners on the image (in yellow)
    for corner in corners:
        x, y = corner
        cv2.circle(occupancy_grid_image, (x, y), 5, (0, 255, 255), -1)

    # Resize the image to the desired window size
    window_width, window_height = 1200, 900
    occupancy_grid_image_resized = cv2.resize(occupancy_grid_image, (window_width, window_height))

    # Show the occupancy grid image using cv2.imshow
    cv2.imshow("All Detected Corners", occupancy_grid_image_resized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw scan locations on the image (in purple)
    for scan_location in scan_locations:
        x, y = scan_location
        cv2.circle(occupancy_grid_image, (x, y), 5, (128, 0, 128), -1)

    # # Draw centroids on the image (in cyan)
    for centroid in centroid_coordinates:
        x, y = centroid
        cv2.circle(occupancy_grid_image, (x, y), 5, (255, 255, 0), -1)
    
    # Resize the image to the desired window size
    window_width, window_height = 1200, 900
    occupancy_grid_image_resized = cv2.resize(occupancy_grid_image, (window_width, window_height))

    # Show the occupancy grid image using cv2.imshow
    cv2.imshow("Optimal Scanning positions", occupancy_grid_image_resized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Export the occupancy grid map to an Excel file
    output_excel_path = "optimal_scan_locations_centroid.xlsx"
    export_to_excel(centroid_coordinates, output_excel_path)

    