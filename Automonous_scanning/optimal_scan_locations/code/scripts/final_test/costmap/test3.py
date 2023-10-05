import cv2
import numpy as np
import pandas as pd

def export_to_excel(occupancy_grid_map, output_path):
    # Get the shape of the occupancy grid map
    rows, cols = occupancy_grid_map.shape

    # Create a DataFrame from the occupancy grid map (NumPy array) with row and column indices
    df = pd.DataFrame(occupancy_grid_map)

    # Add row and column indices to the DataFrame
    df.index = range(rows)  # Row indices
    df.columns = range(cols)  # Column indices

    # Export the DataFrame to an Excel file
    df.to_excel(output_path, index=True, header=True)

def extract_occupancy_grid_map(image_path, threshold_value, occupied_distance):
    # Load the cost map image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the grayscale image to create a binary occupancy grid map
    _, occupancy_grid_map = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # # # Mark nearby pixels within the specified distance as occupied
    # for i in range(occupied_distance, occupancy_grid_map.shape[0] - occupied_distance):
    #     for j in range(occupied_distance, occupancy_grid_map.shape[1] - occupied_distance):
    #         if occupancy_grid_map[i, j] == 255:  # If current pixel is occupied
    #             for dx in range(-occupied_distance, occupied_distance + 1):
    #                 for dy in range(-occupied_distance, occupied_distance + 1):
    #                     # Mark nearby empty pixels as occupied
    #                     if occupancy_grid_map[i + dx, j + dy] == 0:
    #                         occupancy_grid_map[i + dx, j + dy] = 255

    return occupancy_grid_map

def convert_to_image(occupancy_grid_map, scan_location, visible_corners):
    # Convert the binary occupancy grid map back to an image
    image = cv2.cvtColor(occupancy_grid_map, cv2.COLOR_GRAY2BGR)

    # Draw the scan location (in red)
    scan_x, scan_y = scan_location
    cv2.circle(image, (scan_y, scan_x), 5, (0, 0, 255), -1)

    # Draw the visible corners (in green)
    for corner in visible_corners:
        corner_x, corner_y = corner
        cv2.circle(image, (corner_y, corner_x), 5, (0, 255, 0), -1)

    return image


# Example usage
if __name__ == "__main__":
    image_path = "map.png"
    occupancy_grid_map = extract_occupancy_grid_map(image_path, threshold_value=128, occupied_distance=1)

    # Define the scan location and visible corners
    scan_location = (1699, 1046)
    visible_corners = [(1699, 991), (1726, 995)]

    # Convert the binary occupancy grid map to an image and draw the scan location and visible corners
    occupancy_grid_image = convert_to_image(occupancy_grid_map, scan_location, visible_corners)
    print(occupancy_grid_image.shape)

    # Resize the image to the desired window size
    window_width, window_height = 1200, 900
    occupancy_grid_image_resized = cv2.resize(occupancy_grid_image, (window_width, window_height))

    # Show the occupancy grid image using cv2.imshow
    cv2.imshow("Highlighted Image", occupancy_grid_image_resized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()