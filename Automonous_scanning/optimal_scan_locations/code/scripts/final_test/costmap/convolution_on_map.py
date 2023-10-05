import cv2
import numpy as np

def extract_occupancy_grid_map(image_path, threshold_value=128):
    # Load the cost map image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the grayscale image to create a binary occupancy grid map
    _, occupancy_grid_map = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    return occupancy_grid_map

def perform_10x10_convolution(occupancy_grid_map):
    # Define the kernel for 10x10 convolution
    kernel = np.ones((10, 10), dtype=np.uint8)

    # Perform convolution on the occupancy grid map
    convolved_map = cv2.filter2D(occupancy_grid_map, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # Set unoccupied cells between occupied cells to be unoccupied
    reduced_map = convolved_map.copy()
    reduced_map[convolved_map < 255] = 0

    return reduced_map

def convert_to_image(occupancy_grid_map, start_position, goal_position, path=None):
    # Convert the binary occupancy grid map back to an image
    image = cv2.cvtColor(occupancy_grid_map, cv2.COLOR_GRAY2BGR)

    # Draw the start position (in green)
    start_x, start_y = start_position
    cv2.circle(image, (start_y, start_x), 5, (0, 255, 0), -1)

    # Draw the goal position (in red)
    goal_x, goal_y = goal_position
    cv2.circle(image, (goal_y, goal_x), 5, (0, 0, 255), -1)

    if path is not None:
        # Draw the path on the image (in blue)
        for node in path:
            y, x = node
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

    return image

# Example usage
if __name__ == "__main__":
    image_path = "map.png"
    occupancy_grid_map = extract_occupancy_grid_map(image_path)

    # Perform 10x10 convolution and reduce the size
    reduced_occupancy_grid_map = perform_10x10_convolution(occupancy_grid_map)

    # Define the start and goal positions (replace these with the actual coordinates)
    start_position = (1750, 50)
    goal_position = (1000, 1500)

    # Convert the binary occupancy grid map to an image and draw start, goal, and path positions
    occupancy_grid_image = convert_to_image(reduced_occupancy_grid_map, start_position, goal_position)

    # Resize the image to the desired window size
    window_width, window_height = 1200, 900
    occupancy_grid_image_resized = cv2.resize(occupancy_grid_image, (window_width, window_height))

    # Show the occupancy grid image using cv2.imshow
    cv2.imshow("Occupancy Grid Image", occupancy_grid_image_resized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
