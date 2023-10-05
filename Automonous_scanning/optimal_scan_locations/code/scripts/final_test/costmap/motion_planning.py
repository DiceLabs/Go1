import cv2
import numpy as np
import heapq

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

def heuristic(node, goal_position):
    # Euclidean distance heuristic
    return np.sqrt((node[0] - goal_position[0])**2 + (node[1] - goal_position[1])**2)

def a_star_search(occupancy_grid_map, start_position, goal_position):
    # Define possible movements (4-connected grid)
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Create a set to store visited nodes
    visited = set()

    # Create a priority queue (min-heap) to store nodes and their associated costs
    queue = [(0 + heuristic(start_position, goal_position), 0, start_position, [])]

    while queue:
        _, cost, current, path = heapq.heappop(queue)
        if current in visited:
            continue

        visited.add(current)

        if current == goal_position:
            return path

        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)

            if (0 <= neighbor[0] < occupancy_grid_map.shape[0] and
                0 <= neighbor[1] < occupancy_grid_map.shape[1] and
                occupancy_grid_map[neighbor] == 0 and
                neighbor not in visited):
                
                neighbor_cost = cost + 1
                neighbor_heuristic = heuristic(neighbor, goal_position)
                total_cost = neighbor_cost + neighbor_heuristic
                heapq.heappush(queue, (total_cost, neighbor_cost, neighbor, path + [neighbor]))

    # If no path is found, return an empty list
    return []

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

if __name__ == "__main__":
    image_path = "map.png"
    occupancy_grid_map = extract_occupancy_grid_map(image_path)

    # Perform 10x10 convolution and reduce the size
    reduced_occupancy_grid_map = perform_10x10_convolution(occupancy_grid_map)

    # Define the start and goal positions (replace these with the actual coordinates)
    start_position = (1750, 50)
    goal_position = (1000, 1500)

    # Find the shortest path using A* search
    path = a_star_search(reduced_occupancy_grid_map, start_position, goal_position)

    # Convert the binary occupancy grid map to an image and draw start, goal, and path positions
    occupancy_grid_image = convert_to_image(reduced_occupancy_grid_map, start_position, goal_position, path)

    # Resize the image to the desired window size
    window_width, window_height = 1200, 900
    occupancy_grid_image_resized = cv2.resize(occupancy_grid_image, (window_width, window_height))

    # Show the occupancy grid image using cv2.imshow
    cv2.imshow("Occupancy Grid Image", occupancy_grid_image_resized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
