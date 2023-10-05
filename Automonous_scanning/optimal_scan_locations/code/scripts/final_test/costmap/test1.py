import cv2
import numpy as np
import heapq

def extract_occupancy_grid_map(image_path, threshold_value=128):
    # Load the cost map image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the grayscale image to create a binary occupancy grid map
    _, occupancy_grid_map = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    return occupancy_grid_map

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

def heuristic(node, goal):
    # Manhattan distance heuristic
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar(occupancy_grid_map, start_position, goal_position):
    open_list = []
    closed_set = set()

    heapq.heappush(open_list, (0, start_position))

    while open_list:
        _, current_node = heapq.heappop(open_list)

        if current_node == goal_position:
            # Reconstruct the path
            path = []
            while current_node != start_position:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start_position)
            path.reverse()
            return path

        closed_set.add(current_node)

        neighbors = [
            (current_node[0] - 1, current_node[1]),
            (current_node[0] + 1, current_node[1]),
            (current_node[0], current_node[1] - 1),
            (current_node[0], current_node[1] + 1),
        ]

        for neighbor in neighbors:
            if neighbor in closed_set or occupancy_grid_map[neighbor[0], neighbor[1]] == 255:
                continue

            tentative_g_score = g_score[current_node] + 1

            if neighbor not in [node for _, node in open_list] or tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal_position)
                heapq.heappush(open_list, (f_score, neighbor))

    return None  # Path not found

# Example usage
if __name__ == "__main__":
    image_path = "map.png"
    occupancy_grid_map = extract_occupancy_grid_map(image_path)

    # Define the start and goal positions (replace these with the actual coordinates)
    start_position = (1750, 50)
    goal_position = (1000, 1500)

    # Run A* algorithm
    came_from = {}
    g_score = {start_position: 0}
    path = astar(occupancy_grid_map, start_position, goal_position)

    # Convert the binary occupancy grid map to an image and draw start, goal, and path positions
    occupancy_grid_image = convert_to_image(occupancy_grid_map, start_position, goal_position, path)

    # Resize the image to the desired window size
    window_width, window_height = 1200, 900
    occupancy_grid_image_resized = cv2.resize(occupancy_grid_image, (window_width, window_height))

    # Show the occupancy grid image using cv2.imshow
    cv2.imshow("Occupancy Grid Image", occupancy_grid_image_resized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
