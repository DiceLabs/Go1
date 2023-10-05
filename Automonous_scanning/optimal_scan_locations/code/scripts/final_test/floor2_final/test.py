import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define grid parameters
grid_size = 10  # Total size of the grid
cell_size = 1  # Size of each cell

# Define the obstacle locations
obstacle_locations = [(2, 2), (6, 4), (8, 7), (5,2)]

# Create a figure and axes
fig, ax = plt.subplots()

# Set the background color to white
ax.set_facecolor('white')

# Plot the grid lines
for i in range(grid_size+1):
    ax.axhline(y=i, color='black', linewidth=0.5)  # Horizontal grid lines
    ax.axvline(x=i, color='black', linewidth=0.5)  # Vertical grid lines

# Plot the obstacles as black cells
for obstacle_location in obstacle_locations:
    rect = Rectangle(obstacle_location, 1, 1, edgecolor='none', facecolor='black')
    ax.add_patch(rect)

# Find cells 3 cells away from the obstacles in each direction
cells_3_away = []
for obstacle_location in obstacle_locations:
    x, y = obstacle_location
    cells_3_away.extend([(x+3, y), (x, y+3), (x-3, y), (x, y-3)])

# Plot the cells 3 cells away from the obstacles in red color
for cell in cells_3_away:
    rect = Rectangle(cell, 1, 1, edgecolor='none', facecolor='red')
    ax.add_patch(rect)

# Set the axis limits and labels
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Grid with Obstacles and Cells 3 Cells Away', fontsize=14)

# Show the plot
plt.show()

# List the poses in array format
poses_array = []
for cell in cells_3_away:
    poses_array.append([cell[0], cell[1]])

print("Poses in Array Format:")
print(poses_array)
