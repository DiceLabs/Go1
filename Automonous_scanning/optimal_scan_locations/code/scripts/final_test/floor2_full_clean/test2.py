import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import open3d as o3d
import random
from matplotlib.patches import Polygon
import numpy as np
from skimage.measure import label, regionprops
import numpy as np
from skimage.measure import label


mesh = o3d.io.read_triangle_mesh('nonpink.obj')
# mesh = o3d.io.read_triangle_mesh('perpendicular_faces.obj')
pcd = mesh.sample_points_poisson_disk(200000) # Sample points from the mesh

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=16),
    fast_normal_computation=True
)

o3d.visualization.draw_geometries([pcd])  # Works only outside Jupyter/Colab

plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")


# Remove points below the plane equation
points = np.asarray(pcd.points)
distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
threshold = 0.01 # Adjust this threshold if needed
filtered_indices = np.where(distances > threshold)[0]
filtered_pcd = pcd.select_by_index(filtered_indices)

filtered_pcd.paint_uniform_color([0, 0, 1])  # Paint filtered points blue
o3d.visualization.draw_geometries([filtered_pcd])

# Apply statistical outlier removal filter
pcd_filtered, _ = filtered_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries([pcd_filtered])


# Extract x and y coordinates from pcd_filtered
points = np.asarray(pcd_filtered.points)
x_coords = points[:, 0]
y_coords = points[:, 1]

print(np.min(x_coords))

# Define the grid boundaries
x_min = math.floor(np.min(x_coords*10)/10)
x_max = math.ceil(np.max(x_coords))
y_min = math.floor(np.min(y_coords*10)/10)
y_max = math.ceil(np.max(y_coords))

print(x_min)

# Define grid parameters
grid_size_x = math.ceil(x_max) - math.floor(x_min)  # Size of the grid in the x-axis
grid_size_y = math.ceil(y_max) - math.floor(y_min)  # Size of the grid in the y-axis
cell_size = 0.2  # Size of each cell

# Calculate the number of cells in each dimension
num_cells_x = int(np.ceil(grid_size_x / cell_size))
num_cells_y = int(np.ceil(grid_size_y / cell_size))

# Create an empty grid
grid = np.zeros((num_cells_y, num_cells_x), dtype=np.uint8)

# Iterate over the points and fill the corresponding cells in the grid
for point in zip(x_coords, y_coords):
    cell_indices = (
        int((point[0] - x_min) / cell_size),
        int((point[1] - y_min) / cell_size)
    )
    grid[cell_indices[1], cell_indices[0]] = 1

# Create a figure and axes
fig, ax = plt.subplots()

# Set the background color to white
ax.set_facecolor('white')

# Plot the grid with black cells and visible grid lines
ax.imshow(grid, cmap='binary', origin='lower', extent=[x_min, x_max, y_min, y_max])
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Grid', fontsize=14)

# Set the axis interval and naming
ax.set_xticks(np.arange(x_min, x_max + cell_size, cell_size))
ax.set_yticks(np.arange(y_min, y_max + cell_size, cell_size))

# Show the grid lines for each cell
ax.grid(color='black', linewidth=0.5)

# Show the plot
plt.show()

t = 2


# Modify the code to fill the grid with points only if the count is more than 40
for point in zip(x_coords, y_coords):
    cell_indices = (
        int((point[0] - x_min) / cell_size),
        int((point[1] - y_min) / cell_size)
    )
    # Increase the count in the grid cell if it's less than 40
    if grid[cell_indices[1], cell_indices[0]] < t:
        grid[cell_indices[1], cell_indices[0]] += 1


# Perform connected component labeling on the grid
labeled_grid, num_polygons = label(grid >= t, connectivity=1, return_num=True)

# Create a figure and axes
fig, ax = plt.subplots()

# Set the background color to white
ax.set_facecolor('white')

# Plot the labeled grid with different colors for each polygon
ax.imshow(labeled_grid, cmap='rainbow', origin='lower', extent=[x_min, x_max, y_min, y_max])
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Grid with Polygons', fontsize=14)

# Set the axis interval and naming
ax.set_xticks(np.arange(x_min, x_max + cell_size, cell_size))
ax.set_yticks(np.arange(y_min, y_max + cell_size, cell_size))

# Show the grid lines for each cell
ax.grid(color='black', linewidth=0.5)

# Show the number of polygons in the plot
ax.text(x_min, y_max, f"Number of Polygons: {num_polygons}", fontsize=12, color='black')

# Show the plot
plt.show()

# Perform connected component labeling on the grid
labeled_grid, num_polygons = label(grid >= t, connectivity=1, return_num=True)

# Create a function to check if a cell is a corner
def is_corner(cell_x, cell_y):
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Neighbor offsets for x and y directions
    
    occupied_neighbors = []
    empty_neighbors = []
    diagonal_neighbors = []
    
    for dx, dy in neighbor_offsets:
        neighbor_x = cell_x + dx
        neighbor_y = cell_y + dy
        
        if 0 <= neighbor_x < num_cells_x and 0 <= neighbor_y < num_cells_y:
            if grid[neighbor_y, neighbor_x] >=t:
                occupied_neighbors.append((dx, dy))
            else:
                empty_neighbors.append((dx, dy))
    
    if len(occupied_neighbors) == 1:
        dx, dy = occupied_neighbors[0]
        
        if (dx, dy) == (1, 0):  # Occupied in +x direction
            return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size),
                    (x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif (dx, dy) == (-1, 0):  # Occupied in -x direction
            return [(x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size),
                    (x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif (dx, dy) == (0, 1):  # Occupied in +y direction
            return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size),
                    (x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size)]
        elif (dx, dy) == (0, -1):  # Occupied in -y direction
            return [(x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size),
                    (x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
            
    elif len(occupied_neighbors) == 2:
        dx1, dy1 = occupied_neighbors[0]
        dx2, dy2 = occupied_neighbors[1]
        
        if set([(dx1, dy1), (dx2, dy2)]) == set([(1, 0), (-1, 0)]) or set([(dx1, dy1), (dx2, dy2)]) == set([(0, 1), (0, -1)]):
            return None
        
        elif set([(dx1, dy1), (dx2, dy2)]) == set([(1, 0), (0, 1)]):
            if (0, 1) in occupied_neighbors:
                return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size)]
            else:
                return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size),
                        (x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif set([(dx1, dy1), (dx2, dy2)]) == set([(1, 0), (0, -1)]):
            if (0, -1) in occupied_neighbors:
                return [(x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size)]
            else:
                return [(x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size),
                        (x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size)]
        elif set([(dx1, dy1), (dx2, dy2)]) == set([(-1, 0), (0, 1)]):
            if (-1, 0) in occupied_neighbors:
                return [(x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size)]
            else:
                return [(x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size),
                        (x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif set([(dx1, dy1), (dx2, dy2)]) == set([(-1, 0), (0, -1)]):
            if (0, -1) in occupied_neighbors:
                return [(x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
            else:
                return [(x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size),
                        (x_min + cell_x * cell_size, y_min + cell_y * cell_size)]
            
    elif len(occupied_neighbors) == 3:
        dx, dy = empty_neighbors[0]
        diagonal_neighbors = [(dx, dy), (-dy, dx), (dy, -dx)]
        
        if diagonal_neighbors == [(-1, 1), (1, 1), (1, -1)]:
            return None
        elif diagonal_neighbors == [(1, 1), (-1, 1), (-1, -1)]:
            return None
        elif diagonal_neighbors == [(1, 1), (1, -1), (-1, 1)]:
            return [(x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif diagonal_neighbors == [(1, 1), (-1, -1), (1, -1)]:
            return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size)]
        elif diagonal_neighbors == [(-1, 1), (1, -1), (-1, 1)]:
            return [(x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size)]
        elif diagonal_neighbors == [(-1, 1), (-1, -1), (1, 1)]:
            return [(x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif diagonal_neighbors == [(1, 1), (-1, 1), (1, 1)]:
            return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size),
                    (x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif diagonal_neighbors == [(1, 1), (-1, -1), (-1, 1)]:
            return [(x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size),
                    (x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size)]
        elif diagonal_neighbors == [(-1, 1), (1, -1), (1, -1)]:
            return [(x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size),
                    (x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif diagonal_neighbors == [(-1, 1), (-1, -1), (1, -1)]:
            return [(x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size),
                    (x_min + cell_x * cell_size, y_min + cell_y * cell_size)]
            
    elif len(occupied_neighbors) == 4:
        dx, dy = empty_neighbors[0]
        
        if (dx, dy) == (1, 1) and (1, -1) in occupied_neighbors and (-1, 1) in occupied_neighbors:
            return None
        elif (dx, dy) == (-1, -1) and (-1, 1) in occupied_neighbors and (1, -1) in occupied_neighbors:
            return None
        elif (dx, dy) == (1, -1) and (1, 1) in occupied_neighbors and (-1, -1) in occupied_neighbors:
            return None
        elif (dx, dy) == (-1, 1) and (-1, -1) in occupied_neighbors and (1, 1) in occupied_neighbors:
            return None
        elif (dx, dy) == (1, 1) and (1, -1) in occupied_neighbors and (-1, 1) not in occupied_neighbors:
            return [(x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif (dx, dy) == (1, 1) and (1, -1) not in occupied_neighbors and (-1, 1) in occupied_neighbors:
            return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size)]
        elif (dx, dy) == (-1, -1) and (-1, 1) in occupied_neighbors and (1, -1) not in occupied_neighbors:
            return [(x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size)]
        elif (dx, dy) == (-1, -1) and (-1, 1) not in occupied_neighbors and (1, -1) in occupied_neighbors:
            return [(x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif (dx, dy) == (1, -1) and (1, 1) in occupied_neighbors and (-1, -1) not in occupied_neighbors:
            return [(x_min + cell_x * cell_size, y_min + cell_y * cell_size),
                    (x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif (dx, dy) == (1, -1) and (1, 1) not in occupied_neighbors and (-1, -1) in occupied_neighbors:
            return [(x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size),
                    (x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size)]
        elif (dx, dy) == (-1, 1) and (-1, -1) in occupied_neighbors and (1, 1) not in occupied_neighbors:
            return [(x_min + (cell_x + 1) * cell_size, y_min + cell_y * cell_size),
                    (x_min + cell_x * cell_size, y_min + (cell_y + 1) * cell_size)]
        elif (dx, dy) == (-1, 1) and (-1, -1) not in occupied_neighbors and (1, 1) in occupied_neighbors:
            return [(x_min + (cell_x + 1) * cell_size, y_min + (cell_y + 1) * cell_size),
                    (x_min + cell_x * cell_size, y_min + cell_y * cell_size)]
    
    return None



# Create a list to store the corner coordinates for each polygon
polygon_corners = [[] for _ in range(num_polygons)]

# Iterate over the labeled grid to find the corners of each polygon
for y in range(num_cells_y):
    for x in range(num_cells_x):
        label_value = labeled_grid[y, x]
        
        if label_value > 0:
            corner_coords = is_corner(x, y)
            
            if corner_coords is not None:
                polygon_corners[label_value - 1].extend(corner_coords)

# # Print the corner coordinates for each polygon
# for i, corners in enumerate(polygon_corners):
#     print(f"Polygon {i + 1} corners:")
#     for corner in corners:
#         print(corner)
#     print()

print(polygon_corners)

# Create a figure and axes
fig, ax = plt.subplots()

# Set the background color to white
ax.set_facecolor('white')

# Plot the grid with black cells and visible grid lines
ax.imshow(grid, cmap='binary', origin='lower', extent=[x_min, x_max, y_min, y_max])
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Occupancy Grid with Corners', fontsize=14)

# Set the axis interval and naming
ax.set_xticks(np.arange(x_min, x_max + cell_size, cell_size))
ax.set_yticks(np.arange(y_min, y_max + cell_size, cell_size))

# Show the grid lines for each cell
ax.grid(color='black', linewidth=0.5)

# Plot the corners for each polygon
for i, corners in enumerate(polygon_corners):
    corners = np.array(corners)
    corners = corners.reshape(-1,2)
    ax.plot(corners[:, 0], corners[:, 1], 'ro', markersize=4)

# Show the plot
plt.show()

# def sort_corners_clockwise(corners):
#     center_x = sum(x for x, _ in corners) / len(corners)
#     center_y = sum(y for _, y in corners) / len(corners)
#     return sorted(corners, key=lambda p: (math.atan2(p[1] - center_y, p[0] - center_x) + 2 * math.pi) % (2 * math.pi))

# Holes = [sort_corners_clockwise(corners) for corners in polygon_corners]

# for i, corners in enumerate(Holes):
#     print(f"Set {i + 1}: {corners}")


# import matplotlib.pyplot as plt
# import math
# import pyclipper
# from shapely.geometry import Point,Polygon #used to chk pt in or out of poly
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

# X = [];Y = [];Pi = [];PS = [];Xn = []; S = [];Yx = [];Yn = []; Yy = []; Pout = []
# MP = []; Ym = [];Yp = [];Poly = []; YN = [];m = []; Zc = []

# ''' To find the scan locations on the vertices of the polygon, for any polygon "Poly" with holes "Holes",
#     please assign the list of co-ordinates of any polygon to a variable "Poly" and of any holes, in the given polygon, to a variable
#     "Holes" as shown in the test examples below. Make sure that the list contains vertices of the polygon in 
#     anti-clockwise direction'''

# ''' Following are the 2 interesting test example polygons and their holes'''
# Poly = [(24970,19250),(23600,19250),(20740,22110),(22790,24160),(19395,27554),\
#      (17345,25504),(15560,27289),(15560,30215),(11165,30215),(11165,27915),\
#      (12435,27915),(15220,24415),(12445,21630),(16865,17210),(19650,19995),\
#      (23600,16045),(24970,16045)] 
# Holes = [[(16000,20000),(19000,22000),(16000,23000),(15500,21500)]] 

# # Poly = [(100,0),(60,-10),(90,-30),(90,20),(100,100),(50,50),(50,70),(40,30),(10,50),(20,20),(10,10)]
# # Holes = [[(30,-10),(40,-10),(50,5),(35,10),(32,15)],[(70,-15),(80,-10),(85,10),(75,40),(70,10)]]

# Holes = polygon_corners

# # Poly = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

# # Extract x and y coordinates from the polygon
# x_coords, y_coords = zip(*Poly)

# # Create a plot and axis object
# fig, ax = plt.subplots()

# # Create a polygon patch and add it to the axis
# polygon = Polygon(Poly, closed=True, edgecolor='black', facecolor='none')
# ax.add_patch(polygon)

# # Set the x and y axis limits
# ax.set_xlim(min(x_coords), max(x_coords))
# ax.set_ylim(min(y_coords), max(y_coords))

# # Set axis labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')

# # Set plot title
# ax.set_title('Polygon Visualization')

# # Display the plot
# plt.show()


# H = Holes

# Hs = []
# for i in range(len(H)):
#     for j in range(len(H[i])):
#         Hs.append(H[i][j])

# for i in range(len(H)):
#     H[i].append(H[i][0])

# Poly.reverse()
# for i in range(len(H)):
#     H[i].reverse()

# P = Poly; AP = P; P.append(P[0])

# ''' The function det calculates the determinant'''
# def det(a, b): #readymade function taken from the net
#     return a[0] * b[1] - a[1] * b[0]

# ''' The function line_intersection find the point of intersection between the two given lines'''
# def line_intersection(line1, line2):
#     xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
#     ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here
#     div = det(xdiff, ydiff)
#     if div == 0:
#        raise Exception('lines do not intersect')
#     d = (det(*line1), det(*line2))
#     x = det(d, xdiff) / div
#     y = det(d, ydiff) / div
#     return x, y

# ''' The function shrink scales down the polygon by shrink_x and shrink_y factor'''
# def shrink(Poly):
# #how much the coordinates are moved as an absolute value
#     shrink_x = 0.01
#     shrink_y = 0.01
# #coords must be clockwise
#     lines = [[Poly[i-1], Poly[i]] for i in range(len(Poly))]
#     new_lines = []
#     for i in lines:
#         dx = i[1][0] - i[0][0]
#         dy = i[1][1] - i[0][1]
#     #this is to take into account slopes
#         if (dx*dx + dy*dy)==0:
#             continue
#         else:
#             factor = 1/(dx*dx + dy*dy)**0.5
#             new_dx = dy*shrink_x * factor
#             new_dy = dx*shrink_y * factor
#             new_lines.append([(i[0][0] + new_dx, i[0][1] - new_dy),
#                             (i[1][0] + new_dx, i[1][1] - new_dy)])
# #find position of intersection of all the lines
#     new_polygon = []
#     for i in range(len(new_lines)):
#         new_polygon.append((line_intersection(new_lines[i-1], new_lines[i])))
#     return new_polygon

# ''' The function expand scales up the polygon by shrink_x and shrink_y factor'''
# def expand(Poly):
# # how much the coordinates are moved as an absolute value
#     shrink_x = -0.01
#     shrink_y = -0.01
# # coords must be clockwise
#     lines = [[Poly[i-1], Poly[i]] for i in range(len(Poly))]
#     new_lines = []
#     for i in lines:
#         dx = i[1][0] - i[0][0]
#         dy = i[1][1] - i[0][1]
#     # this is to take into account slopes
#         if (dx*dx + dy*dy)==0:
#             continue
#         else:
#             factor = 1 / (dx*dx + dy*dy)**0.5
#             new_dx = dy*shrink_x * factor
#             new_dy = dx*shrink_y * factor
#             new_lines.append([(i[0][0] + new_dx, i[0][1] - new_dy),
#                             (i[1][0] + new_dx, i[1][1] - new_dy)])
#     new_polygon = []
#     for i in range(len(new_lines)):
#         new_polygon.append((line_intersection(new_lines[i-1], new_lines[i])))
#     return new_polygon

# Pc = shrink(Poly)

# def sampling_points(Pc):
#     Pl = []
#     for i in range(len(Pc)-1):
#         a = np.linspace(Pc[i], Pc[i+1], num=50)
#         for j in a:
#             if (j[0],j[1]) not in Pl:
#                 Pl.append((j[0],j[1]))
#     Pl.append(Pl[0])
#     return Pl
# # Pc = sampling_points(Pc)
# AAP = Pc;Ac = Pc;Ac.append(Ac[0]);Hc = []

# for i in range(len(H)):
#     Hc.append(expand(H[i]))

# Bc = Hc
# for i in range(len(Bc)):
#     Bc[i].append(Bc[i][0])

# '''Now put all the vertices of the Hc in Pc'''
# for i in range(len(Ac)-1):
#     Zc.append(Ac[i])
# for i in range(len(Bc)):
#     for j in range(len(Bc[i])-1):
#         Zc.append(Bc[i][j])

# ''' The function Sorting sorts the list'''
# def Sorting(lst):
#     lst2 = sorted(lst, key=len, reverse = True)
#     return lst2


# ''' orientation function: To check the orientation on points (x1,y1),(x2,y2),(x3,y3)'''
# def orientation(x1,y1,x2,y2,x3,y3):
#         val = (float((y2-y1)*(x3-x2)))-(float((x2-x1)*(y3-y2)))
#         if (val>0):
#             return 1 #clockwise
#         elif (val<0):
#             return 2 #counterclockwise
#         else:
#             return 0 #collinear


# ''' point_in_seg_area function: To check if the point lies in segment area'''
# def point_in_seg_area(x1,y1,x2,y2,x3,y3):
#         if ((x2<=max(x1,x3)) and (x2>=min(x1,x3))\
#                 and (y2<=max(y1,y3)) and (y2>=min(y1,y3))):
#             return True
#         return False


# ''' check_intersection function: To check if the line formed by points (x1,y1) and (x2,y2) intersects line
#     formed by (x3,y3) and (x4,y4)'''
# def check_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
#         o1 = orientation(x1,y1,x2,y2,x3,y3)
#         o2 = orientation(x1,y1,x2,y2,x4,y4)
#         o3 = orientation(x3,y3,x4,y4,x1,y1)
#         o4 = orientation(x3,y3,x4,y4,x2,y2)
#         if ((o1 == 0) and point_in_seg_area(x1,y1,x3,y3,x2,y2)): #both are neede to tell if the point is on the segment
#             return False
#         if ((o2 == 0) and point_in_seg_area(x1,y1,x4,y4,x2,y2)):
#             return False
#         if ((o3 == 0) and point_in_seg_area(x3,y3,x1,y1,x4,y4)):
#             return False
#         if ((o4 == 0) and point_in_seg_area(x3,y3,x1,y1,x4,y4)):
#             return False
#         if ((o1!=o2) and (o3!=o4)):
#             return True
#         return  False


# '''The function create_point_pair creates edges from points'''
# def create_point_pair(P):
#     Pb = []
#     for i in range(len(P)-1):
#         Pa = []
#         Pa.append(P[i])
#         Pa.append(P[i+1])
#         Pb.append(Pa)
#     return Pb
# Pb = create_point_pair(P)

# '''Making pair of the vertices of the holes to make edges'''
# Hb = []
# for i in range(len(H)):
#     Hp = create_point_pair(H[i])
#     for i in range(len(Hp)):
#         Hb.append(Hp[i])

# '''Combining holes' edges with polygon edges'''
# for i in range(len(Hb)):
#     Pb.append(Hb[i])

# Pf = []
# for i in range(len(P)-1):
#     Pf.append(P[i])
# for i in range(len(H)):
#     for j in range(len(H[i])-1):
#         Pf.append(H[i][j])


# ''' The function non_intersecting_diag creates non intersecting diagonals in the polygon. 
#     Non intersecting diagonals do not intersect with the exterior of the polygon'''
# def non_intersecting_diag(Zc,P,Pf):
#     Yx = [];Zn = []
#     for i in range(len(Zc)):
#         S = []
#         for j in range(len(Pf)):
#             Pi = []
#             Pi.append(Zc[i])
#             Pi.append(Pf[j])
#             S.append(Pi)
#         PS.append(S)
#     #print("The PS is:",PS)
#     for n in range(len(PS)):
#         for k in range(len(PS[n])):
#             Xn = []
#             for l in range(len(P)-1):
#                 if  check_intersection(PS[n][k][0][0],PS[n][k][0][1],\
#                     PS[n][k][1][0],PS[n][k][1][1],P[l][0],P[l][1],\
#                     P[l+1][0],P[l+1][1])==True:
#                       continue
#                 else:
#                       Xn.append(PS[n][k][0])
#                       Xn.append(PS[n][k][1])
#             Y = []
#             if len(Xn) == 2*(len(P)-1): #no intersection with any polygon side
#                Y.append(Xn[0])
#                Y.append(Xn[1])
#             if Y == []:
#                 continue
#             else:
#                 Yx.append(Y)
#     #print("The Yx is",Yx)
#     for p in range(len(Yx)):
#          for q in range(len(H)):
#              for r in range(len(H[q])-1):
#                  if check_intersection(Yx[p][0][0],Yx[p][0][1],\
#                     Yx[p][1][0],Yx[p][1][1],H[q][r][0],H[q][r][1],\
#                     H[q][r+1][0],H[q][r+1][1])==True:
#                     Zn.append(Yx[p])

#     for i in range(len(Zn)):
#          if Zn[i] in Yx:
#             Yx.remove(Zn[i])
#     #print("Yx is:",Yx)
#     for m in range(len(Yx)):
#         px = float((Yx[m][0][0]+Yx[m][1][0])/2)
#         py = float((Yx[m][0][1]+Yx[m][1][1])/2)
#         mp = (px,py)
#         if not (Point(mp).within(Polygon(AP))): #chk point in or out
#                 Pout.append(Yx[m])
#         for i in range(len(Holes)):
#             if (Point(mp).within(Polygon(Holes[i]))):
#                 Pout.append(Yx[m])
#         MP.append(mp)
#     #print("The list of outer lines:",Pout)
#     for n in range(len(Pout)):
#             if Pout[n] in Yx:
#                 Yx.remove(Pout[n])
#     return Yx
# Yx = non_intersecting_diag(Zc,P,Pf)


# ''' The function mini_chk_pts implements the proposed algorithm and returns the list of the scan locations' diagonals'''
# def mini_chk_pts(Ac,Zc,Bc,Pb,P,Yx,H):
#     Yn=[];M=[];Ys1=[];Ys2=[];Yk1=[];Yy1=[];Yf1 = [];Ye1 = []; R = []
#     for r in range(len(Zc)):#this is important for arranging the diagonals.
#         Yy1 = []
#         for s in range(len(Yx)):  #you have to separate it
#             if (Zc[r] == Yx[s][0]):
#                 for t in range(len(P)-1):
#                     if (P[t] == Yx[s][1]):
#                         Yy1.append(Yx[s])
#         if not Yy1 == []:
#                Yy1.append(Yy1[0])
#         Ys1.append(Yy1)
#     for r in range(len(Zc)):#this is important for arranging the diagonals.
#         Yy2 = []
#         for s in range(len(Yx)):  #you have to separate it
#             if (Zc[r] == Yx[s][0]):
#                 for t in range(len(H)):
#                     for u in range(len(H[t])-1):
#                         if (H[t][u] == Yx[s][1]):
#                             Yy2.append(Yx[s])
#         if not Yy2 == []:
#                Yy2.append(Yy2[0])
#         Ys2.append(Yy2)

#     for i in range(len(Ys1)):   #have a look at this, I have made some changes
#         for j in range(len(Ys2)):
#             for k in range(len(Ys2[j])):
#                 if Ys1[i][0][0] == Ys2[j][k][0]:
#                    Ys1[i].append(Ys2[j][k])
#     Yk1 = Sorting(Ys1)
#     #print("The list Yk1 is:",Yk1)
#     for b in range(len(Yk1)):
#             Yg = []
#             for c in range(len(Yk1[b])-1):
#                  for a in range(len(P)-1):
#                      Yf = []
#                      if ((P[a] == Yk1[b][c][1]) and (P[a+1] == Yk1[b][c+1][1])):
#                          Yf.append(Yk1[b][c])
#                          Yf.append(Yk1[b][c+1])
#                          Yg.append(Yf)
#                  for d in range(len(H)):
#                      for e in range(len(H[d])-1):
#                          Ye = []
#                          if ((H[d][e] == Yk1[b][c][1]) and (H[d][e+1] == Yk1[b][c+1][1])):
#                              Ye.append(Yk1[b][c])
#                              Ye.append(Yk1[b][c+1])
#                              Yg.append(Ye)
#             if not Yg == []:
#                 Ye1.append(Yg)
#     Yf1 = Sorting(Ye1)
#     F = Pb
#     Yf2 = []
#     # print(Yf1)
#     # print(Pb)
#     ''' Make changes in the algorithm for the triangles' case'''
#     while F != []:
#         # print(F)
#         Yy = []; Ys = []; M = []
#         for a in range(len(Yf1)):
#             Yy = []
#             for b in range(len(Yf1[a])):
#                 for c in range(len(F)):
#                     if (F[c][0] in Yf1[a][b][0]) and (F[c][1] in Yf1[a][b][1])\
#                        and (Yf1[a][b][0][1] in F[c]) and Yf1[a][b][1][1] in F[c]:
#                     #  This is not that easy, think about the shrink polygon vertex, it might still connect
#                         Yy.append(Yf1[a][b])
#             if not Yy == []:
#                    Ys.append(Yy)
#         Yf2 = Sorting(Ys)

#         # '''...........................................................'''
#         # '''This part of code compares the distances of the guards with the previous guards, in the hope of binding them closer'''

#         # Yf2_len = []
#         # for i in Yf2:
#         #     Yf2_len.append(len(i))
#         # # print(Yf2_len)
        
#         # high = []
#         # for i in Yf2:
#         #     if len(i) == len(Yf2[0]):
#         #         high.append(Yf2.index(i))
       
#         # Dist = []
#         # for i in high:
#         #     if Yn == []:
#         #         continue
#         #     else: 
#         #         a = Yn[len(Yn)-1][0][0]    # Because the first element has no one to compare with
#         #         b = Yf2[i][0][0][0]        # current elements list
#         #         dist = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
#         #         Dist.append(dist)
           
#         # '''..........................................................'''
#         # if not Yf2 == []:
#         #     if Yn == []:
#         #         A2 = Yf2[0]
#         #     else:
#         #         A2 = Yf2[Dist.index(min(Dist))]

#         #print("Yf2:",Yf2)
#         if not Yf2 == []:
#                A2 = Yf2[0]
#         for i in range(len(Yf2[0])):
#             Yn.append(Yf2[0][i])
#         Yf2.remove(Yf2[0])
#         for j in range(len(F)):
#             for k in range(len(A2)):
#                 if (F[j][0] == A2[k][0][1]) and (F[j][1] == A2[k][1][1]):
#                         M.append(F[j])
#                 else:
#                         continue
#         F2 = []
#         for l in range(len(F)):
#                 if not F[l] in M:
#                         F2.append(F[l])
#                 else:
#                         continue
#         Yf1 = Yf2
#         F = F2
#     return Yn
# Yn = mini_chk_pts(Ac,Zc,Bc,Pb,P,Yx,H)


# ''' The function Guards given the final list of the scan locations '''
# def clean_up_final(Yn):
#     final = []; R = []; r = []
#     for i in Yn:  #avoiding repetition
#         if not i in final:
#             final.append(i)
#     for p in range(len(final)): #solution for adjecent points
#         for q in range(len(final)): #this is a big change!!!!!!!!!
#             for r in range(len(Pc)-1):
#                 if (final[p][0][0] or final[p][1][0]) == Pc[r]:
#                     if (Pc[r+1] or Pc[r-1])==(final[q][0][1] or final[q][1][1]):
#                         R.append(final[q])
#     for r in range(len(R)):
#         if R[r] in final:
#             final.remove(R[r])
#     Yn = final
#     return Yn
# Final_Diagonals = clean_up_final(Yn) 
# Yn = Final_Diagonals


# ''' The function Guards given the final list of the scan locations '''
# def Guards(Final_Diagonals):
#     Guards = []
#     for i in range(len(Final_Diagonals)):
#         if not Final_Diagonals[i][0][0] in Guards:
#             Guards.append(Final_Diagonals[i][0][0])
#     return Guards


# ''' The function plt_plot plots the polygon and scan locations with diagonals'''
# def plt_plot(P,Yn,H,Hc):
#     Hx = [] ; Hy = [];Hsx = []; Hsy = []
#     Px = [];Py = [];Dx = [];Dy = [];Sx = [];Sy = [];APx = [];APy = []
#     for h in range(len(AAP)):
#         APx.append(AAP[h][0])
#         APy.append(AAP[h][1])
#     for i in range(len(P)):
#         Px.append(P[i][0])
#         Py.append(P[i][1])
#     for c in range(len(Hc)):
#         for d in range(len(Hc[c])):
#             Hsx.append(Hc[c][d][0])
#             Hsy.append(Hc[c][d][1])
#             #plt.plot(Hsx,Hsy,color = 'r')
#     for j in range(len(Yn)):
#         Dx=[];Dy=[]
#         Dx.append(Yn[j][0][0][0])
#         Dy.append(Yn[j][0][0][1])
#         Dx.append(Yn[j][0][1][0])
#         Dy.append(Yn[j][0][1][1])
#         Dx.append(Yn[j][1][0][0])
#         Dy.append(Yn[j][1][0][1])
#         Dx.append(Yn[j][1][1][0])
#         Dy.append(Yn[j][1][1][1])
#         Sx.append(Yn[j][0][0][0])
#         Sy.append(Yn[j][0][0][1])
#         plt.plot(Dx,Dy, color = 'g')
#     plt.plot(Px,Py, color = 'b')
#     for a in range(len(H)):
#         Hx = [] ; Hy = []
#         for b in range(len(H[a])):
#             Hx.append(H[a][b][0])
#             Hy.append(H[a][b][1])
#         plt.fill(Hx,Hy,color = 'r')
#     # plt.fill(Px,Py,color = 'b')
#     plt.plot(APx,APy,color = 'b')
#     plt.scatter(Sx,Sy,s = 700,marker = '.',color = 'k')
#     return plt.show()

# print(Guards(Final_Diagonals))
# plt_plot(P,Yn,H,Hc)