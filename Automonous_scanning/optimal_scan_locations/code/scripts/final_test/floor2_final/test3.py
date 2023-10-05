import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

Poly = [
    (24970, 19250), (23600, 19250), (20740, 22110), (22790, 24160), (19395, 27554),
    (17345, 25504), (15560, 27289), (15560, 30215), (16490, 30215), (16490, 31500),
    (20670, 31500), (20670, 33700), (23370, 33700), (23370, 31150), (25785, 31150),
    (25785, 41415), (16740, 41416), (16740, 39400), (10060, 39400), (10060, 41415),
    (4315, 41415), (4315, 39400), (1300, 39400), (1300, 31300), (3545, 31300),
    (3545, 34300), (6245, 34300), (6245, 29085), (4785, 29085), (4785, 26570),
    (2085, 26570), (2085, 28615), (0, 28615), (0, 21110), (11925, 21110), (12445, 21630),
    (16865, 17210), (14600, 14946), (16407, 13139), (14498, 11230), (12691, 13036),
    (9085, 9430), (11430, 7085), (11430, 4800), (19590, 4800), (19590, 9250), (26255, 9250),
    (26255, 7085), (32000, 7085), (32000, 9720), (36510, 9720), (36510, 15050),
    (34330, 15050), (34330, 12850), (31430, 12850), (31430, 19250), (34330, 19250),
    (34330, 17050), (37480, 17050), (37480, 23430), (34330, 23430), (34330, 26060),
    (28385, 26060), (28385, 24260), (24970, 24260)
]

# Extract x and y coordinates from the polygon
x_coords, y_coords = zip(*Poly)

# Create a plot and axis object
fig, ax = plt.subplots()

# Create a polygon patch and add it to the axis
polygon = Polygon(Poly, closed=True, edgecolor='black', facecolor='none')
ax.add_patch(polygon)

# Set the x and y axis limits
ax.set_xlim(min(x_coords), max(x_coords))
ax.set_ylim(min(y_coords), max(y_coords))

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Set plot title
ax.set_title('Polygon Visualization')

# Display the plot
plt.show()
