import numpy as np
import matplotlib.pyplot as plt

def visualize_material_colors(mtl_file_path):
    unique_colors = set()

    with open(mtl_file_path, 'r') as file:
        current_material = None

        for line in file:
            line = line.strip()

            if line.startswith('newmtl '):
                current_material = line[7:]
            elif line.startswith('Kd '):
                color_values = list(map(float, line[3:].split()))
                unique_colors.add(tuple(color_values[:3]))

    num_colors = len(unique_colors)
    color_array = np.zeros((1, num_colors, 3), dtype=np.float)

    for i, color in enumerate(unique_colors):
        color_array[0, i] = color

    plt.imshow(color_array, aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print("Unique Kd colors:")
    for color in unique_colors:
        print(color)

# Provide the path to the .mtl file
mtl_file_path = './data/floor2.mtl'

# Visualize the unique colors defined in the .mtl file
visualize_material_colors(mtl_file_path)



