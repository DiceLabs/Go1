from PIL import Image, ImageOps

def increase_border_thickness(image_path, border_size, border_color=(0, 0, 0)):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Calculate the new size of the image with increased border
    new_width = image.width + 2 * border_size
    new_height = image.height + 2 * border_size

    # Create a new image with the desired size and border color
    new_image = Image.new("RGB", (new_width, new_height), border_color)

    # Paste the original image onto the new image, leaving the border space
    new_image.paste(image, (border_size, border_size))

    return new_image

# Replace 'path_to_your_image.jpg' with the actual path to your image file
image_path = 'map.png'
border_size = 200  # Adjust this value to change the border thickness

new_image = increase_border_thickness(image_path, border_size)

# Save the new image with increased border thickness
new_image.save('map_new.png')
