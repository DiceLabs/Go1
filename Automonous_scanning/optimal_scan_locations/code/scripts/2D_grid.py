import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter
from skimage.morphology import medial_axis

# Line ends filter
def lineEnds(P):
    """Central pixel and just one other must be set to be a line end"""
    return 255 * ((P[4]==255) and np.sum(P)==510)

# Open image and make into Numpy array
im = Image.open('line.png').convert('L')
im = np.array(im)

# Skeletonize
skel = (medial_axis(im)*255).astype(np.uint8)

# Find line ends
result = generic_filter(skel, lineEnds, (3, 3))

# Save result
Image.fromarray(result).save('result.png')