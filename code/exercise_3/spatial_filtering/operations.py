import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def correlation(image, filter):

    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape

    # Calculate padding size
    pad_height = filter_height // 2
    pad_width = filter_width // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Create an output image of the original size
    output_image = np.zeros_like(image)

    # Loop through all the pixels of the original image dimensions
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of the padded image
            image_region = padded_image[i:i + filter_height, j:j + filter_width]
            # Perform correlation
            correlation_result = np.sum(image_region * filter)
            # Store the result in the output image
            output_image[i, j] = correlation_result
    return output_image


def convolution(image, filter):
    # Rotate the filter using numpy
    rotated_filter = np.flip(np.flip(filter, axis=0), axis=1)

    # Perform correlation with the rotated filter (resulting in convolution)
    output_image = correlation(image, rotated_filter)

    return output_image
