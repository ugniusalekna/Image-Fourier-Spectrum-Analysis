import numpy as np
import matplotlib.pyplot as plt

# Define some functions for simple image manipulations

def show_image(image, vmin=None, vmax=None, title=None):
    # Display the image without axes
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.show()


def window_image(image):
    height, width = image.shape
    # Apply the 2D Hamming window to the image
    window_width = np.hamming(width)
    window_height = np.hamming(height)
    windowed_image = image * np.outer(window_height, window_width)
    
    return windowed_image


def pad_image(image):
    # Add 2M x 2N padding of zeros to an image
    height, width = image.shape
    padded_image = np.zeros((2 * height, 2 * width))
    padded_image[0:height, 0:width] = image
    
    return padded_image


def shift_image(image):
    # Shift the image by M/2 N/2 in frequency space
    height, width = image.shape

    # Manually loop over every pixel
    # for y in range(height):
    #     for x in range(width):
    #         image[y, x] *= (-1) ** (y + x)

    # Use numpy for faster computation (approx 1.75s vs 0.12s)
    y_indices, x_indices = np.indices((height, width))
    image *= (-1) ** (y_indices + x_indices)

    return image


def crop_image(image):
    # Crop the image back to its original size
    height, width = image.shape

    return image[0:height//2, 0:width//2]