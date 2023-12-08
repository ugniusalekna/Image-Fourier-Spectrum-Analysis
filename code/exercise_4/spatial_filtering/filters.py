import numpy as np
from functions.format_conversion import toFloat, to8bit
from spatial_filtering.operations import correlation, convolution

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def generate_average_block(size):
    # Create a 2D array for the average filter filled with ones
    w = np.ones([size, size])
    # Normalize the 2D array by the total number of elements
    w /= size**2
    
    return w


def apply_block_averaging(image, filter_size):
    # Generate an average block filter
    average_filter = generate_average_block(filter_size)
    # Apply correlation to the input image and the generated filter
    output_image = correlation(toFloat(image), average_filter)
    
    return output_image


def generate_gaussian(size, sigma=1.):
    # Create empty 2D array for gaussian filter
    w = np.zeros([size, size])
    linspace = np.linspace(-(size - 1) / 2, (size - 1) / 2, size, dtype=np.int8)

    if (size % 2 != 0):
        # Calculate the gaussian values for each position in the filter
        for x in linspace:
            for y in linspace:
                w[x + (size-1)//2, y + (size-1)//2] = np.exp(-(x**2 + y**2) / (2. * sigma**2))

        # Normalize the filter
        w /= np.sum(w)
    else:
        print('Filter size should be an odd integer')

    return w


def apply_gaussian_filter(image, filter_size, sigma=1.):
    # Generate gaussian filter
    gaussian_filter = generate_gaussian(filter_size, sigma)
    # Apply correlation to the input image and the generated filter
    output_image = correlation(toFloat(image), gaussian_filter)
    
    return output_image


def apply_laplace_filter(image, type='axial', strength=1.):
    # Create laplace filter
    if type == 'axial':
        laplace_filter = np.array([[0,  1, 0],
                                   [1, -4, 1],
                                   [0,  1, 0]])
    
    elif type == 'diagonal':
        laplace_filter = np.array([[1,  1, 1],
                                   [1, -8, 1],
                                   [1,  1, 1]])

    else: print('Laplace filter type not valid!')
    
    # Convolve the input image with a created filter
    laplacian = convolution(image, laplace_filter)

    return laplacian


def generate_sobel_kernels():
    # Define the Sobel kernels for X and Y gradients
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    
    return kernel_x, kernel_y


def apply_sobel_gradients(image):

    kernel_x, kernel_y = generate_sobel_kernels()

    # Calculate the gradients by convolving an input image with the kernels
    gradient_x = convolution(image, kernel_x)
    gradient_y = convolution(image, kernel_y)

    # Calculate the magnitude of the gradient vector
    # (sum and power operations are computed element-wise) 
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return magnitude


def apply_spatial_filtering(image, filter_name, filter_size=5, strength=1):

    if filter_name == 'average':
        filtered_image = apply_block_averaging(image, filter_size)
    elif filter_name == 'gaussian':
        filtered_image = apply_gaussian_filter(image, filter_size, sigma=1.)
    elif filter_name == 'laplace':
        filtered_image = apply_laplace_filter(image, type='diagonal', strength=strength)
    elif filter_name == 'sobel':
        filtered_image = apply_sobel_gradients(image)

    return filtered_image