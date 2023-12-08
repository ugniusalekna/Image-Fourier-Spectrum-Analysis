import numpy as np
import matplotlib.pyplot as plt

from functions.format_conversion import to8bit
from frequency_filtering.transformations import apply_fourier_transform, apply_inverse_fourier_transform
from spatial_filtering.filters import generate_gaussian, generate_average_block, generate_sobel_kernels

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def pad_filter_to_image(filter_kernel, image_shape):
    image_height, image_width = image_shape

    filter_size = filter_kernel.shape[0]

    padded_filter = np.zeros((image_height, image_width))

    pad_height = (image_height - filter_size) // 2
    pad_width = (image_width - filter_size) // 2
    padded_filter[pad_height:pad_height + filter_size, pad_width:pad_width + filter_size] = filter_kernel

    return padded_filter


def apply_filter(image, filter):

    padded_filter = pad_filter_to_image(filter, image.shape)

    filter_amplitude, filter_phase = apply_fourier_transform(padded_filter)
    image_amplitude, image_phase = apply_fourier_transform(image)

    filtered_amplitude = filter_amplitude * image_amplitude
    filtered_phase = image_phase

    # filter_fourier_transform = filter_amplitude * np.exp(1j * filter_phase) 
    # image_fourier_transform = image_amplitude * np.exp(1j * image_phase)

    # filtered_fourier_transform = filter_fourier_transform * image_fourier_transform

    # filtered_amplitude = np.abs(filtered_fourier_transform)
    # filtered_phase = np.angle(filter_fourier_transform)

    reconstructed_image = apply_inverse_fourier_transform(filtered_amplitude, filtered_phase)

    return reconstructed_image


def apply_filter_frequency_domain(image, filter_type, filter_size=5, sigma=1.):

    if filter_type == 'gaussian':
        filter = generate_gaussian(size=filter_size, sigma=sigma)
        reconstructed_image = apply_filter(image, filter)

    elif filter_type == 'average':
        filter = generate_average_block(size=filter_size)
        reconstructed_image = apply_filter(image, filter)
        
    elif filter_type == 'sobel':
        kernel_x, kernel_y = generate_sobel_kernels()
        gradient_x = apply_filter(image, kernel_x)
        gradient_y = apply_filter(image, kernel_y)

        reconstructed_image = np.sqrt(gradient_x**2 + gradient_y**2)

    return reconstructed_image