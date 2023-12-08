import time
import numpy as np
import matplotlib.pyplot as plt

from functions.image_io import read_image
from functions.format_conversion import to8bit, toFloat
from frequency_filtering.spatial_filters_for_frequency import apply_filter_frequency_domain
from spatial_filtering.filters import apply_gaussian_filter, apply_block_averaging, apply_sobel_gradients

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def plot_images(input_image, filtered_image_1, filtered_image_2, title_1, title_2):
    # Plot the image in spatial and frequency representations
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Input image', fontname='Times New Roman', fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(filtered_image_1, cmap='gray')
    axs[1].set_title(title_1, fontname='Times New Roman', fontsize=14)
    axs[1].axis('off')

    axs[2].imshow(filtered_image_2, cmap='gray')
    axs[2].set_title(title_2, fontname='Times New Roman', fontsize=14)
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()


input_for_blurring_8bit = read_image('../../data/Fig0438(a)(bld_600by600).tif')
input_for_blurring = toFloat(input_for_blurring_8bit)

# Averaging filter in image space
average_blur_spatial_domain = apply_block_averaging(input_for_blurring, filter_size=81)
average_blur_spatial_domain_8bit = to8bit(average_blur_spatial_domain)
# Averaging filter in frequency space
average_blur_frequency_domain = apply_filter_frequency_domain(input_for_blurring, filter_type='average', filter_size=81)
average_blur_frequency_domain_8bit = to8bit(average_blur_frequency_domain)

plot_images(input_for_blurring_8bit, average_blur_spatial_domain_8bit, average_blur_frequency_domain_8bit,
            title_1='Average block blur in spatial domain', title_2='Average block blur in frequency domain')


# Gaussian filter in image space
gaussian_blur_spatial_domain = apply_gaussian_filter(input_for_blurring_8bit, filter_size=9, sigma=5.)
gaussian_blur_spatial_domain_8bit = to8bit(gaussian_blur_spatial_domain)

# Gaussian filter in frequency space
gaussian_blur_frequency_domain = apply_filter_frequency_domain(input_for_blurring, filter_type='gaussian', filter_size=9, sigma=5.)
gaussian_blur_frequency_domain_8bit = to8bit(gaussian_blur_frequency_domain)

plot_images(input_for_blurring_8bit, gaussian_blur_spatial_domain_8bit, gaussian_blur_frequency_domain_8bit,
            title_1='Gaussian blur in spatial domain', title_2='Gaussian blur in frequency domain')


# Differentiaion in image space
sobel_magnitude_spatial_domain = apply_sobel_gradients(input_for_blurring)
sobel_magnitude_spatial_domain_8bit = to8bit(sobel_magnitude_spatial_domain)

# Differentiation in frequency space
sobel_magnitude_frequency_domain = apply_filter_frequency_domain(input_for_blurring, filter_type='sobel')
sobel_magnitude_frequency_domain_8bit = to8bit(sobel_magnitude_frequency_domain)

plot_images(input_for_blurring_8bit, sobel_magnitude_spatial_domain_8bit, sobel_magnitude_frequency_domain_8bit,
            title_1='Sobel magnitude in spatial domain', title_2='Sobel magnitude in frequency domain')


# Function execution time calculations
start_time = time.time()
# Sobel filters in image space
sobel_magnitude_spatial_domain = apply_sobel_gradients(input_for_blurring)
end_time_1  = time.time()

# Sobel filters in frequency space
sobel_magnitude_frequency_domain = apply_filter_frequency_domain(input_for_blurring, filter_type='sobel')
end_time_2 = time.time()

time_convolution = end_time_1 - start_time
time_transform = end_time_2 - end_time_1
print(f'Convolution execution time: {time_convolution} seconds')
print(f'Fourier transform execution time: {time_transform} seconds')