import numpy as np
import matplotlib.pyplot as plt

from functions.image_io import read_image
from functions.format_conversion import to8bit
from frequency_filtering.transformations import apply_fourier_transform, apply_inverse_fourier_transform
from frequency_filtering.filters import apply_frequency_filtering

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def plot_images(input_image, filtered_image, title):
    # Plot the image in spatial and frequency representations
    fig, axs = plt.subplots(1, 2, figsize=(9, 6))
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Input image', fontname='Times New Roman', fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(filtered_image, cmap='gray')
    axs[1].set_title(title, fontname='Times New Roman', fontsize=14)
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()


image = read_image('../../data/Fig0333(a)(test_pattern_blurring_orig).tif')

image_amplitude, image_phase = apply_fourier_transform(image)

lowpass_filtered_amplitude = apply_frequency_filtering(image_amplitude, 'gaussian', 'lowpass', 30)
highpass_filtered_amplitude = apply_frequency_filtering(image_amplitude, 'gaussian', 'highpass', 30)

filtered_lowpass_image = apply_inverse_fourier_transform(lowpass_filtered_amplitude, image_phase)
filtered_highpass_image = apply_inverse_fourier_transform(highpass_filtered_amplitude, image_phase)

filtered_lowpass_image_8bit = to8bit(filtered_lowpass_image)
filtered_highpass_image_8bit = to8bit(filtered_highpass_image)

plot_images(image, filtered_lowpass_image_8bit, title='Gaussian low-pass filter applied')
plot_images(image, filtered_highpass_image_8bit, title='Gaussian high-pass filter applied')