import numpy as np
import matplotlib.pyplot as plt

from functions.image_io import read_image
from functions.format_conversion import to8bit, toFloat
from frequency_filtering.transformations import apply_fourier_transform
from frequency_filtering.noise_filters import apply_notch_filter

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def plot_images(noisy_image, final_image, noisy_spectrum, final_spectrum):
    # Plot the image in spatial and frequency representations

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))

    axs[0, 0].imshow(noisy_image, cmap='gray')
    axs[0, 0].set_title('Input noisy image', fontname='Times New Roman', fontsize=14)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(final_image, cmap='gray')
    axs[0, 1].set_title('Image with removed noise', fontname='Times New Roman', fontsize=14)
    axs[0, 1].axis('off')

    axs[1, 0].imshow(noisy_spectrum, cmap='gray')
    axs[1, 0].set_title('Spectrum of noisy image', fontname='Times New Roman', fontsize=14)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(final_spectrum, cmap='gray')
    axs[1, 1].set_title('Spectrum of image with removed noise', fontname='Times New Roman', fontsize=14)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


noisy_Kidney1_8bit = read_image('../../data/Kidney1-Crop-Noise1.tif')
noisy_Kidney1_float = toFloat(noisy_Kidney1_8bit)

noisy_amplitude_Kidney1, _ = apply_fourier_transform(noisy_Kidney1_float)
noisy_amplitude_Kidney1_8bit = to8bit(np.log1p(noisy_amplitude_Kidney1))

notch_locations = [(0, 380), (0, -380), (380, 0), (-380, 0),
                   (270, 270), (270, -270), (-270, 270), (-270, -270)]

filtered_amplitude_Kidney1, filtered_Kidney1 = apply_notch_filter(noisy_Kidney1_float, notch_locations, radius=10)

filtered_amplitude_Kidney1_8bit = to8bit(np.log1p(filtered_amplitude_Kidney1))
filtered_Kidney1_8bit = to8bit(filtered_Kidney1)

plot_images(noisy_Kidney1_8bit, filtered_Kidney1_8bit, noisy_amplitude_Kidney1_8bit, filtered_amplitude_Kidney1_8bit)

