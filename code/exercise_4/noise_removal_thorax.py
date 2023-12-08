import numpy as np
import matplotlib.pyplot as plt

from functions.image_io import read_image
from functions.format_conversion import to8bit, toFloat
from spatial_filtering.nonlinear_filters import apply_nonlinear_filter

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def plot_images(noisy_image, final_image):
    # Plot the image in spatial and frequency representations

    fig, axs = plt.subplots(1, 2, figsize=(9, 6))

    axs[0].imshow(noisy_image, cmap='gray')
    axs[0].set_title('Image with salt-and-pepper', fontname='Times New Roman', fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(final_image, cmap='gray')
    axs[1].set_title('Image with removed noise', fontname='Times New Roman', fontsize=14)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


noisy_thorax1_8bit = read_image('../../data/Thorax-Noise1.tif')
noisy_thorax1_float = toFloat(noisy_thorax1_8bit)

filtered_thorax1 = apply_nonlinear_filter(noisy_thorax1_float, size=3, type='median')

filtered_thorax1_8bit = to8bit(filtered_thorax1)

plot_images(noisy_thorax1_8bit, filtered_thorax1_8bit)