import numpy as np
import matplotlib.pyplot as plt

from functions.format_conversion import to8bit
from functions.transformations import apply_fourier_transform
from functions.generate_images import generate_image

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def plot_images(input_image, final_image):
    # Plot the image in spatial and frequency representations
    fig, axs = plt.subplots(1, 2, figsize=(9, 6))
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Synthetic image', fontname='Times New Roman', fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(final_image, cmap='gray')
    axs[1].set_title('Log of frequency spectrum', fontname='Times New Roman', fontsize=14)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def apply_and_visualize_fft_on_generated_images(height=256, width=256, type=None, variation=None, window=False, padding=True, shifting=True):
    synthetic_image_8bit = generate_image(height, width, type=type, variation=variation)
    synthetic_image_amplitude, _ = apply_fourier_transform(synthetic_image_8bit, window=window, padding=padding, shifting=shifting)
    synthetic_image_amplitude_8bit = to8bit(np.log(synthetic_image_amplitude + 1), mode='minmax')

    plot_images(synthetic_image_8bit, synthetic_image_amplitude_8bit)