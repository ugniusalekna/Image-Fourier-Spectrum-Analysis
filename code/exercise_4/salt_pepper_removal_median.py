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


noisy_CKTboard_8bit = read_image('../../data/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif')
noisy_CKTboard_float = toFloat(noisy_CKTboard_8bit)

filtered_CKTboard = apply_nonlinear_filter(noisy_CKTboard_float, size=3, type='median')

filtered_CKTboard_8bit = to8bit(filtered_CKTboard)

plot_images(noisy_CKTboard_8bit, filtered_CKTboard_8bit)