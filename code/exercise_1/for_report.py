import numpy as np
import matplotlib.pyplot as plt

from functions.image_io import read_image
from functions.format_conversion import to8bit, toFloat
from functions.image_manipulation import *
from functions.transformations import *

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# a) Read the image of size M x N
image_blown_ic_8bit = read_image('../../data/Fig0429(a)(blown_ic_crop).tif')

# Convert intensity values to float type for further calculations
image_blown_ic = toFloat(image_blown_ic_8bit)

# b) Pad the image with zeros to size 2M x 2N
padded_blown_ic = pad_image(image_blown_ic)
padded_blown_ic_8bit = to8bit(padded_blown_ic, mode='truncate')

# c) Shift the image's phase by scaling the intensity values for periodicity
shifted_blown_ic = shift_image(padded_blown_ic)
shifted_blown_ic_8bit = to8bit(shifted_blown_ic, mode='truncate')

# d) Apply Fourier Transform to the shifted image
image_magnitude, image_phase = forward_fourier_transform(shifted_blown_ic)
image_magnitude_8bit = to8bit(np.log(image_magnitude + 1), mode='minmax')

# e) Reconstruct the image using reverse Fourier Transform using magnitude and phase
reconstructed_image = reverse_fourier_transform(image_magnitude, image_phase)
# Scale the intensity values back
reconstructed_shifted_image = shift_image(reconstructed_image)
reconstructed_shifted_8bit = to8bit(reconstructed_shifted_image, mode='truncate')

# f) Crop the upper left quadrant of the reconstructed image
reconstructed_cropped_image = crop_image(reconstructed_shifted_image)

# Convert the float values to 8bit integer values
reconstructed_cropped_image_8bit = to8bit(reconstructed_cropped_image, mode='truncate')

images = [
    image_blown_ic_8bit, padded_blown_ic_8bit, shifted_blown_ic_8bit,
    image_magnitude_8bit, reconstructed_shifted_8bit, reconstructed_cropped_image_8bit]

descriptions = [
    'a: Input image of size M x N (scaled)',
    'b: Padded image to size 2M x 2N',
    'c: b shifted for periodicity by multiplying by $(-1)^{x+y}$',
    'd: Log Magnitude of the Fourier Transform of c',
    'e: Inverse of the Fourier Transform, multiplied by $(-1)^{x+y}$',
    'f: Upper left quadrant of e (scaled)']

labels = ['a', 'b', 'c', 'd', 'e', 'f']

font_name = "Times New Roman"

# Plot the images
fig, axs = plt.subplots(2, 3, figsize=(9, 16))

for j in range(2):
    for i in range(3):
        axs[j, i].imshow(images[3 * j + i], cmap='gray')
        axs[j, i].text(0.95, 0.95, labels[3* j + i], fontsize=16, fontname=font_name, color='white', transform=axs[j, i].transAxes, ha='center', va='center')
        axs[j, i].axis('off')

# Add one more axis and display the descriptions
right_axis = fig.add_axes([0.67, 0.1, 0.02, 0.8])
right_axis.axis('off')

for i, description in enumerate(descriptions):
    right_axis.text(0.1, 0.76 - i * 0.1, description, transform=right_axis.transAxes, fontsize=14, fontname=font_name, va='top', ha='left')

plt.subplots_adjust(
    top=0.92,
    bottom=0.05,
    left=0.0,
    right=0.65,
    hspace=0.03,
    wspace=0.03)
plt.show()

# Check the mean absolute difference of the image and its reconstructed version
show_image(image_blown_ic_8bit - reconstructed_cropped_image_8bit, vmin=0, vmax=255)
print(np.mean(np.abs(image_blown_ic_8bit - reconstructed_cropped_image_8bit)))

