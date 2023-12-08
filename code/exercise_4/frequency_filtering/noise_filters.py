import numpy as np

from frequency_filtering.transformations import apply_fourier_transform, apply_inverse_fourier_transform

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def apply_notch_filter(image, notch_locations, radius=10):

    image_amplitude, image_phase = apply_fourier_transform(image)

    P, Q = image.shape
    P *= 2
    Q *= 2
    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    notch_filter = np.ones((P, Q), dtype=float)

    # Create notch filter
    for x, y in notch_locations:
        D = np.sqrt((U - Q//2 - x)**2 + (V - P//2 - y)**2)
        notch_filter[D <= radius] = 0

    image_amplitude_filtered = image_amplitude * notch_filter

    reconstructed_image = apply_inverse_fourier_transform(image_amplitude_filtered, image_phase)

    return image_amplitude_filtered, reconstructed_image



def apply_band_filter(image, low_cutoff, high_cutoff):

    image_amplitude, image_phase = apply_fourier_transform(image)

    P, Q = image.shape
    P *= 2
    Q *= 2
    center_P, center_Q = P // 2, Q // 2
    U, V = np.meshgrid(np.arange(Q), np.arange(P))

    D = np.sqrt((U - center_Q) ** 2 + (V - center_P) ** 2)

    # Create band-pass filter
    band_filter = np.ones((P, Q), dtype=int)
    band_filter[(D >= low_cutoff) & (D <= high_cutoff)] = 0

    image_amplitude_filtered = image_amplitude * band_filter

    reconstructed_image = apply_inverse_fourier_transform(image_amplitude_filtered, image_phase)

    return image_amplitude_filtered, reconstructed_image

