import numpy as np
from functions.format_conversion import toFloat, to8bit
from functions.image_manipulation import *

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Define forward and reverse discrete 2D Fourier Transforms using numpy.fft
# Using 'ortho' norm, which scales both forward and reverse transforms by 1/sqrt(MN)

def forward_fourier_transform(image):
    # Obtain complex Fourier Transform array of an image
    image_fft = np.fft.fft2(image, norm='ortho')
    # Calculate the amplitude and the phase of the complex values
    amplitude_spectrum = np.abs(image_fft)
    phase_spectrum = np.angle(image_fft)

    return amplitude_spectrum, phase_spectrum


def reverse_fourier_transform(amplitude_spectrum, phase_spectrum):
    # Reconstruct the polar form from the amplitude and phase arrays
    image_fft = amplitude_spectrum * np.exp(1j * phase_spectrum)
    # Do the reverse Fourier Transform on this array to reconstruct the image
    reconstructed_image = np.fft.ifft2(image_fft, norm='ortho')
    # Discard the very small values, that appeared from the floating-point errors 
    # reconstructed_image[np.abs(reconstructed_image) < 1e-10] = 0
    # Return only the real part (as the imaginary part is equal to 0)
    return reconstructed_image.real


# Define functions for full forward and inverse Fourier transform 'pipeline'

def apply_fourier_transform(input_image, window=False, padding=True, shifting=True):
    # Convert intensity values to float type for further calculations
    synthetic_image = toFloat(input_image)
    if window == True:
        # Apply the window function
        windowed_synthetic_image = window_image(synthetic_image)
    else: windowed_synthetic_image = synthetic_image
    if padding == True:
        # Pad the image with zeros to size 2M x 2N
        padded_synthetic_image = pad_image(windowed_synthetic_image)
    else: padded_synthetic_image = windowed_synthetic_image
    if shifting == True:
        # Shift the image's phase by scaling the intensity values for periodicity
        shifted_synthetic_image = shift_image(padded_synthetic_image)
    else: shifted_synthetic_image = padded_synthetic_image
    # Apply Fourier Transform to the shifted image
    image_amplitude, image_phase = forward_fourier_transform(shifted_synthetic_image)

    return image_amplitude, image_phase


def apply_inverse_fourier_transform(image_amplitude, image_phase):
    # Reconstruct the image using reverse Fourier Transform using magnitude and phase
    reconstructed_image = reverse_fourier_transform(image_amplitude, image_phase)
    # Scale the intensity values back
    reconstructed_shifted_image = shift_image(reconstructed_image)
    # Crop the upper left quadrant of the reconstructed image
    reconstructed_cropped_image = crop_image(reconstructed_shifted_image)

    return reconstructed_cropped_image
