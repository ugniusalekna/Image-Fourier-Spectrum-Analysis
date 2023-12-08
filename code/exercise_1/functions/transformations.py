import numpy as np

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