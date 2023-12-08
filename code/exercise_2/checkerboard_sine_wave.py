from fourier_transform_generated import apply_and_visualize_fft_on_generated_images

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Sine checkerboard pattern default
apply_and_visualize_fft_on_generated_images(type='sine', variation='checkerboard_2')

# Sine checkerboard pattern without padding
apply_and_visualize_fft_on_generated_images(type='sine', variation='checkerboard_2', padding=False, window=False)