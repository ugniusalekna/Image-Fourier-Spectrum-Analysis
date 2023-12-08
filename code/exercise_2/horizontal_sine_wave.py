from fourier_transform_generated import apply_and_visualize_fft_on_generated_images

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Sine vertical pattern default
apply_and_visualize_fft_on_generated_images(type='sine', variation='horizontal')

# Sine vertical pattern with window function
apply_and_visualize_fft_on_generated_images(type='sine', variation='horizontal', window=True)

# Sine vertical pattern without padding
apply_and_visualize_fft_on_generated_images(type='sine', variation='horizontal', padding=False, window=False)

