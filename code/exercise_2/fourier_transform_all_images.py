from fourier_transform_generated import apply_and_visualize_fft_on_generated_images

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Sine vertical pattern
apply_and_visualize_fft_on_generated_images(type='sine', variation='vertical', padding=False)

# Sine horizontal pattern
apply_and_visualize_fft_on_generated_images(type='sine', variation='horizontal', padding=False)

# Sine diagonal pattern
apply_and_visualize_fft_on_generated_images(type='sine', variation='diagonal', padding=False)

# Sine checkerboard pattern 1
apply_and_visualize_fft_on_generated_images(type='sine', variation='checkerboard_1', padding=False)

# Sine checkerboard pattern 2
apply_and_visualize_fft_on_generated_images(type='sine', variation='checkerboard_2', padding=False)

# Circle shape
apply_and_visualize_fft_on_generated_images(type='circle')

# Square shape
apply_and_visualize_fft_on_generated_images(type='square')

# Cross shape
apply_and_visualize_fft_on_generated_images(type='cross')

# A mixture of multiple shapes
apply_and_visualize_fft_on_generated_images(type='mixed_shape')

# Mandelbrot fractal image
apply_and_visualize_fft_on_generated_images(height=256*4, width=256*4, type='mandelbrot')