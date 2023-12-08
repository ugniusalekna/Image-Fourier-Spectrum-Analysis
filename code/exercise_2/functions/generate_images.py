import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def add_sine(image, height, width, variation):
    frequency = 20 * 1/height
    # Create some patterns using sin and cos
    for y in range(height):
        for x in range(width):
            match variation:
                case 'horizontal':
                    # Horizontal lines
                    image[y, x] = 255 * (np.sin(2 * np.pi * frequency * y) + 1) / 2
                case 'vertical':
                    # Vertical lines
                    image[y, x] = 255 * (np.sin(2 * np.pi * frequency * x) + 1) / 2
                case 'diagonal':
                    # Diagonal lines
                    image[y, x] = 255 * (np.sin(2 * np.pi * frequency * (x + y)) + 1) / 2
                case 'checkerboard_1':
                    # Checkerboard pattern
                    image[y, x] = 255 * ((np.sin(2 * np.pi * frequency * x) + 1) / 2) * ((np.sin(2 * np.pi * frequency * y) + 1) / 2)
                case 'checkerboard_2':
                    # Different checkerboard pattern
                    image[y, x] = 255 * (np.cos(2 * np.pi * frequency * x) + np.sin(2 * np.pi * frequency * y) + np.sqrt(2))/(2 * np.sqrt(2))


def add_circle(image, height, width, params):
    # Extract circle parameters
    r, R = params
    # Draw a circle in the image
    for y in range(height):
        for x in range(width):
            circle_equation = (x - width//2)**2 + (y - height//2)**2
            if(circle_equation >= r**2 and circle_equation < R**2):
                image[y, x] = 255


def add_square(image, height, width, params):
    # Extract square parameters
    a, b = params
    # Draw a square in the image
    for y in range(height):
        for x in range(width):
            square_equation = np.abs(x + y - width)/np.sqrt(2) + np.abs(y - x)/np.sqrt(2)
            if(square_equation >= a and square_equation < b):
                image[y, x] = 255


def add_cross(image, height, width, params):
    # Extract X parameters
    thickness = params
    # Draw X in the image
    for y in range(height):
        for x in range(width):
            if abs(x - y) < thickness:
                image[y, x] = 255
            elif abs(x - (width - y)) < thickness:
                image[y, x] = 255


def add_mandelbrot(image, height, width):

    def mandelbrot(c, max_iter):
        # Mandelbrot algorithm
        z = 0
        n = 0
        while np.abs(z) <= 2 and n < max_iter:
            z = z*z + c
            n += 1
        return n
    
    x = np.linspace(-2, 0.5, height)
    y = np.linspace(-1.25, 1.25, width)

    for i in range(height):
        for j in range(width):
            image[i, j] = mandelbrot(complex(x[i], y[j]), 100)


def generate_image(height, width, type=None, variation=None):

    # Create an empty 2D NumPy array for an 8-bit image
    grayscale_image = np.zeros((height, width))

    match type:
        case 'sine':
            add_sine(grayscale_image, height, width, variation)
        
        case 'circle':
            # r, R = 80, 110
            r, R = 0, 60
            params = [r, R]
            add_circle(grayscale_image, height, width, params)
        
        case 'square':
            # a, b = 80, 110
            a, b = 0, 50
            params = [a, b]
            add_square(grayscale_image, height, width, params)
            
        case 'cross':
            thickness = 20
            add_cross(grayscale_image, height, width, thickness)

        case 'mixed_shape':
            r, R = 90, 110
            params_circle = [r, R]
            add_circle(grayscale_image, height, width, params_circle)
            
            a, b = 60, 90
            params_square = [a, b]
            add_square(grayscale_image, height, width, params_square)
            
            thickness = 15
            add_cross(grayscale_image, height, width, thickness)

        case 'mandelbrot':
            add_mandelbrot(grayscale_image, height, width)
        
    return grayscale_image