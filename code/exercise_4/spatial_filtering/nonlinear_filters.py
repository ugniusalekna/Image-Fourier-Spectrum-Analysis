import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def apply_nonlinear_filter(image, size, type):

    image_width, image_length = image.shape
    output_image = np.copy(image).astype(float)

    # Define a margin to avoid processing (looping over) border pixels
    margin = size // 2

    for i in range(margin, image_width-margin):
        for j in range(margin, image_length-margin):
            # Extract the neighborhood around the current pixel
            neighborhood = image[i-margin:i+margin+1, j-margin:j+margin+1]

            # Apply the different non-linear filters and assign the results to the output image
            if type == 'mean':
                output_image[i, j] = np.mean(neighborhood)

            elif type == 'median':
                output_image[i, j] = np.median(neighborhood)
            
            elif type == 'min':
                output_image[i, j] = np.min(neighborhood)

            elif type == 'max':
                output_image[i, j] = np.max(neighborhood)

            elif type == 'std':
                output_image[i, j] = np.std(neighborhood)

            else:
                print('Non-linear filter type not valid!')
                    
    return output_image