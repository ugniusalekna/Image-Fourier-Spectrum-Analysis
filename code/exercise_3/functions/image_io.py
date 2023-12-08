import matplotlib.pyplot as plt
from libtiff import TIFF

# Define image i/o functions for further use

def read_image(file_path):
    # Load TIFF file data
    tifImg = TIFF.open(file_path)
    # Read TIFF file data to an array
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

def read_one_tile(file_path, x_coord, y_coord):
    # Load TIFF file data
    tifImg = TIFF.open(file_path)
    # Read one tile of TIFF file to an array
    tile = TIFF.read_one_tile(tifImg, x_coord, y_coord)
    tifImg.close()
    return tile

def show_image(image):
    # Display the image
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()

def write_image(filename_out, image):
    # Open a new TIFF file for writing in 'w' (write) mode
    output_tif = TIFF.open(filename_out, mode='w')
    # Write the image to the TIFF file
    output_tif.write_image(image)
    # Close the TIFF file
    output_tif.close()