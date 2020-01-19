from collections import defaultdict
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from typing import Iterable

class ImageUtil:
    """
    Documentation
    Tool class for preprocessing : contains method for image
    transformation as rgb to gray
    """

    def __init__(self, input_path: str, image_name: str, image: Iterable=None):
        """
        Documentation
        Constructor for ImageUtil class ; used in colour class as a
        class attribute. Image is open only if image is None.
        Parameters:
            input_path: path to the image to load
            image_name: image name (file name)
            image: image loaded
        """
        self.input_path = input_path
        self.image_name = image_name
        if image is None:
            self.image_pil = Image.open(self.input_path + self.image_name)
            self.image_plt = plt.imread(self.input_path + self.image_name)
        else:
            self.image_pil = image
            self.image_plt = image
            self.image = image

        self.sort_pixel = {}

    # Setter for image updating
    def set_image(self, image: iter):
        """
        Documentation
        Setter for image parameter
        Parameter:
            image: image to set
        """
        self.image_pil = image
        self.image_plt = image
        self.image = image

    def sort_pixel(self):
        """
        Documentation
        Sort the pixel value by number of occurences that they appear in the image
        """
        by_color = defaultdict(int)
        for pixel in self.image_pil.getdata():
            by_color[pixel] += 1

        self.sort_pixel = {k: v for k, v in
                           sorted(by_color.items(), key=lambda item: item[1],
                                  reverse=True)}

    def visualisation(self, x_size, y_size):
        """
        Documentation
        Show the image
        Parameters:
            x_size: width of the plot
            y_size: height of the plot
        """
        plt.figure(figsize=(x_size, y_size))
        if self.image is not None:
            plt.imshow(self.image.astype('uint8'))
        else:
            plt.imshow(self.image_plt.astype('uint8'))

    def to_rgb(self):
        """
        Documentation
        Convert the image to an RGB format from a BGR format
        Out:
            Image in an RGB format
        """
        return cv.cvtColor(self.image_plt, cv.COLOR_BGR2RGB)

    def to_gray(self):
        """
        Documentation
        Convert the image to a GRAY format from a BGR format
        Out:
            Image in a gray format
        """
        return cv.cvtColor(self.image_plt, cv.COLOR_BGR2GRAY)

    def save_image(self, output_path: str):
        """
        Documentation
        Save the image to specific location
        Parameter:
            output_path: where the image will be saved
        """
        plt.imsave(output_path + self.image_name,
                   self.image_plt.astype('uint8'))

