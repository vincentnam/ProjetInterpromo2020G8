from pipeline import Pipeline, Process, Postprocess, Preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

from collections import defaultdict
from PIL import Image
from tqdm import tqdm_notebook as tqdm


from matplotlib import image
import matplotlib.patches as mpatches
from skimage import io
import skimage.segmentation as seg
from skimage.segmentation import clear_border
import skimage.filters as filters
from skimage.filters import threshold_otsu
import skimage.draw as draw
import skimage.color as color
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

origin_path = '/home/sid2018-6/Documents/proget_interpromo'
data_path = '/Interpromo2020/Interpromo2020/All Data/ANALYSE IMAGE/'

COLOURS = {
        'LAYOUT SEATGURU': {
            'jpg':{
                "blue":[139, 168, 198],
                "yellow": [247, 237, 86],
                "exit": [222, 111, 100],
                "green": [89, 185, 71],
                "red_bad_seat": [244, 121, 123],
                "blue_seat_crew": [140,169,202],
#                 "baby": [184,214,240]
            },
            'png':{
                "blue":[41,182,209],
                "yellow": [251,200,2],
                "exit": [190,190,190],
                "green": [41,209,135],
                "red_bad_seat": [226,96,82],
                "blue_seat_crew": [41,182,209],
#                 "baby": [197,197,197]
            }
        },
        'LAYOUT SEATMAESTRO': {
            'png': {
                "blue":[81,101,181],
                "exit": [1,120,175],
                "green": [120,189,198],
                "red_bad_seat": [207,90,150],
                "blue_seat_crew": [138,165,190] 
            }
        }
    }

class ImageUtil():
    def __init__(self, input_path, image_name, image=None):
        self.input_path = input_path
        self.image_name = image_name
        if image is  None:
            self.image_pil = Image.open(self.input_path + self.image_name)
            self.image_plt = plt.imread(self.input_path + self.image_name)
            self.image = image
        else:
            self.image_pil = image
            self.image_plt = image
            self.image = image
        
        self.sort_pixel = {}
        
    def sort_pixel(self):
        """
            Sort the pixel value by number of occurences that they appear in the image
        """
        by_color = defaultdict(int)
        for pixel in self.image_pil.getdata():
            by_color[pixel] += 1

        self.sort_pixel =  {k: v for k, v in sorted(by_color.items(), key=lambda item: item[1], reverse=True)}

    def visualisation(self, x_size, y_size):
        """
            Show the image
            params : 
                x_size - width of the plot
                y_size - height of the plot
        """
        plt.figure(figsize=(x_size,y_size))
        if self.image is not None:
            plt.imshow(self.image.astype('uint8'))
        else:
            plt.imshow(self.image_plt.astype('uint8'))

    def to_rgb(self):
        """
            Convert the image to an RGB format from a BGR format
        """
        return cv.cvtColor(self.image_plt, cv.COLOR_BGR2RGB)

class Colour():
    
    def __init__(self, given_image, image_name, given_layout):
	# given_image is already readable
        # given_layout is the document were is stock the image 'LAYOUT SEATGURU' or 'LAYOUT SEATMAESTRO'
	# given_extension is the format of the image 'png' or 'jpg'

        self.image = plt.imread(self.given_image)
        self.image_util = ImageUtil(self.given_image)
	self.layout = self.given_layout
	self.image_extension = image_name.split('.')[-1]
        
    def colour_detection(self, colours, epsilon, rgb_len, colour_mode, default_colour):
        """
            This function will detect the colour and will do some pre-process on it
            params : 
                colours : a dictionnary with a list of specified colours
                epsilon : threshold that allows to consider a colour from another one as close
                rgb_len : only take the 3 first elements from pixel (RGB norm)
                colour_mode : 
                    if true : it means that if we consider a colour from the image close 
                    to a colour from the "colours" dict, then it will replace the colour 
                    by the one in the dict. 
                    if false : it means that if we consider a colour from the image close 
                    to a colour from the "colours" dict, then it will replace the colour 
                    by the default color value.
                default_color : default color value that a pixel has to take
        """
        # make a copy to avoid to erase the original image
        img_copy = self.image_util.to_rgb()

        # for each line we get the pixel value
        for i, line in enumerate(self.image):
            for j, pixel in enumerate(line):
                # Get only 3 first value corresponding to R,G,B
                pixel = [int(val) if val >  1.0 else int(val*255) for val in self.image[i][j]][:rgb_len]

                # if we want to show a specific colour
                if colour_mode:
                    # default value
                    img_copy[i][j] = default_colour

                    # for each colour we change the pixel value if we find the same colour
                    for colour in colours.values():
                        if sum([1 if abs(p-b) < epsilon else 0 for p,b in zip(pixel, colour)]) == rgb_len:
                            img_copy[i][j] = colour

                # if we want to hide a colour by a default value
                else:
                    # default value
                    img_copy[i][j] = pixel

                    # for each recognized colour, we change it by the default value
                    for colour in colours.values():
                            if sum([1 if abs(p-b) < epsilon else 0 for p,b in zip(pixel, colour)]) == rgb_len:
                                img_copy[i][j] = default_colour
        return img_copy


    def colour_pipeline(self, colours = {}, epsilon = 20, colour_mode = True, 
                    default_colour = [0, 0, 0], rgb_len = 3):
        """
            Call colour_detection function in order to pre-process colours in image
            params : 
                colours : a dictionnary with a list of specified colours
                epsilon : threshold that allows to consider a colour from another one as close
                rgb_len : only take the 3 first elements from pixel (RGB norm)
                colour_mode : 
                    - if true (highlight colours in "colours" dict by standardize it) : it means that 
                    if we consider a colour from the image close to a colour from the "colours" dict, 
                    then it will replace the colour by the one in the dict. 
                    - if false (remove colours in "colours" dict by the default one) : it means that 
                    if we consider a colour from the image close to a colour from the "colours" dict, 
                    then it will replace the colour by the default color value.
                default_color : default color value that a pixel has to take
        """
        # if colours is empty we take the default value
        if not bool(colours): colours = COLOURS[self.layout][self.image_extension]
            
        # get the image result from colour decection pre-process wanted
        image_res = self.colour_detection(colours, epsilon, rgb_len, colour_mode, default_colour)

        return image_res

data_path= '/home/sid2018-6/Documents/proget_interpromo/Interpromo2020/Interpromo2020/All Data/ANALYSE IMAGE/'

json = {} 

image_name = 'Aer_Lingus_Airbus_A321_plane10.jpg'
# image_name is used for the extension .png or .jpg

# change the LAYOUT, for now we only use the first
layouts = ['LAYOUT SEATGURU', 'LAYOUT SEATMAESTRO']

image = cv2.imread(data_path + layout[0] + '/' + image_name)

# Create a Colour object
col_obj = Colour(image, layouts[0], image_name)
    
# Make a colour detection based on the layout type ('GURU' or 'MAESTRO') and image type('png', 'jpg')
image_pre_process  = col_obj.colour_pipeline(colours = {}, epsilon = 30, colour_mode = True, 
                        default_colour = [0, 0, 0], rgb_len = 3)
	


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def image_process_label(image):
    grayscale = rgb2gray(image)
    thresh = threshold_otsu(grayscale)
    bw = closing(grayscale > thresh, square(3))
    cleared = clear_border(bw)
    label_image = label(cleared)
    return label_image

def label_results(image, im_name , min_rectangle_area = 80):
    # the result will be store in this list
    image_detection_result = []
        
    # get the different area
    label_image = image_process_label(image)
    props = regionprops(label_image)

    # prepare the image info
    image_detection_result.append({
        'name_im': im_name,
        "areas": [],
        "rectangles": [],
        "diameters": [],
        "coordinates": []
    })
        
    # the last index in the list
    len_list = len(image_detection_result) - 1
        
    # by region find every rectangle that will interesting us
    for region in props:
        # bigger enough area chosen
        if region.area >= min_rectangle_area:
            image_detection_result[len_list]['areas'].append(region['Area'])
            image_detection_result[len_list]['rectangles'].append(region['BoundingBox'])
            image_detection_result[len_list]['diameters'].append(region['EquivDiameter'])
            image_detection_result[len_list]['coordinates'].append(region['Coordinates'])
                
    return pd.DataFrame(image_detection_result)

# df['rectangle'][0] give us all the interesting zone of the image
df = label_results(im_pre_process, image_name)



