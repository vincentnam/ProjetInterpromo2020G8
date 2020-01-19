from .process import Process
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
import cv2 as cv
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import matplotlib.pyplot as plt
import numpy as np
import os


class SegmentationZone(Process):
    """Documentation
    
    """
    process_desc = "OpenCV4.1.2.30 / Scikit-image 0.16-> segmentation over colour areas"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def image_process_label(self, image: iter):
        """Documentation
        Make processes the image to retrieve easily information
        Parameter:
            image: image chosen
        Out:
            A processed image
        """
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = clear_border(bw)
        label_image = label(cleared)
        return label_image

    def label_results(self, image: iter, json: dict, data_image: str=None,
                      min_rectangle_area: int=80):
        """
        Documentation
        Retrieve information concerning an specific area
        Parameters:
            image: image chosen
            json : json containing the image info
            data_image : data path
            min_rectangle_area : minimum dimension area, 80 by default
        Out :
            A list of rectangles representing specific elements in the image
        """

        if data_image is None:
            print("Data_image is None")

        # get the different area
        label_image = self.image_process_label(image)
        props = regionprops(label_image)

        # prepare the image info
        json[data_image.split('/')[-1]] = {"areas": [], "rectangles": [],
                                           "diameters": [], "coordinates": []}

        # by region find every rectangle that will interesting us
        for region in props:
            # only some region are chosen
            if region.area >= min_rectangle_area:
                json[data_image.split('/')[-1]]['areas'].append(region['Area'])
                json[data_image.split('/')[-1]]['rectangles'].append(
                    region['BoundingBox'])
                json[data_image.split('/')[-1]]['diameters'].append(
                    region['EquivDiameter'])
                json[data_image.split('/')[-1]]['coordinates'].append(
                    region['Coordinates'])

    def image_detection_result(self, image_name: str, im_pre: iter, limit_area: int):
        """
        Documentation
        Detect every rectangle in the image nearby specific elements
        Parameters:
            image_name : image chosen
            im_pre: 
            limit_area : minimum dimension area, 80 by default
        Out :
            A list of rectangles representing specific elements in the image
        """

        # the result will be store in this list
        image_detection_result = []

        # detect the regions of an image
        label_image = self.image_process_label(im_pre)
        props = regionprops(label_image)

        # prepare the image info
        image_detection_result.append({
            'image_name': image_name,
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
            if region.area >= limit_area:
                image_detection_result[len_list]['areas'].append(
                    region['Area'])
                image_detection_result[len_list]['rectangles'].append(
                    region['BoundingBox'])
                image_detection_result[len_list]['diameters'].append(
                    region['EquivDiameter'])
                image_detection_result[len_list]['coordinates'].append(
                    region['Coordinates'])

        return image_detection_result

    def coord_template_matching_image_single(self, image: iter, json: dict, liste_temp: list,
                                             path_temp: str, image_name: str, threshold: float):
        """
        Documentation
        Recognize every specific elements in the image with a template analysis
        Parameters:
            image : image chosen
            json : json containing image info
            liste_temp : list of templates
            path_temp : path to access the list of templates
            image_name : image
            threshold : chosen, by default 0.9
        Out : 
            A list of specific elements in the image
        """
        

        # Initialize the dictionnary which will display the results
        temp_rcgnzd = {}

        # Initialize dictionnary of templates type for the image
        type_temp = {}

        for templ in liste_temp:
            # Initialize list of (all) coordinates for each recognized template
            liste_position = []
            
            # Open template
            template = cv.imread(path_temp + templ, 0)

            # List of matches
            res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

            # Position that pattern match with our templates
            position = [pos for pos in zip(*np.where(res >= threshold)[::-1])]

            # for each position
            for pos in position:
                # for each rectangle representing a position
                for rect in json[image_name]['rectangles']:
                    # if the rect in outside the position 
                    if rect[1] < pos[0] < rect[3] \
                            and rect[0] < pos[1] < rect[2]:
                        # we  add the rectangle to our list of positions
                        if rect not in liste_position:
                            liste_position.append(rect)

            type_temp[templ] = liste_position

            temp_rcgnzd[image_name] = type_temp

        json[image_name] = temp_rcgnzd[image_name]

    def run(self, image: iter, json: dict, image_rgb: iter=None, col_obj: str=None, 
            templates: iter=None, data_image=None, image_name: str=None, **kwargs) -> None:
        """
        Documentation
        Main of this class
        Parameters:
            image : image chosen
            json : json containing image information
            image_rgb : image in rgb
            col_obj : a Colour object
            templates : the templates
            data_image : 
            image_name : image name
        Out :
            A list of specific elements in the image
        """
        plt.imshow(image)
        plt.show()
        self.label_results(image, json, data_image)
        temp_zone_fold_path = "./images/zone_templates/"
        list_temp = [name_template for name_template in
                     os.listdir(temp_zone_fold_path) 
                     if 'png' in name_template]
        print(list_temp)
        self.coord_template_matching_image_single(
            image, json,
            image_name=image_name,
            liste_temp=list_temp,
            path_temp=temp_zone_fold_path,
            threshold=0.5)
