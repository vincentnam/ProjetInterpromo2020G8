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
    process_desc = "OpenCV4.1.2.30 / Scikit-image 0.16-> segmentation over colour areas"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def image_process_label(self, image):
        # grayscale = rgb2gray(image)
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = clear_border(bw)
        label_image = label(cleared)
        return label_image

    def label_results(self, image, json, data_image=None,
                      min_rectangle_area=80):
        # the result will be store in this list
        image_detection_result = []
        if data_image is None:
            print("Data_image is None")
        # get the image

        # get the different area
        label_image = self.image_process_label(image)
        props = regionprops(label_image)

        # prepare the image info
        json[data_image.split('/')[-1]] = {"areas": [], "rectangles": [],
                                           "diameters": [], "coordinates": []}
        #         image_detection_result.append({
        #             'image_name': image_path.split('/')[-1],
        #             "areas": [],
        #             "rectangles": [],
        #             "diameters": [],
        #             "coordinates": []
        #         })

        # the last index in the list
        len_list = len(image_detection_result) - 1

        # by region find every rectangle that will interesting us
        for region in props:
            # bigger enough area chosen
            if region.area >= min_rectangle_area:
                json[data_image.split('/')[-1]]['areas'].append(region['Area'])
                json[data_image.split('/')[-1]]['rectangles'].append(
                    region['BoundingBox'])
                json[data_image.split('/')[-1]]['diameters'].append(
                    region['EquivDiameter'])
                json[data_image.split('/')[-1]]['coordinates'].append(
                    region['Coordinates'])

    def image_detection_result(self, image_name, im_pre, limit_area):
        # image_name : image chosen
        # data_path : path to access those images
        # layouts : seatguru or seatmaestro
        # limit_area : minimum dimension area, 80 by default

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

    def coord_template_matching_image_single(self, image, json, liste_temp,
                                             path_temp, image_name, threshold,
                                             limit_area=80):
        # liste_temp : list of templates
        # path_temp : path to access the list of templates
        # image_name : image
        # data_path : path to access the image
        # layouts : list of layouts
        # threshold : chosen, by default 0.9
        # limit_area : minimum dimension area, 80 by default

        # Initialize the dictionnary which will display the results
        temp_rcgnzd = {}
        # print(json[image_name])
        # Pre-process the image

        # Image rgb to gray

        dict_data = self.image_detection_result(image_name, image, limit_area)

        # Initialize dictionnary of templates type for the image
        type_temp = {}

        for templ in liste_temp:
            # Initialize list of (all) coordinates for each recognized template
            liste_position = []
            # Open template
            template = cv.imread(path_temp + templ, 0)
            h, w = template.shape

            # List of match
            res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

            position = [pos for pos in zip(*np.where(res >= threshold)[::-1])]

            for pos in position:
                # Draw rectangle around recognized element
                # cv.rectangle(
                #     image, pos, (pos[0] + w, pos[1] + h), (255, 255, 255), 2)

                for rect in json[image_name]['rectangles']:

                    if rect[1] < pos[0] < rect[3] \
                            and rect[0] < pos[1] < rect[2]:

                        if rect not in liste_position:
                            liste_position.append(rect)

            type_temp[templ] = liste_position

            temp_rcgnzd[image_name] = type_temp

        json[image_name] = temp_rcgnzd[image_name]

    def run(self, image, json, image_rgb=None, col_obj=None, templates=None,
            data_image=None, image_name=None, **kwargs) -> None:
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
