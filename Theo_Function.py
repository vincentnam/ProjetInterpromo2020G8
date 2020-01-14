import os
from pipeline import Pipeline, Process, Postprocess, Preprocess
import numpy as np
import pandas as pd
import cv2


class MyPreProcess(Preprocess):
    process_desc = "Exemple de pre-process -> ne fait rien"

    def run(self, images):
        pass


class MyProcess(Process):
    process_desc = "Exemple de process -> ne fait rien"

    def run(self, images):
        pass


class MyPostProcess(Postprocess):
    process_desc = "Exemple de post-process -> ne fait rien"

    def run(self, images):
        pass


def coord_pattern_finder(image, template, threshold: float):
    """
    input:
        image : image plane cv2.imread() black and white
        template : image pattern cv2.imread() black and white
        threshold : threshold for this pattern
    output:
        position : list right angle position for this pattern on the image

    """
    position = []  # Variable output
    # List of match
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    for pos in zip(*np.where(res >= threshold)[::-1]):
        position.append(pos)
    return(position)


def templ_category(category, path='./images/'):
    """
    Input:
        Path: directory path of templates
        category: name of category (ex : business)
    Output:
        templates: list of template name
    """
    imagesTemp = os.listdir(path)
    templates = []

    for i in imagesTemp:
        if 'temp_' + category in i:
            templates.append(i)
    return(templates)


def template_from_template(img, template, thresholdMin=0.70):
    """
    intput:
        img : image plane
        template : template
        thresholdMin : threshold min to keep template or not
    output:
        template, boolean : true if found
    """
    # default Threshold
    threshold = 1
    position = coord_pattern_finder(img, template, threshold)
    h, w = template.shape
    # Reduce Threshold while no template match
    while len(position) < 1 and threshold > thresholdMin:
        threshold -= 0.005
        position = coord_pattern_finder(img, template, threshold)

    if threshold > thresholdMin:
        return((img[position[0][1]:position[0][1] + h,\
         position[0][0]:position[0][0] + w], True))
    return((None, False))


def count_list(list):
    """
    input:
        list : list
    output:
        ordored list with single occurence
    """
    dictio_count = {}
    for el in list:
        dictio_count[el] = list.count(el)
    return {k: v for k, v in sorted(dictio_count.items(),\
        key=lambda item: item[1], reverse=True)}


def best_position(img, template, nbSeat, step=0.005, thresholdMin=0.65):
    """
    Keep the best nbSeat positions
    input:
        img : image plane
        template : template find from this image
        nbSeat : for this cat
        steps : steps for threshold
    output:
        position : coord for each match
    """
    threshold = 1
    position = []
    for threshold in np.arange(thresholdMin, 1 + step, step):
        position += coord_pattern_finder(img, template, threshold)
    result = list(count_list(position).keys())
    if len(result) < nbSeat:
        return(result)
    return(result[:nbSeat])


def rematch(img, nbObjectToFind, diction, planeName, path='./images/'):
    """
    input:
        img : image plane
        nbObjectToFind : Dictionnary : {'Total_seat': nbSeatTotal,
                                        'business': nbBusinessSeat,
                                        'bar': nbBar}
        diction : diction output
        planeName :
        path : path for template directory
    output:
        diction : dictionnary {'class':[(coordX1, coordY1, h, w),
                                        (coordX2, coordY2, h, w)]}
    """
    for cat in nbObjectToFind.keys():
        diction[planeName][cat] = []
        if cat != 'Total_seat':
            # Take all template name for this category
            templates = templ_category(category=cat)
            print(templates)
            for templ in templates:
                template = cv2.imread(path + templ, 0)
                templateFind, find = template_from_template(img, template)
                # print('Ok!')
                if find:
                    # print('Ok2!')
                    position = best_position(img, templateFind,\
                     nbObjectToFind[cat])
                    h, w = template.shape
                    for i in range(len(position)):
                        position[i] = position[i] + (h, w)
                    diction[planeName][cat] += position
