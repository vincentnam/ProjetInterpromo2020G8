import os
import numpy as np
import cv2
from pipeline import Pipeline, Process, Postprocess, Preprocess


class SeatFinder(Process):
    process_desc = "A partir d'une image donne les coordonnées des sieges"

    def coord_pattern_finder(self, image, template, threshold: float):
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

    def templ_category(self, path='./images/TEMPLATE/', category='BUSINESS',
                       seatType='STANDARD', planeName='test.jpg'):
        """
        Create list of template open with cv2 by category and seatType
        Input:
            Path: directory path of templates
            category: name of category
            seatType: Seat type
            planeName: plane name
        Output:
            templates: list of template name
        """
        if '.png' in planeName:
            extension = 'PNG/'
        else:
            extension = 'JPG/'

        imagesTemp = os.listdir(path + category + '/' + extension)
        templates = []

        for i in imagesTemp:
            if seatType in i:
                templates.append(cv2.imread(
                    path + category + '/' + extension + i, 0))
        return(templates)

    def template_from_template(self, img, template, thresholdMin=0.70):
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
        position = self.coord_pattern_finder(self, img, template, threshold)
        h, w = template.shape
        # Reduce Threshold while no template match
        while len(position) < 1 and threshold > thresholdMin:
            threshold -= 0.005
            position = self.coord_pattern_finder(self, img, template,
                                                 threshold)

        if threshold > thresholdMin:
            return(img[position[0][1]:position[0][1] + h,
                       position[0][0]:position[0][0] + w], True)
        return(None, False)

    def count_list(self, list):
        """
        input:
            list : list
        output:
            ordored list with single occurence
        """
        dictio_count = {}
        for el in list:
            dictio_count[el] = list.count(el)
        return {k: v for k, v in sorted(dictio_count.items(),
                                        key=lambda item: item[1],
                                        reverse=True)}

    def best_position(self, img, template, nbSeat, step=0.005,
                      thresholdMin=0.65):
        """
        input:
            img : image plane
            template : template find from this image
            nbSeat : integrer - for this cat
            steps : steps for threshold
        output:
            position : coord for each match
        """
        position = []
        for threshold in np.arange(thresholdMin, 1 + step, step):
            position += self.d_pattern_finder(self, img, template, threshold)
        result = list(self.count_list(self, position).keys())
        if len(result) < nbSeat*1.1:
            return(result)
        return(result[:int(nbSeat*1.1)])

    def run(self, img, nbObjectToFind, diction, planeName,
            path='./images/'):
        """
        input:
            img : image plane
            nbObjectToFind : Dictionnary : {
                                            'Total_seat': nbSeatTotal,
                                            'business': nbBusinessSeat,
                                            'bar': nbBar
                                            }
            diction : diction output
            planeName :
            path : path for template directory
        output:
            diction : dictionnary {'class':[
                                        (coordX1, coordY1, h, w),
                                        (coordX2, coordY2, h, w)
                                    ]}
        """
        for objet in nbObjectToFind:
            if not objet['Category'] in diction[planeName].keys():
                diction[planeName][objet['Category']] = []
            # Take all template name for this category
            templates = self.templ_category(self,
                                            category=objet['Category'],
                                            seatType=objet['Seat_Type'],
                                            planeName=planeName)
            for templ in templates:
                templateFind, find = self.template_from_template(self, img,
                                                                 templ)
                if find:
                    position = self.best_position(
                        self, img, templateFind, objet['Count'])
                    h, w = templ.shape
                    for i in range(len(position)):
                        position[i] = position[i] + (h, w)
                    diction[planeName][objet['Category']] += position
