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
        position = self.coord_pattern_finder(img, template, threshold)
        h, w = template.shape
        # Reduce Threshold while no template match
        while len(position) < 1 and threshold > thresholdMin:
            threshold -= 0.005
            position = self.coord_pattern_finder(img, template,
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
            position += self.coord_pattern_finder(img, template, threshold)
        result = list(self.count_list(position).keys())
        if len(result) < nbSeat*1.1:
            return(result)
        return(result[:int(nbSeat*1.1)])

    def run(self, img, nbObjectToFind, diction, planeName,
            path='./images/'):
        """
        input:
            img : image plane
            nbObjectToFind : Dictionnary : {
                                            'business': nbBusinessSeat,
                                            'bar': nbBar
                                            }
            diction : diction output
            planeName :
            path : path for template directory
        output:
            diction : dictionnary [{'Category': 'BUSINESS',
                                    'Seat_Type': 'FLAT_BED',
                                    'Count': 30},
                                   {'Category': 'ECONOMY',
                                    'Seat_Type': 'STANDARD',
                                    'Count': 287}]
        """
        for objet in nbObjectToFind:
            if not objet['Category'] in diction[planeName].keys():
                diction[planeName][objet['Category']] = []
            # Take all template name for this category
            templates = self.templ_category(category=objet['Category'],
                                            seatType=objet['Seat_Type'],
                                            planeName=planeName)
            for templ in templates:
                templateFind, find = self.template_from_template(img,
                                                                 templ)
                if find:
                    position = self.best_position(
                        img, templateFind, objet['Count'])
                    h, w = templ.shape
                    for i in range(len(position)):
                        position[i] = position[i] + (h, w)
                    diction[planeName][objet['Category']] += position

    def show_seats_find(self, image_rgb, json=None, img_name=None):
        """
        input:
            image : Opened image with color
            json : coordonate
            img_name : image name
        output:
            NONE
        """
        color = {'BUSINESS': (255, 0, 0),
                 'ECONOMY': (0, 0, 255),
                 'FIRST': (0, 255, 0),
                 'PREMIUM': (255, 255, 0)}

        for category in json[img_name].keys():
            for pos in json[img_name][category]:
                cv2.rectangle(
                    image_rgb, pos[0:2], (pos[0] + pos[3], pos[1] + pos[2]),
                    color[category], 2)

        cv2.imshow(img_name, image_rgb)
        cv2.waitKey()
        cv2.destroyAllWindows()


nbSeat = {'Aer_Lingus_Airbus_A330-300_A_plane6.jpg': [{'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 30},
                                                      {'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 287}],
          'Aer_Lingus_Airbus_A330-200_B_plane7.jpg': [{'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 248},
                                                      {'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 23}],
          'Aer_Lingus_Airbus_A320_plane9.jpg': [{'Category': 'ECONOMY',
                                                 'Seat_Type': 'STANDARD',
                                                 'Count': 174}],
          'Aer_Lingus_Airbus_A321_plane10.jpg': [{'Category': 'ECONOMY',
                                                  'Seat_Type': 'STANDARD',
                                                  'Count': 212}],
          'Aer_Lingus_Boeing_757-200_plane2.jpg': [{'Category': 'BUSINESS',
                                                    'Seat_Type': 'FLAT_BED',
                                                    'Count': 12},
                                                   {'Category': 'ECONOMY',
                                                    'Seat_Type': 'STANDARD',
                                                    'Count': 165}],
          'Aer_Lingus_Airbus_A330-200_plane6.jpg': [{'Category': 'ECONOMY',
                                                     'Seat_Type': 'STANDARD',
                                                     'Count': 248},
                                                    {'Category': 'BUSINESS',
                                                     'Seat_Type': 'FLAT_BED',
                                                     'Count': 23}],
          'Aer_Lingus_Airbus_A330-200_plane4.jpg': [{'Category': 'ECONOMY',
                                                     'Seat_Type': 'STANDARD',
                                                     'Count': 248},
                                                    {'Category': 'BUSINESS',
                                                     'Seat_Type': 'FLAT_BED',
                                                     'Count': 23}],
          'Aer_Lingus_Airbus_A330-300_A_plane8.jpg': [{'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 30},
                                                      {'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 287}],
          'Aer_Lingus_Airbus_A330-200_B_plane5.jpg': [{'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 248},
                                                      {'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 23}],
          'Aer_Lingus_Airbus_A321_plane1.jpg': [{'Category': 'ECONOMY',
                                                 'Seat_Type': 'STANDARD',
                                                 'Count': 212}],
          'Aer_Lingus_Boeing_757-200_plane11.jpg': [{'Category': 'BUSINESS',
                                                     'Seat_Type': 'FLAT_BED',
                                                     'Count': 12},
                                                    {'Category': 'ECONOMY',
                                                     'Seat_Type': 'STANDARD',
                                                     'Count': 165}]}

# nbSeat = {}
plane_name = 'Aer_Lingus_Boeing_757-200_plane11.jpg'
img_gray = cv2.imread('../All Data/ANALYSE IMAGE/LAYOUT SEATGURU/' +
                      plane_name, 0)
img_rgb = cv2.imread('../All Data/ANALYSE IMAGE/LAYOUT SEATGURU/' +
                     plane_name, 1)

result = {plane_name: {}}
test = SeatFinder()
test.run(img_gray, nbSeat[plane_name], result, plane_name)
test.show_seats_find(img_rgb, json=result, img_name=plane_name)
print(result)

# for i in result[plane_name].values():
#     for j in i:
#         cv2.rectangle(img_rgb, j[0:2], (j[0] + j[3],
#                                         j[1] + j[2]), (0, 0, 255), 2)
# cv2.imshow('TEST', img_rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()
