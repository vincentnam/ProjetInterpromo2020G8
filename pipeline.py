"""
Created on Fri Jan 3 13:28:13 CET 2019
Group 8
@authors: DANG Vincent-Nam
"""

# TODO :
#  - Tests unitaires et tests intégrations : test pipeline
#  (run_pipeline), levées d'erreur, etc...
#  - Traduire les commentaires en anglais (si besoin ?)
#  - Mettre à jour le pipeline pour prendre en compte des resultats
#    auxiliaires nécessaire pour le traitement suivant
#  - Gestion des hints plus formellement
#  - Gestion de l'héritage des docstrings
#  - Gestion des preprocess sur les templates
from abc import ABCMeta, abstractmethod
from typing import Iterable
import sys
import inspect
import re
import traceback
import os
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import defaultdict
from PIL import Image




import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image

import sklearn
from sklearn.cluster import DBSCAN


import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image

from matplotlib import image
import matplotlib.patches as mpatches
from matplotlib import patches
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
import sklearn
from sklearn.cluster import DBSCAN

def overrides(method):
    # actually can't do this because a method is really
    # just a function while inside a class def'n
    # assert(inspect.ismethod(method))

    stack = inspect.stack()
    base_classes = re.search(r'class.+\((.+)\)\s*', stack[2][4][0]).group(1)

    # handle multiple inheritance
    base_classes = [s.strip() for s in base_classes.split(',')]
    if not base_classes:
        raise ValueError('overrides decorator: unable to determine base class')

    # stack[0]=overrides, stack[1]=inside \
    # class def'n, stack[2]=outside class def'n
    derived_class_locals = stack[2][0].f_locals

    # replace each class name in base_classes with the actual class type
    for i, base_class in enumerate(base_classes):

        if '.' not in base_class:
            base_classes[i] = derived_class_locals[base_class]

        else:
            components = base_class.split('.')

            # obj is either a module or a class
            obj = derived_class_locals[components[0]]

            for c in components[1:]:
                assert(inspect.ismodule(obj) or inspect.isclass(obj))
                obj = getattr(obj, c)

            base_classes[i] = obj

    assert(any(hasattr(cls, method.__name__) for cls in base_classes))
    return method

class NotProcessClass(Exception):
    def __init__(self, expression, message):
        """
        Classe d'exception pour indiquer qu'autre chose qu'un process a
        été rajouté dans le pipeline.
        Attributes:
            process_desc -- description du process
            process_class -- class du process
        """
        self.expression = expression
        self.message = message




COLOURS = {
    'LAYOUT SEATGURU': {
        'jpg': {
            "blue": [139, 168, 198],
            "yellow": [247, 237, 86],
            "exit": [222, 111, 100],
            "green": [89, 185, 71],
            "red_bad_seat": [244, 121, 123],
            "blue_seat_crew": [140, 169, 202],
            "baby": [184, 214, 240]
        },
        'png': {
            "blue": [41, 182, 209],
            "yellow": [251, 200, 2],
            "exit": [190, 190, 190],
            "green": [41, 209, 135],
            "red_bad_seat": [226, 96, 82],
            "blue_seat_crew": [41, 182, 209],
            "baby": [197, 197, 197]
        }
    },
    'LAYOUT SEATMAESTRO': {
        'png': {
            "blue": [81, 101, 181],
            "exit": [1, 120, 175],
            "green": [120, 189, 198],
            "red_bad_seat": [207, 90, 150],
            "blue_seat_crew": [138, 165, 190]
        }
    }
}



class ImageUtil():
    def __init__(self, input_path, image_name, image=None):
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
    def set_image(self, image):
        self.image_pil = image
        self.image_plt = image
        self.image = image

    def sort_pixel(self):
        """
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
            Show the image
            params :
                x_size - width of the plot
                y_size - height of the plot
        """
        plt.figure(figsize=(x_size, y_size))
        if self.image is not None:
            plt.imshow(self.image.astype('uint8'))
        else:
            plt.imshow(self.image_plt.astype('uint8'))

    def to_rgb(self):
        """
            Convert the image to an RGB format from a BGR format
        """
        return cv.cvtColor(self.image_plt, cv.COLOR_BGR2RGB)

    def to_gray(self):
        """
            Convert the image to a GRAY format from a BGR format
        """
        return cv.cvtColor(self.image_plt, cv.COLOR_BGR2GRAY)

    def save_image(self, output_path):
        """
            Save the image to specific location
            params :
                output_path - where the image will be saved
        """
        plt.imsave(output_path + self.image_name,
                   self.image_plt.astype('uint8'))


class Colour():

    def __init__(self, input_path, layout, image_name):
        self.input_path = input_path
        self.layout = layout
        self.image_name = image_name
        self.image_extension = image_name.split('.')[-1]

        self.image = plt.imread(
            self.input_path + self.layout + '/' + self.image_name)
        self.util_obj = ImageUtil(self.input_path + self.layout + '/',
                                    self.image_name)

    # Setter for image update on preprocess
    def set_image(self, image):
        self.image = image
        self.util_obj.set_image(image)

    def colour_detection(self, colours, epsilon, rgb_len, colour_mode,
                         default_colour):
        """
            This function will detect the colour and will do some pre-process on it
            params :
                colours : a dictionnary with a list of specified colours
                epsilon : threshold that allows to consider a colour from another one as close
                rgb_len : only take the 3 first elements from pixel (RGB norm)
                colour_mode :
                    if true : it means that if we consider a colour from the image close
                    to a colour from the "colours" dict, then it will replace the colour by the one in the dict.
                    if false : it means that if we consider a colour from the image close
                    to a colour from the "colours" dict, then it will replace the colour by the default color value.
                default_color : default color value that a pixel has to take
        """
        # make a copy to avoid to erase the original image
        img_copy = self.util_obj.to_rgb()

        # for each line we get the pixel value
        for i, line in enumerate(self.image):
            for j, pixel in enumerate(line):
                # Get only 3 first value corresponding to R,G,B
                pixel = [int(val) if val > 1.0 else int(val * 255) for val in
                         self.image[i][j]][:rgb_len]

                # if we want to show a specific colour
                if colour_mode:
                    # default value
                    img_copy[i][j] = default_colour

                    # for each colour we change the pixel value if we find the same colour
                    for colour in colours.values():
                        if sum([1 if abs(p - b) < epsilon else 0 for p, b in
                                zip(pixel, colour)]) == rgb_len:
                            img_copy[i][j] = colour

                # if we want to hide a colour by a default value
                else:
                    # default value
                    img_copy[i][j] = pixel

                    # for each recognized colour, we change it by the default value
                    for colour in colours.values():
                        if sum([1 if abs(p - b) < epsilon else 0 for p, b in
                                zip(pixel, colour)]) == rgb_len:
                            img_copy[i][j] = default_colour
        return img_copy

    def colour_pipeline(self, colours={}, epsilon=20, colour_mode=True,
                        default_colour=[0, 0, 0], rgb_len=3):
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
        if not bool(colours): colours = COLOURS[self.layout][
            self.image_extension]

        # get the image result from colour detection pre-process wanted
        image_res = self.colour_detection(colours, epsilon, rgb_len,
                                          colour_mode, default_colour)

        return image_res



# Metaclasse pour les processus
class MetaProcess(metaclass=ABCMeta):
    """
    Metaclasse pour la definition d'un processus. Permet de définir le
    comportement que doit avoir un processus pour fonctionner avec les
    pipeline de travail (classe Pipeline).
    """

    def check_attributes(self):
        """
        Attribut abstrait obligatoire permettant de définir une
        description du process effectué affiché si la verbosité est
        supérieure à 0.
        Cet attribut doit décrire ce qui est réalisée par le process
        et la librairie majoritairement utilisée pour réaliser ce
        process ainsi que la version de cette bibliothèque s'il y a.
        """
        if self.process_desc is None or self.process_desc == "":
            raise NotImplementedError("Définissez une description pour "
                                      + "le process.")

    def __init__(self, verbose=1,*args, **kwargs):
        self.verbose = verbose
        self.check_attributes()
        super().__init__()
        if self.verbose > 0:
            print(self.__class__.__base__.__name__ + " : ", end=' ')
            print(self.process_desc)

    @property
    @abstractmethod
    def process_desc(self):
        """Name of the process to be able to identify the process"""
        return self.process_desc

    @abstractmethod
    def run(self, image: Iterable, **kwargs) -> None:
        """
        Réalise le traitement à effectuer sur la liste d'images. Ne
        retourne rien. Les modifications sont effectuées directement sur
        les images dans la liste.
        :param image: image à traiter : objet array-like
        """


# Classe abstraite pour les processus de pré-traitement de données
class Preprocess(MetaProcess):
    """
    Classe abstraite de définition d'un processus de pré-traitement des
    données.
    Pour définir un pré-traitement, il est nécessaire de définir une
    classe héritant de cette classe (Preprocess). Il doit
    obligatoirement être implémenté la fonction run(self,images) et
    l'attribut process_desc (avec une valeur différente de None ou "").
    Ex :
    class Preprocess_exemple(Proprocess):
        process_desc = "OpenCV4.0 -> data augmentation : rotations"
    ...
    """

    process_desc = None

    def __init__(self, col_obj,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.col_obj = col_obj

    @overrides
    @abstractmethod
    def run(self, **kwargs) -> Iterable:
        pass

class ColourPipelineSeat(Preprocess):
    process_desc = "Standard Python >= 3.5 -> preprocess colours"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> Iterable:
        return self.col_obj.colour_pipeline(colours={}, epsilon=40,
                                            colour_mode=False,
                                            default_colour=[255, 255, 255],
                                            rgb_len=3)

class BlackWhite(Preprocess):
    process_desc = "OpenCV4.1.2.30 -> rgb to grey"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> Iterable:
        return self.col_obj.util_obj.to_gray()


class ColourPipelineZones(Preprocess):
    process_desc = "Standard Python >= 3.5 -> preprocess colours"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> Iterable:
        return self.col_obj.colour_pipeline(colours={}, epsilon=30,
                                            colour_mode=True,
                                            default_colour=[0, 0, 0],
                                            rgb_len=3).astype('uint8')


# Classe abstraite pour les processus de traitement de données
class Process(MetaProcess):
    """
    Classe abstraite de définition d'un processus de traitement des
    données.
    Pour définir un traitement, il est nécessaire de définir une
    classe héritant de cette classe (Process). Il doit
    obligatoirement être implémenté la fonction run(self,images) et
    l'attribut process_desc (avec une valeur différente de None ou "").
    Ex :
    class Process_exemple(Process):
        process_desc = "PyTesseract2.0-> recherche de caractères"
    ...
    """
    process_desc = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    @abstractmethod
    def run(self, image: Iterable, json: dict, **kwargs) -> None:
        pass

class SeatFinder(Process):
    process_desc = "OpenCV4.1.2.30 -> Pattern Matching seat"

    def __init__(self, csv_data_path=None, *args, **kwargs):
        if csv_data_path is None:
            raise Exception("Data_path is empty in init function of " + str(
                self.__class__))
        super().__init__()
        self.csv_data_path = csv_data_path
        self.layout_folder_path = csv_data_path + "ANALYSE IMAGE/"
        self.seatguru_image_data_path = self.layout_folder_path + "LAYOUT SEATGURU/"
        self.seatmaestro_image_data_path = self.layout_folder_path + "LAYOUT SEATMAESTRO/"

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def longestValue(self, inputList):
        if len(inputList) > 0:
            max_len = 0
            for i, el in enumerate(inputList):
                if len(el) > max_len:
                    max_len = i
            return inputList[max_len]
        return 0

    def get_relevant_aircraft_builders(self, image_names,
                                       proportion_min=0.02,
                                       proportion_max=0.75):
        # retrieve aircraft types

        aircraft_builders = np.unique(
            [a_type.split(' ')[0].lower() for a_type in
             pd.read_csv(
                 self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv',
                 sep=';')['Aircraft_Type'].tolist() if
             len(a_type.split(' ')) > 1])

        # remove aircraft_builder that have Numerics in it
        aircraft_builders = [a_builder for a_builder in aircraft_builders if
                             not self.hasNumbers(a_builder) and len(
                                 a_builder) > 2]

        # remove aircraft with too low occurences in the "IMAGES_NAMES" list
        relevant_aircraft_builders = {}
        for a_builder in aircraft_builders:
            proportion = sum(
                [1 for img_n in image_names if
                 a_builder in img_n.lower()]) / len(
                image_names)
            if proportion > proportion_min and proportion < proportion_max:
                relevant_aircraft_builders[a_builder] = proportion

        # remove non aircradft builder that remain
        del relevant_aircraft_builders['airlines']
        del relevant_aircraft_builders['irbus']
        del relevant_aircraft_builders['airways']
        del relevant_aircraft_builders['ays']

        # sort by proportion
        relevant_aircraft_builders = {k: v for k, v in
                                      sorted(
                                          relevant_aircraft_builders.items(),
                                          key=lambda item: item[1])}
        return relevant_aircraft_builders

    def get_airline_aircraft_builder_pos(self, image_names, aircraft_builders,
                                         airlines, aircraft_ref, pattern,
                                         layout="LAYOUT SEATGURU/"):
        # Initialisation of dict
        dictio_airlines_aircraft_builders = []

        for image_name in image_names:

            size = Image.open(
                self.layout_folder_path + layout + image_name).size

            dictio_airlines_aircraft_builders.append({
                'image_name': image_name,
                'aircraft_builder': 'not_relevant_aircraft_builders',
                'position_aircraft_builder': -1,
                'airline': '',
                'aircraft_ref': [],
                'x_size': int(size[0]),
                'y_size': int(size[1])
            })

            # a little pre-process to clean-up image name
            img_inf = image_name.lower().split('.')[0].split(pattern)

            for i, item in enumerate(img_inf):
                for a_builder in aircraft_builders:
                    # check if the image contains the aircraft builder
                    if a_builder == item:
                        # add the aircraft builder in the image name
                        dictio_airlines_aircraft_builders[-1][
                            'aircraft_builder'] = a_builder
                        dictio_airlines_aircraft_builders[-1][
                            'position_aircraft_builder'] = i

            # add airlines deduce by the image name
            for airline in airlines:
                if airline + pattern in image_name.lower():
                    dictio_airlines_aircraft_builders[-1][
                        'airline'] = airline

            # add aircraft_ref
            for a_ref in aircraft_ref:
                if a_ref in image_name.lower():
                    dictio_airlines_aircraft_builders[-1][
                        'aircraft_ref'].append(a_ref)
            # take the longest str element in the list of ref
            dictio_airlines_aircraft_builders[-1][
                'aircraft_ref'] = self.longestValue(
                dictio_airlines_aircraft_builders[-1]['aircraft_ref'])

        return dictio_airlines_aircraft_builders

    def main_layout_seatguru(self, layout="LAYOUT SEATGURU/"):
        image_name_list = [img for img in
                           os.listdir(self.layout_folder_path + layout)]

        relevant_aircraft_builders = self.get_relevant_aircraft_builders(
            image_name_list)
        airlines = \
            pd.read_csv(self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv',
                        sep=';')[
                'Airline_name'].unique()
        airlines = [airline.replace('-', '_') for airline in airlines]

        aircraft_ref = np.unique([a_type.lower() for a_type in pd.read_csv(
            self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv', sep=';')[
            'Aircraft_Type'].tolist()])
        aircraft_ref = np.unique(
            [t for text in aircraft_ref for t in text.split(' ') if
             self.hasNumbers(t)])

        dictio_airlines_aircraft_builders = self.get_airline_aircraft_builder_pos(
            image_name_list, relevant_aircraft_builders, airlines,
            aircraft_ref,
            '_')
        return pd.DataFrame(dictio_airlines_aircraft_builders)

    def get_correspondance(self, dataframe):
        # retrieve aircraft builders
        aircraft_builders = dataframe['aircraft_builder'].tolist()
        # retrieve arcraft references
        aircraft_refs = dataframe['aircraft_ref'].tolist()
        # retrieves airline names
        airlines = dataframe['airline'].tolist()
        # retrieve image names
        image_names = dataframe['image_name'].tolist()

        df_seat_guru_info = pd.read_csv(
            self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv', sep=';')
        aircraft_type = df_seat_guru_info['Aircraft_Type']
        # dictionnary where the info will be stored
        dictio = {}
        # for each aircraft builder, aircraft ref, airline and image name,
        # we research the relevant information in df_seat_guru_info dataframe
        for a_b, a_r, airline, im in \
                zip(aircraft_builders, aircraft_refs, airlines, image_names):
            res = pd.DataFrame([df_seat_guru_info.loc[i] for i, a_t in
                                enumerate(aircraft_type) if
                                a_b in a_t.lower() and a_r in a_t.lower()]).reset_index()
            if 'Airline_name' in res.columns:  # res not empty right after the research
                res = pd.DataFrame([res.loc[i] for i, a_n in
                                    enumerate(res['Airline_name'].tolist()) if
                                    airline.replace('_', '-') in a_n.lower()])
                if 'Total_seat' in res.columns:  # res not empty right after the research
                    # get the maximum of seat number
                    nb_max_seats = max(res['Total_seat'].unique().tolist())
                    # select the seats with the help of nb_max_seats
                    res['selected'] = res['Total_seat'].apply(
                        lambda x: 1 if x == nb_max_seats else 0)
                    # select the column that we need
                    res = res[['Category', 'Count', 'Seat_Type', 'selected']]
                    res = res[res[
                                  'selected'] == 1].drop_duplicates()  # remove duplicated lines
                    dictio[im] = res.drop(columns=['selected'])
        return dictio

    def retrieve_relevant_seat_info(self, dictio, image_name):
        total_seat_info = []
        for i, row in dictio[image_name].iterrows():
            total_seat_info.append({
                'Category': row['Category'],
                'Seat_Type': row['Seat_Type'],
                'Count': row['Count']
            })
        return total_seat_info

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
        res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

        for pos in zip(*np.where(res >= threshold)[::-1]):
            position.append(pos)
        return (position)

    def templ_category(self, path='./images/TEMPLATE/', category='BUSINESS',
                       seat_type='STANDARD', plane_name='test.jpg'):
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
        if '.png' in plane_name:
            extension = 'PNG/'
        else:
            extension = 'JPG/'

        imagesTemp = os.listdir(path + category + '/' + extension)
        templates = []

        for i in imagesTemp:
            if seat_type in i:
                templates.append(cv.imread(
                    path + category + '/' + extension + i, 0))
        return (templates)

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
            position = self.coord_pattern_finder( img, template,
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
        if len(result) < nbSeat * 1.1:
            return (result)
        return (result[:int(nbSeat * 1.1)])

    def run(self, image, json, image_name=None,
            layout=["LAYOUT SEATGURU/", "LAYOUT SEATMAESTRO/"],
            path='./images/', **kwargs):
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

        if not image_name in json.keys():
            json[image_name] = {}
        df_layout_seatguru = self.main_layout_seatguru(layout[0])
        dictio_correspondance = self.get_correspondance(
            df_layout_seatguru[df_layout_seatguru['image_name'] == image_name])
        nbObjectToFind = self.retrieve_relevant_seat_info(
            dictio_correspondance, image_name)

        #       df_layout_seatguru = self.main_layout_seatguru(layout)

        #      dictio = self.get_correspodance(df_layout_seatguru)

        #        nbOjbjectToFind = self.retrieve_relevant(dictio, im_names)

        for objet in nbObjectToFind:
            if not objet["Category"] in json[image_name].keys():
                json[image_name][objet['Category']] = []
            # Take all template name for this category
            templates = self.templ_category(
                category=objet['Category'],
                seat_type=objet['Seat_Type'],
                plane_name=image_name)
            for templ in templates:
                templateFind, find = self.template_from_template(image,
                                                                 templ)
                if find:
                    position = self.best_position(image, templateFind, objet['Count'])

                    h, w = templ.shape
                    for i in range(len(position)):
                        position[i] = position[i] + (h, w)
                    json[image_name][objet['Category']] += position


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
        #print(json[image_name])
        # Pre-process the image

        # Image rgb to gray

        dict_data = self.image_detection_result(image_name, image, 80)

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

                    if pos[0] > rect[1] and pos[0] < rect[3] and pos[1] > rect[
                        0] and pos[1] < rect[2]:

                        if rect not in liste_position:
                            liste_position.append(rect)

            type_temp[templ] = (liste_position)

            temp_rcgnzd[image_name] = type_temp

        json[image_name] = temp_rcgnzd[image_name]




    def run(self, image, json, image_rgb=None, col_obj=None, templates=None,
            data_image=None, image_name=None, **kwargs) -> None:
        plt.imshow(image)
        plt.show()
        self.label_results(image, json, data_image)
        temp_zone_fold_path = "./images/zone_templates/"
        list_temp = [name_template for name_template in
                     os.listdir(temp_zone_fold_path)]
        print(list_temp)
        self.coord_template_matching_image_single(image, json,
                                                  image_name=image_name,
                                                  liste_temp=list_temp,
                                                  path_temp=temp_zone_fold_path,
                                                  threshold=0.5)



# Classe abstraite pour les processus de post-traitement de données
class Postprocess(MetaProcess):
    """
    Classe abstraite de définition d'un processus de post-traitement des
    données.
    Pour définir un pré-traitement, il est nécessaire de définir une
    classe héritant de cette classe (Postprocess). Il doit
    obligatoirement être implémenté la fonction run(self,images) et
    l'attribut process_desc (avec une valeur différente de None ou "").
    Ex :
    class Preprocess_exemple(Proprocess):
        process_desc = "OpenCV4.0 -> alignement des predictions"
    ...
    """
    process_desc = None

    def __init__(self):
        super().__init__()

    @overrides
    @abstractmethod
    def run(self, image: Iterable, json: dict, **kwargs) -> None:
        super()
        pass


class RemoveDouble(Postprocess):
    process_desc = "Standard Python >= 3.5 -> remove double point in list"

    def __init__(self, *args, **kwargs):
        super().__init__()

    def remove_duplicate(self, coordinate: list):
        """Documentation
        Parameters:
            coordinate: original coordinates without treatment
        Out:
            dup: list of coordinate which are duplicated
        """
        dup = []
        print(coordinate)
        for point1 in coordinate:
            for point2 in coordinate:
                if point2 != point1 and point1 not in dup:
                    if ((abs(point1[0] - point2[0]) <= 5) and (
                            abs(point1[1] - point2[1]) <= 5)):
                        dup.append(point2)
        for d in dup:
            if d in coordinate:
                coordinate.remove(d)
        return (coordinate)

    def run(self, json, **kwargs):
        for seat_index in json:
            json[seat_index] = self.remove_duplicate(json[seat_index])


# Classe pour le pipeline
class Pipeline:
    """
    Classe permettant de définir un pipeline. Le pipeline execute dans
    l'ordre le pré_processing, le processing et le post_processing.
    Cette classe contient 3 numpy array contenant la liste des
    traitements pour chaque étapes à réaliser. Il faut utiliser les
    fonction add_pre_process
    """

    def __init__(self, data_path, list_images_name: Iterable[str] = None,
                 layouts: Iterable[str] =
                 ['LAYOUT SEATGURU', 'LAYOUT SEATMAESTRO']) -> None:
        self.pre_process: Iterable[type] = np.array([])
        self.process: Iterable[type] = np.array([])
        self.post_process: Iterable[type] = np.array([])
        self.json = {}
        self.list_images_name = list_images_name

        # definition of input path for images
        # data_path : path to Interpromo2020
        self.data_path = data_path
        # data_path : list of layouts folders names
        self.layouts = layouts
        self.csv_path = data_path + "All Data/"
        self.layout_folder_path = self.csv_path + "ANALYSE IMAGE/"
        self.image_folder_path = self.layout_folder_path + layouts[0] +"/"


    def add_processes(self, in_process: Iterable):
        """
        Ajoute une liste de process dans le pipeline.
        :param in_process: Iterable[MetaProcess] : Liste des process à
        ajouter au pipeline. Chaque
        """
        wrong_processes: tuple = ()
        for process in in_process:
            if not (Preprocess in process.__mro__
                    or Process in process.__mro__
                    or Postprocess in process.__mro__):
                if MetaProcess in process.__mro__:
                    wrong_process = ((process.process_desc,
                                      process.__class__,),)

                    wrong_processes = wrong_processes + wrong_process

                else:
                    wrong_processes = wrong_processes + ((type(process),),)
                    continue

            else:
                if Preprocess in process.__mro__:
                    self.pre_process = np.append(self.pre_process,
                                                 np.array([process]))
                    print(process.process_desc + " a été ajouté.")
                if Process in process.__mro__:
                    self.process = np.append(self.process,
                                             np.array([process]))
                    print(process.process_desc + " a été ajouté.")

                if Postprocess in process.__mro__:
                    self.post_process = np.append(self.post_process,
                                                  np.array([process]))
                    print(process.process_desc + " a été ajouté.")
        if len(wrong_processes) > 0:
            raise NotProcessClass("Autre chose que des process ont "
                                  "été ajoutés au pipeline.", wrong_processes)

    def print_process(self):
        """
        Affiche les processus qui seront executés permettant ainsi de
        voir l'ordre d'execution des traitements dans le pipeline.
        """
        for process in self.pre_process:
            print(process.process_desc)
        for process in self.process:
            print(process.process_desc)
        for process in self.post_process:
            print(process.process_desc)

    # Pas besoin de retourner les variables : on modifie directement les images
    def run_pipeline(self, nb_images: int, data_path = None,**kwargs) -> None:
        """
        Execute le pipeline. Il sera executé dans l'ordre
            - le pré-processing
            - le processing
            - le post-processing
        Chaque process est conservé dans une liste et chaque groupe de
        process sera executé dans l'ordre dans lequel les process ont
        été ajoutés dans la liste de traitements.
        Les images sont directement modifiés.
        :param images: objet array-like : contient la liste de images
        :return: None
        """
        self.print_process()
        if self.list_images_name is None :
            self.list_images_name = os.listdir(self.image_folder_path)[:nb_images]

        for image_name in self.list_images_name:
            # Create a Colour object
            col_obj = Colour(self.layout_folder_path, self.layouts[0], image_name)
            # Create a
            #util_obj = ImageUtil(self.data_path + self.layouts[0] + "/", image_name)

            plt.figure(figsize=(4,4))
            plt.imshow(col_obj.image)
            plt.show()
            print("Début du pipeline : ")
            image = None
            for num, pre_process in enumerate(self.pre_process):

                pre_pro = pre_process(col_obj)
                try:
                    # Instanciation du process
                    print("Doing : " + pre_pro.process_desc)
                    image = pre_pro.run(**kwargs)
                    col_obj.set_image(image)
                    plt.figure(figsize=(40,40))
                    plt.imshow(col_obj.image)
                    plt.show()
                except Exception as e:
                    traceback.print_tb(e.__traceback__)
                    print("Le pré-processing numéro " + str(num)
                          + "( " + pre_pro.process_desc
                          + " ) a levé une erreur.")
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[
                        1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(e)

            for num, process in enumerate(self.process):
                pro = process(csv_data_path=self.csv_path)
                try:
                    if image is not None:
                        print("Doing : " + pro.process_desc)
                        # data_image = le path de l'image
                        pro.run(image, self.json, image_rgb = col_obj.image,
                                data_image =self.image_folder_path + image_name,
                                image_name = image_name, **kwargs)
                    else:
                        raise ValueError("Image = None")
                except Exception as e:
                    traceback.print_tb(e.__traceback__)
                    print("Le processing numéro " + str(num)
                          + "( " + pro.process_desc + " ) a levé une erreur.")
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[
                        1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(e)

            for num, post_process in enumerate(self.post_process):
                post_pro = post_process()
                try:
                    print("Doing : " + post_pro.process_desc)
                    post_pro.run(self.json,image_name = image_name, **kwargs)
                except Exception as e:
                    traceback.print_tb(e.__traceback__)
                    print("Le processing numéro " + str(num)
                          + "( " + post_pro.process_desc
                          + " ) a levé une erreur.")
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[
                        1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(e)




class DistPipeline:

    def find_cluster(self, epsilon: int, min_sample: int, list_wo_dup: list):
        """Documentation
        Parameters:
            epsilon: the maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_sample: the number of samples in a neighborhood for a point to be considered as a core point
            list_wo_dup: list of seats coordinates not duplicated
        Out:
            dbscan: clustering result with DBSCAN
        """
        x_wo_dup = [a for a, b in list_wo_dup]
        y_wo_dup = [b for a, b in list_wo_dup]
        dbscan = DBSCAN(eps=epsilon, min_samples=min_sample).fit(list_wo_dup)
        plt.scatter(x_wo_dup, y_wo_dup, c=dbscan.labels_.astype(
            float), s=50, alpha=0.5)
        plt.show()
        return(dbscan)

    def clusters_to_rect(self, dbscan: sklearn.cluster.DBSCAN, array_wo_dup: np.array):
        """Documentation
        Parameters:
            dbscan: clustering result with DBSCAN
        Out:
            list_rect: list of rectangles representing each cluster
        """
        list_coord = array_wo_dup
        label_groups = pd.Series(dbscan.labels_).unique()
        list_rect = []  # to plot with plt.patches
        list_rect2 = []  # all corners of the rectangles
        HEIGHT: int = 30
        WIDTH: int = 20
        for group in label_groups:
            index = [i for i, x in enumerate(
                list(dbscan.labels_)) if x == group]
            points_cluster = list_coord[index]
            corner_bottom_right = (max(i[0] for i in points_cluster) + WIDTH, min(
                i[1] for i in points_cluster) - HEIGHT)
            corner_top_right = (max(i[0] for i in points_cluster) + WIDTH, max(
                i[1] for i in points_cluster))
            corner_top_left = (min(i[0] for i in points_cluster), max(
                i[1] for i in points_cluster))
            corner_bottom_left = (min(i[0] for i in points_cluster), min(
                i[1] for i in points_cluster) - HEIGHT)
            height = corner_top_right[1] - corner_bottom_right[1]
            width = corner_bottom_right[0] - corner_bottom_left[0]
            list_rect.append(((corner_bottom_left), width, height))
            list_rect2.append(
                (corner_bottom_left, corner_top_left, corner_top_right, corner_bottom_right))
        return list_rect, list_rect2

    def centroid_obstacle(self, coord_obs: list):
        """Documentation
        Parameters:
            coord_obs: cooardinate of the obstacle (list of tuple)
        Out:
            coord_bar_obs: barycenter cooardinate of the obstacle
        """
        A_point = coord_obs[0][1], coord_obs[0][0]
        B_point = coord_obs[1][1], coord_obs[1][0]
        coord_bar_obs = int(np.mean([A_point[0], B_point[0]])), int(
            np.mean([A_point[1], B_point[1]]))
        return coord_bar_obs

    def centroid_seat(self, coord_seat: tuple):
        """Documentation
        Parameters:
            coord_seat: cooardinate of the seat
        Out:
            coord_bar_seat: barycenter cooardinate of the seat
        """
        x, y = coord_seat[0], coord_seat[1]
        h, w = 30, 20
        coord_bar_seat = int(np.mean([x, x+w])), int(np.mean([y, y+h]))
        return coord_bar_seat

    def dist_crow_flies(self, coord_bar_seat: tuple, coord_bar_obs: tuple):
        """Documentation
        Parameters:
            coord_bar_seat: barycenter coordinate of the seat
            coord_bar_obs: barycenter cooardinate of the obstacle
        Out:
            dist: distance between the two barycenter
        """
        dist = np.sqrt(((coord_bar_obs[0]-coord_bar_seat[0])
                        ** 2)+((coord_bar_obs[1]-coord_bar_seat[1])**2))
        return round(dist, 2)

    def run_pipeline(self, json_seat: dict, json_zone: dict, HEIGHT: int = 30, WIDTH: int = 20):
        """Documentation
        Parameters:
            pipeline_zone.json: json ???
            pipeline.json: json ???
        Out:
            dicimg: json final structure
        """
        dicimg = {}
        for img in list(json_zone.keys()):
            dictypeseat = {}
            for typeseat in range(len(list(json_seat.keys()))):
                dicseat = {}
                for seat in list(json_seat.values())[typeseat]:
                    j = 0
                    dicobs = {}
                    for obs in json_zone[list(json_zone.keys())[0]]["rectangles"]:
                        j += 1
                        dicobs[("obstacle"+str(j))] = [self.centroid_obstacle([obs[0:2], obs[2:4]]),
                                                       self.dist_crow_flies(self.centroid_seat(seat), self.centroid_obstacle([obs[0:2], obs[2:4]]))]
                    dicseat[(self.centroid_seat(seat), WIDTH, HEIGHT)] = dicobs
                dictypeseat[list(json_seat.keys())[typeseat][5:len(list(json_seat.keys())[typeseat])-4]] = dicseat
            dicimg[img] = dictypeseat
        return dicimg



# pipeline_zone = Pipeline("/data/dataset/projetinterpromo/Interpromo2020/",["Aer_Lingus_Airbus_A330-300_A_plane6.jpg"])
#
# pipeline_zone.add_processes([BlackWhite,SeatFinder, RemoveDouble])
# pipeline_zone.run_pipeline(1)
# print(pipeline_zone.json)
#
#
#
#









#
#
#
#
#
#
# class Distpipeline:
#
#     def __init__(self, seat_pipeline, element_pipeline):
#         self.seat_pipeline = seat_pipeline
#         self.element_pipeline = element_pipeline
#
#     def run_output(self, json_seat, json_elements) -> dict:
#         pass
#









''' 

# Fonction de test : à mettre en place si besoin 

    def run_pipeline_with_copy(self, images: Iterable[Iterable]) -> None:
        try:
            images_copy = images.copy()
        except AttributeError:
            print("L'objet utilisé ne possède pas de fonction \"copy()"
                  "\".Il n'est pas possible d'utiliser le pipeline avec "
                  "copy.")
            raise
        for preprocess in self.pre_process:
            preprocess.run(images_copy)
        for process in self.process:
            process.run(images)
        for post_process in self.post_process:
            post_process.run(images)
'''

''' 

# Exemple d'utilisation du pipeline.
if __name__ == "__main__":
    class Augmentation(Preprocess):
        process_desc = "OpenCV4.0 -> data augmentation"

        def run(self, images):
            print("Pre_processing...")


    class PyTesseract(Process):
        process_desc = "PyTesseract3.8.0 -> recherche caractères"

        def run(self, images):
            print("Processing...")


    class Alignement(Postprocess):
        process_desc = "Pandas -> alignement predictions"

        def run(self, images):
            print("Post_processing...")


    class Wrong_Process(MetaProcess):
        pass


    class Coucou_process(Wrong_Process):
        process_desc = "Errortest"

        def run(self, images):
            print("Error")


    pipeline = Pipeline()

    pipeline.add_processes([Augmentation(), "", Coucou_process()])
    pipeline.add_processes([PyTesseract()])
    pipeline.add_processes([Alignement()])

    pipeline.run_pipeline([])
    print(Augmentation.run.__doc__)
    
    
    
    
    
    
    
                    
class PatternMatchingNormalSeat(Process):
    process_desc = "OpenCV4.1.2.30 -> pattern matching normal seat"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def coord_pattern_finder(self,image, template, threshold: float):
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
        res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        for pos in zip(*np.where(res >= threshold)[::-1]):
            position.append(pos)
        return(position)
    
    
    def run(self, image, json, image_rgb = None, col_obj = None, templates = None, data_image=None, **kwargs) -> None :
        #col_obj = Colour(data_path, layouts[0], p)
        #col_obj = Colour(data_image[0], data_image[1][0], data_image[2])
        
    # Make a colour detection based on the layout type ('GURU' or 'MAESTRO') and image type('png', 'jpg')
        img_to_show = image

        img_rgb = image_rgb
        img_gray = image
#         img_rgb = cv.imread(data_image, 1)
        img_gray = cv.imread(data_image, 0)
        print(data_image + ' shape', img_gray.shape)

        for template_name in templates:
            #template = cv.imread(templates_path + template_name, 0)
            
            template = cv.imread("./images/" + template_name, 0)
            if template is not None:
                h, w = template.shape
                print(template_name, h,w,img_gray.shape)

                threshold = 0.8 #  Default threshold
                # Choose threshold for each template
                if template_name == 'temp_business_seat_armrest_left.jpg':
                    threshold = 0.6
                elif template_name == 'temp_business_seat_armrest_right.jpg':
                    threshold = 0.6
                elif template_name == 'temp_business_seat_footrest.jpg':
                    threshold = 1
                elif template_name == 'temp_first_seat.jpg':
                    threshold = 1
                elif template_name == 'temp_normal_seat.png':
                    threshold = 0.7
                elif template_name == 'temp_normal_seat_elec.jpg':
                    threshold = 1
                elif template_name == 'temp_normal_seat_little.jpg':
                    threshold = 0.65
                elif template_name == 'temp_yellow_seat.jpg':
                    threshold = 1

                position = self.coord_pattern_finder(img_gray, template, threshold)

                for pos in position:
                    img_to_show = cv.rectangle(img_to_show, pos, (pos[0] + w, pos[1] + h), (0, 0, 255), 2)
                json[template_name] = position
        plt.figure(figsize=(20, 40))
        plt.imshow(img_to_show.astype('uint8'))
        print(json)
'''


#
# class RemoveDouble(Postprocess):
#     process_desc = "Standard Python >= 3.5 -> remove double point in list"
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#     def remove_duplicate(self,coordinate: list):
#         """Documentation
#         Parameters:
#             coordinate: original coordinates without treatment
#         Out:
#             dup: list of coordinate which are duplicated
#         """
#         dup = []
#         print(coordinate)
#         for point1 in coordinate:
#             for point2 in coordinate:
#                 if point2 != point1 and point1 not in dup:
#                     if ((abs(point1[0] - point2[0]) <= 5) and (abs(point1[1] - point2[1]) <= 5)):
#                         dup.append(point2)
#         for d in dup:
#             if d in coordinate:
#                 coordinate.remove(d)
#         return(coordinate)
#
#     def run(self, json, **kwargs):
#         for seat_index in json:
#             json[seat_index] = self.remove_duplicate(json[seat_index])
#
#
# class ColourPipelineSeat(Preprocess):
#     process_desc = "Standard Python >= 3.5 -> preprocess colours"
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def run(self, **kwargs) -> Iterable:
#         return self.col_obj.colour_pipeline(colours={}, epsilon=40,
#                                             colour_mode=False,
#                                             default_colour=[255, 255, 255],
#                                             rgb_len=3)
#
#
# class BlackWhite(Preprocess):
#     process_desc = "OpenCV4.1.2.30 -> rgb to grey"
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def run(self, **kwargs) -> Iterable:
#         return self.col_obj.util_obj.to_gray()
#
#
# class ColourPipelineZones(Preprocess):
#     process_desc = "Standard Python >= 3.5 -> preprocess colours"
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def run(self, **kwargs) -> Iterable:
#         return self.col_obj.colour_pipeline(colours={}, epsilon=30,
#                                             colour_mode=True,
#                                             default_colour=[0, 0, 0],
#                                             rgb_len=3)
#
#
# pipeline = Pipeline("/data/dataset/projetinterpromo/Interpromo2020/","Aer_Lingus_Airbus_A330-300_A_plane6.jpg")
# pipeline.add_processes([BlackWhite,SeatFinder, RemoveDouble])
# print(pipeline.json)
# pipeline.run_pipeline(1, planes_data_csv=None, plane_name="Aer_Lingus_Airbus_A330-300_A_plane6.jpg", csv_data_path="/data/dataset/projetinterpromo/Interpromo2020")
#
# class ColourPipelineSeat(Preprocess):
#     process_desc = "Standard Python >= 3.5 -> preprocess colours"
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def run(self, **kwargs) -> Iterable:
#         return self.col_obj.colour_pipeline(colours={}, epsilon=40,
#                                             colour_mode=False,
#                                             default_colour=[255, 255, 255],
#                                             rgb_len=3)
#
#
# class BlackWhite(Preprocess):
#     process_desc = "OpenCV4.1.2.30 -> rgb to grey"
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def run(self, **kwargs) -> Iterable:
#         return self.col_obj.util_obj.to_gray()
#
#
# class ColourPipelineZones(Preprocess):
#     process_desc = "Standard Python >= 3.5 -> preprocess colours"
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def run(self, **kwargs) -> Iterable:
#         return self.col_obj.colour_pipeline(colours={}, epsilon=30,
#                                             colour_mode=True,
#                                             default_colour=[0, 0, 0],
#                                             rgb_len=3).astype('uint8')
