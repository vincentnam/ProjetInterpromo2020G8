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
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from matplotlib import patches

from skimage.segmentation import clear_border

from skimage.filters import threshold_otsu

from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import sklearn
from sklearn.cluster import DBSCAN


def overrides(method):
    """
    Decorator implementation for overriding
    Come frome : https://stackoverflow.com/questions/1167617/
    in-python-how-do-i-indicate-im-overriding-a-method
    :param method: the method to overrides
    :return: the proper version of the method
    """
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
                assert (inspect.ismodule(obj) or inspect.isclass(obj))
                obj = getattr(obj, c)

            base_classes[i] = obj

    assert (any(hasattr(cls, method.__name__) for cls in base_classes))
    return method


class NotProcessClass(Exception):
    def __init__(self, expression, message):
        """
        Exception class to handle problem of object insertion in
        pipeline
        Attributes:
            process_desc -- process descrition to print
            process_class -- process class to print
        """
        self.expression = expression
        self.message = message


# Global variable : colors dictionnary for layout : used by colors
# preprocess - to determine wich colors has to be kept has element
# color in the image
COLOURS = {
    'LAYOUT SEATGURU': {
        'jpg': {
            "blue": [139, 168, 198],
            "yellow": [247, 237, 86],
            "exit": [222, 111, 100],
            "green": [89, 185, 71],
            "red_bad_seat": [244, 121, 123],
            "blue_seat_crew": [140, 169, 202]
        },
        'png': {
            "blue": [41, 182, 209],
            "yellow": [251, 200, 2],
            "exit": [190, 190, 190],
            "green": [41, 209, 135],
            "red_bad_seat": [226, 96, 82],
            "blue_seat_crew": [41, 182, 209]
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


class ImageUtil:
    """
    Tool class for preprocessing : contains method for image
    transformation as rgb to gray
    """

    def __init__(self, input_path, image_name, image=None):
        """
        Constructor for ImageUtil class ; used in colour class as a
        class attribute. Image is open only if image is None.
        :param input_path: str : path to the image to load
        :param image_name: str : image name (file name)
        :param image: Iterable : image loaded
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
    def set_image(self, image):
        """
        Setter for image parameter
        :param image: image to set
        """
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


class Colour:
    """
    Tool class : used to make transformations over images
    """

    def __init__(self, csv_data_path, layout, image_name):
        """
        Constructor for Colour class ; tool box for image preprocessing.
        Can be extended to add preprocesses easily.
        :param csv_data_path: str : input path to the base folder
        containing the csv. Folder architecture is based on the archive
        given at the project beginning (i.e. "ProjetInterpromo2020"
        containing sub folder :  "All\ Data" -> "ANALYSE\ IMAGE/" and
        sub folder with the layout from website.
        :param layout: Iterable[str] : list of layout to process /
        list of str that are folders name.
        :param image_name: str : file name of the image to work on
        """

        # Path of the base folder containing all sub folder : sub-folder
        # created after unzipping the archive containing all CSV
        self.csv_data_path = csv_data_path
        # Path of the folder containing layout folder :
        # sub folder of "ALL DATA"
        self.layout_folder_path = csv_data_path + "ANALYSE IMAGE/"
        # Path to the seatguru folder containing all layouts
        self.seatguru_image_data_path = self.layout_folder_path \
                                        + "LAYOUT SEATGURU/"
        # Path to the seatmaestro folder containing all layouts
        self.seatmaestro_image_data_path = self.layout_folder_path \
                                           + "LAYOUT SEATMAESTRO/"

        # Redundant variable : only used to keep function calling
        self.input_path = self.csv_data_path
        # List of layout
        self.layout = layout
        # File image name preprocessed
        self.image_name = image_name
        # File image extension : images graphic charts are
        # differentiated by extension in our dataset.
        self.image_extension = image_name.split('.')[-1]
        # Reading of the image with matplotlib
        self.image = plt.imread(
            self.input_path + self.layout + '/' + self.image_name)
        # Creating an ImageUtil object used for preprocessing
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
            Call colour_detection function in order to pre-process
            colours in image.
            params :
                colours : dict : a dictionnary with a list of specified colours
                epsilon : int : threshold that allows to consider a colour
                from another one as close
                rgb_len : List :  only take the 3 first elements from pixel
                (RGB norm)
                colour_mode : bool :
                    - if true (highlight colours in "colours" dict by standardize it) : it means that
                    if we consider a colour from the image close to a colour from the "colours" dict,
                    then it will replace the colour by the one in the dict.
                    - if false (remove colours in "colours" dict by the default one) : it means that
                    if we consider a colour from the image close to a colour from the "colours" dict,
                    then it will replace the colour by the default color value.
                default_color : default color value that a pixel has to take
        """
        # if colours is empty we take the default value
        if not bool(colours) : colours = COLOURS[self.layout][
            self.image_extension]

        # get the image result from colour detection pre-process wanted
        image_res = self.colour_detection(colours, epsilon, rgb_len,
                                          colour_mode, default_colour)

        return image_res


class MetaProcess(metaclass=ABCMeta):
    """
    Metaclass for process definition. Used to define a process behaviour
    to be able to make a pipeline of processes. (Cf. Pipeline class)
    """

    def check_attributes(self):
        """
        Check if attributes is defined and not empty. Raise an error
        if not defined.
        """
        if self.process_desc is None or self.process_desc == "":
            raise NotImplementedError("Définissez une description pour "
                                      + "le process.")

    def __init__(self, verbose=1, *args, **kwargs):
        """
        MetaProcess constructor. Check if process_desc is implemented.
        :param verbose: int : >0 implies a printing of process_desc
        when called.
        """
        self.verbose = verbose
        self.check_attributes()
        super().__init__()
        if self.verbose > 0:
            print(self.__class__.__base__.__name__ + " : ", end=' ')
            print(self.process_desc)

    @property
    @abstractmethod
    def process_desc(self):
        """
        Abstract attribute defining a process definition. The process
        description printing can be avoid by setting "verbose=0".
        This attribute is used to describe the computation and the
        major library used in the run (and the version).
        """
        return self.process_desc

    @abstractmethod
    def run(self, image: Iterable, **kwargs) -> None:
        """
        Run function and do the computation of the class. This function
        is used in pipeline and only this is launched in pipeline. Work
        as a main and parameters are filled with **kwargs dictionnary.
        :param image: image to process : objet array-like
        """


class Preprocess(MetaProcess):
    """
    Abstract class to define a preprocess.
    To define a new proprocess, it is needed to define process
    description and run function has to be implemented as the main
    of the process.
    The process_desc is used to describe libraries and the process
    computations.
    Ex :
    class Preprocess_exemple(Proprocess):
        process_desc = "OpenCV4.1.2.30 -> data augmentation : rotations"

        def run(self, image, **kwargs):
            ...
    """

    process_desc = None

    def __init__(self, col_obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Colour object containing the image to preprocess and
        # may be already preprocessed.
        self.col_obj = col_obj

    @overrides
    @abstractmethod
    def run(self, **kwargs) -> Iterable:
        """
        Image is open in col_obj and preprocess are done on this image.
        The col_obj is the same during all the pipeline and all
        preprocess compute on this image.
        :return : Iterable : image preprocessed to keep changes and set
        the new image in the col_obj.
        """
        pass


class ColourPipelineSeat(Preprocess):
    """
    Pre process class for color preprocessing. Is used the col_obj to
    transform colours of an image.
    """
    process_desc = "Standard Python >= 3.5 -> preprocess colours"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> Iterable:
        return self.col_obj.colour_pipeline(colours={}, epsilon=40,
                                            colour_mode=False,
                                            default_colour=[255, 255, 255],
                                            rgb_len=3)


class BlackWhite(Preprocess):
    """
    Transform colored image into a grey scale image.
    """
    process_desc = "OpenCV4.1.2.30 -> rgb to grey"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> Iterable:
        return self.col_obj.util_obj.to_gray()


class ColourPipelineZones(Preprocess):
    """
    Transform image colors.
    """
    process_desc = "Standard Python >= 3.5 -> preprocess colours"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> Iterable:
        return self.col_obj.colour_pipeline(colours={}, epsilon=30,
                                            colour_mode=True,
                                            default_colour=[0, 0, 0],
                                            rgb_len=3).astype('uint8')


class Process(MetaProcess):
    """
    Abstract class to define a preprocess.
    To define a new process, it is needed to define process
    description and run function has to be implemented as the main
    of the process.
    The process_desc is used to describe libraries and the process
    computations.
    Ex :
    class Process_exemple(Process):
        process_desc = "OpenCV4.2.1.30-> Pattern matching seat"
        def run():
            ...
    ...
    """
    process_desc = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    @abstractmethod
    def run(self, image: Iterable, json: dict, **kwargs) -> None:
        """

        :param image: Iterable : image preprocessed
        :param json:  dict : input/output parameter : the dictionnary
        to fill with informations
        """
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
            image : image plane cv.imread() black and white
            template : image pattern cv.imread() black and white
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
        Create list of template open with cv by category and seatType
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
            position = self.coord_pattern_finder(img, template,
                                                 threshold)

        if threshold > thresholdMin:
            return (img[position[0][1]:position[0][1] + h,
                    position[0][0]:position[0][0] + w], True)
        return (None, False)

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
                    position = self.best_position(image, templateFind,
                                                  objet['Count'])

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
        # print(json[image_name])
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

                    if rect[1] < pos[0] < rect[3] \
                            and rect[0] < pos[1] < rect[2]:

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


class Postprocess(MetaProcess):
    """
    Abstract class to define a postprocess.
    To define a new post-process, it is needed to define process
    description and run function has to be implemented as the main
    of the process.
    The process_desc is used to describe libraries and the process
    computations.
    Ex :
    class Process_exemple(Process):
        process_desc = "Standard python >3.5 -> Remove seat detected
        multiple times"
        def run():
            ...
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


class RemoveDoubleSeat(Postprocess):
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
        dup = {}

        for category in coordinate:
            dup[category] = []
            for point1 in coordinate[category]:
                for point2 in coordinate[category]:
                    if point2 != point1 and point1 not in dup:
                        if ((abs(point1[0] - point2[0]) <= 5) and (
                                abs(point1[1] - point2[1]) <= 5)):
                            dup[category].append(point2)
        for d in dup:
            for category in coordinate:
                if d in coordinate[category]:
                    coordinate.remove(d)

        return coordinate

    def run(self, json, **kwargs):
        for seat_index in json:
            json[seat_index] = self.remove_duplicate(json[seat_index])


# Classe pour le pipeline
class Pipeline:
    """
    Pipeline class : define the process order (pre-process -> process
    -> post process) and informations exchanges between processes.
    To add a process, add_processes take a list of processes even if
    this list contain only 1 process.
    """

    def __init__(self, data_path, list_images_name: Iterable[str] = None,
                 layouts: Iterable[str] =
                 ['LAYOUT SEATGURU', 'LAYOUT SEATMAESTRO']) -> None:
        """
        Pipeline constructor
        :param data_path: str : path to the base folder directly
        extracted from the .zip archive
        :param list_images_name: Iterable[str] : list of files images
        to make the pipeline on.
        :param layouts: Iterable[str] : list of layout folder to
        consider
        """
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
        self.image_folder_path = self.layout_folder_path + layouts[0] + "/"

    def add_processes(self, in_process: Iterable):
        """
        Add a list of processes in the pipeline. Processes are run in
        the same order that they are processed
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
        Print process that are in the pipeline. The order of printing is
        the same as the running order.
        """
        for process in self.pre_process:
            print(process.process_desc)
        for process in self.process:
            print(process.process_desc)
        for process in self.post_process:
            print(process.process_desc)

    def print_traceback(self, proc : MetaProcess, num: int, e: Exception) -> None :
        traceback.print_tb(e.__traceback__)
        print("Proprocess number " + str(num)
              + "( " + proc.process_desc
              + " ) raised an error.")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[
            1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)

    def run_pipeline(self, nb_images: int, verbose=True, **kwargs) -> None:
        """
        Run the pipeline. Compute, in order :
            - preprocessing
            - processing
            - postprocessing
        Each process is kept in a list for each type of process. Process
        are run in the order they are put in the list.
        Run the computations over 1 image at a time and are directly
        modified.
        The output are saved in the json class argument.
        :param nb_images: int : number of images to compute the pipeline
        on.
        :param verbose : bool : set visualisation on or off. Image are
        plot with matplotlib and process list are printed if
        verbose = True.
        :param **kwargs : allow argument passing by this dictionnary. It
        is used to give parameters to process in the run. Don't forget
        to name parameter the same as in the process definition.
        """
        if verbose is True :
            self.print_process()
        if self.list_images_name is None:
            self.list_images_name = os.listdir(self.image_folder_path)[
                                    :nb_images]

        for image_name in self.list_images_name:
            # Create a Colour object containing the image to process.
            col_obj = Colour(self.layout_folder_path, self.layouts[0],
                             image_name)
            if verbose is True:
                plt.figure(figsize=(4, 4))
                plt.imshow(col_obj.image)
                plt.show()
            print("Pipeline start : ")
            image = None
            for num, pre_process in enumerate(self.pre_process):

                pre_pro = pre_process(col_obj)
                try:
                    # Process instantiation
                    print("Doing : " + pre_pro.process_desc)
                    image = pre_pro.run(**kwargs)
                    col_obj.set_image(image)
                    if verbose is True:
                        plt.figure(figsize=(40, 40))
                        plt.imshow(col_obj.image)
                        plt.show()
                except Exception as e:
                    self.print_traceback(pre_pro, num, e)

            for num, process in enumerate(self.process):
                pro = process(csv_data_path=self.csv_path)
                try:
                    if image is not None:
                        print("Doing : " + pro.process_desc)
                        # data_image = le path de l'image
                        pro.run(image, self.json, image_rgb=col_obj.image,
                                data_image=self.image_folder_path + image_name,
                                image_name=image_name, **kwargs)
                        if verbose is True:
                            print(pipeline.json)
                    else:
                        raise ValueError("Image = None")
                except Exception as e:
                    self.print_traceback(pro, num, e)

            for num, post_process in enumerate(self.post_process):
                post_pro = post_process()
                try:
                    print("Doing : " + post_pro.process_desc)
                    post_pro.run(self.json, image_name=image_name, **kwargs)
                except Exception as e:
                    self.print_traceback(post_pro, num, e)

class DistPipeline():

    def __init__(self, pipeline, pipeline_zone):
        self.pipeline = pipeline
        self.pipeline_zone = pipeline_zone

    def change_format(self, dict_seat: dict):
        """Documentation
        Parameters:
            epsilon: the maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_sample: the number of samples in a neighborhood for a point to be considered as a core point
            list_wo_dup: list of seats coordinates not duplicated
        Out:
            list_wo_dup: list of seats coordinates
            height_width: height and width
        """
        list_wo_dup = []
        height_width = []
        for i in list(dict_seat.values()):
            for j in i:
                list_wo_dup.append((j[0], j[1]))
                height_width.append((j[2], j[3]))
        return list_wo_dup, height_width

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
        return (dbscan)

    def clusters_to_rect(self, dbscan: sklearn.cluster.dbscan_.DBSCAN,
                         array_wo_dup: np.array):
        """Documentation
        Parameters:
            dbscan: clustering result with DBSCAN
        Out:
            list_rect: list of rectangles representing each cluster
            list_rect2: list of rectangles representing each cluster
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
            corner_bottom_right = (
            max(i[0] for i in points_cluster) + WIDTH, min(
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
                (corner_bottom_left, corner_top_left, corner_top_right,
                 corner_bottom_right))
        return list_rect, list_rect2

    def centroid_obstacle(self, coord_obs: list):
        """Documentation
        Parameters:
            coord_obs: cooardinate of the obstacle (top left and bottom right)
        Out:
            coord_bar_obs: barycenter cooardinate of the obstacle
        """
        A_point = coord_obs[1], coord_obs[0]
        B_point = coord_obs[3], coord_obs[2]
        return int(np.mean([A_point[0], B_point[0]])), int(
            np.mean([A_point[1], B_point[1]]))

    def centroid_seat(self, coord_seat: tuple):
        """Documentation
        Parameters:
            coord_seat: cooardinate of the seat
        Out:
            coord_bar_seat: barycenter cooardinate of the seat
        """
        x, y = coord_seat[0], coord_seat[1]
        h, w = coord_seat[2], coord_seat[3]
        return (int(x + w / 2), int(y + h / 2))

    def dist_crow_flies(self, coord_bar_seat: tuple, coord_bar_obs: tuple):
        """Documentation
        Parameters:
            coord_bar_seat: barycenter coordinate of the seat
            coord_bar_obs: barycenter cooardinate of the obstacle
        Out:
            dist: distance between the two barycenter
        """
        dist = np.sqrt(((coord_bar_obs[0] - coord_bar_seat[0])
                        ** 2) + ((coord_bar_obs[1] - coord_bar_seat[1]) ** 2))
        return round(dist, 2)

    def AStarSearch(self, start: tuple, end: tuple, graph: AStarGraph):
        """Documentation
        A* algorithm to find the best path for from one point to another
        Parameters:
            start: Point of the start for the A* algorithm
            end: Point of the end for the A* algorithm
            graph: Graph for the execution of the A* algorithm
        Out:
            path: All points of the best path
            F[end]: Cost of the best path
        """

        G: dict = {}  # Actual movement cost to each position from the start position
        F: dict = {}  # Estimated movement cost of start to end going via this position
        # Initialize starting values
        G[start] = 0
        F[start] = graph.heuristic(start, end)  ###appeler class
        closedVertices: set = set()
        openVertices: set = set([start])
        cameFrom: dict = {}
        while len(openVertices) > 0:
            # Get the vertex in the open list with the lowest F score
            current = None
            currentFscore = None
            for pos in openVertices:
                if current is None or F[pos] < currentFscore:
                    currentFscore = F[pos]
                    current = pos
            # Check if we have reached the goal
            if current == end:
                # Retrace our route backward
                path = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    path.append(current)
                path.reverse()
                return path, F[end]  # Done!
            # Mark the current vertex as closed
            openVertices.remove(current)
            closedVertices.add(current)
            # Update scores for vertices near the current position
            for neighbour in graph.get_vertex_neighbours(current):
                if neighbour in closedVertices:
                    continue  # We have already processed this node exhaustively
                candidateG = G[current] + graph.move_cost(current, neighbour)
                if neighbour not in openVertices:
                    openVertices.add(neighbour)  # Discovered a new vertex
                elif candidateG >= G[neighbour]:
                    continue  # This G score is worse than previously found
                # Adopt this G score
                cameFrom[neighbour] = current
                G[neighbour] = candidateG
                H = graph.heuristic(neighbour, end)
                F[neighbour] = G[neighbour] + H

    def create_barriers_obs(self, coord_obstacle: iter, goal: tuple):
        """Documentation
        Return a list of lists representing the different obstacles
        Parameters:
            coord_obstacle: coordinates of the corners of each obstacles
        Out:
            list_barriers: List of lists representing the different obstacles with all their points
        """
        list_barriers: iter = []
        for coord in coord_obstacle:
            list_temp_1 = []
            list_temp_2 = []
            list_temp_3 = []
            list_temp_4 = []
            x_range = abs(coord[3] - coord[1])
            y_range = abs(coord[2] - coord[0])
            for x in range(x_range):
                list_temp_2.append((coord[1] + x, coord[2]))
                list_temp_4.append((coord[3] - x, coord[0]))
            for y in range(y_range):
                if coord[0] + y != goal[1]:
                    list_temp_1.append((coord[1], coord[0] + y))
                if coord[2] - y != goal[1]:
                    list_temp_3.append((coord[3], coord[2] - y))
            list_barriers.append(
                list_temp_1 + list_temp_2 + list_temp_3 + list_temp_4)
        return list_barriers

    def create_barriers_seat(self, corners_rect: iter, start: tuple):
        """Documentation
        Return a list of lists representing the different cluster of the seats
        Parameters:
            corners_rect: corners of the clusters representing the seats
        Out:
            list_corners: List of lists representing the different cluster with all the points of the outline
        """
        list_points = []
        for corners in corners_rect:
            x_range = corners[-1][0] - corners[0][0]
            y_range = corners[1][1] - corners[0][1]
            list_temp_1 = []
            list_temp_2 = []
            for x in range(x_range):
                list_temp_1.append((corners[1][0] + x, corners[0][1]))
                list_temp_2.append((corners[3][0] - x, corners[2][1]))
            list_temp_3 = []
            list_temp_4 = []
            for y in range(y_range):
                if corners[0][1] + y != start[1]:
                    list_temp_3.append((corners[0][0], corners[0][1] + y))
                if corners[2][1] - y != start[1]:
                    list_temp_4.append((corners[2][0], corners[2][1] - y))
            list_points.append(
                list_temp_1 + list_temp_2 + list_temp_3 + list_temp_4)
        return list_points

    def plane_contours(self, barriers_obs: iter, barriers_seat: iter):
        """Documentation
        Parameters:
            barriers_obs: List of lists representing the different obstacles with all their points
            barriers_seat: List of lists representing the different cluster with all the points of the outline
        Out:
            list_contours: List of the points of the outline representing the outline of the plane
        """
        x_min = y_min = np.inf
        x_max = y_max = -np.inf
        for barrier in barriers_seat:
            for point in barrier:
                if point[0] < x_min:
                    x_min = point[0]
                if point[0] > x_max:
                    x_max = point[0]
        for barrier in barriers_obs:
            for point in barrier:
                if point[1] < y_min:
                    y_min = point[1]
                if point[1] > y_max:
                    y_max = point[1]
        x_range = x_max - x_min
        y_range = y_max - y_min
        list_contours = []
        list_temp_1 = []
        list_temp_2 = []
        list_temp_3 = []
        list_temp_4 = []
        for y in range(y_range):
            list_temp_1.append((x_min, y_min + y))
            list_temp_3.append((x_max, y_max - y))
        for x in range(x_range):
            list_temp_2.append((x_min + x, y_max))
            list_temp_4.append((x_max - x, y_min))
        list_contours = list_temp_1 + list_temp_2 + list_temp_3 + list_temp_4
        return list_contours

    def pathfinder(self, start: tuple, goal: tuple, list_rect2: iter,
                   obstacles: iter):
        """Documentation
        Create the graph for the A* algorithm and calculate the best path
        Parameters:
            start: Start point for the A* algorithm
            goal: End point for the A* algorithm
            list_rect2: List of the corners of the cluster of the seat
            obstacles: List of the coordinates of the obstacles
        Out:
            path: Points of the best path
            cost: Cost of the path in pixel
        """
        barriers_seat = self.create_barriers_seat(list_rect2, start)
        barriers_obs = self.create_barriers_obs(obstacles, goal)
        outline = self.plane_contours(barriers_obs, barriers_seat)
        barriers = barriers_seat + barriers_obs + [outline]
        graph = AStarGraph(barriers)
        path, cost = self.AStarSearch(start, goal, graph)
        plt.figure(figsize=(40, 40))
        plt.plot([v[0] for v in path], [v[1] for v in path])
        for barrier in graph.barriers:
            plt.plot([v[0] for v in barrier], [v[1] for v in barrier],
                     color='red')
        plt.xlim(100, 400)
        plt.ylim(0, 1400)
        plt.show()
        return path, cost

    def draw_path(self, path: str, img: str, obs_number: int, seat_number: int,
                  obstacle: list, json_seat: dict):
        """Documentation
        Parameters:
            path: folder path
            img: image name
            obs_number: observation number
            seat_number: seat number
            obstacle: obstacles list, for each obstacle : oordinates of the top left and bottom right points
            json_seat: json
        """
        list_seat = []
        for i in list(json_seat.values()):
            list_seat += i

        dbscan = self.find_cluster(38, 3, list_seat)
        list_rect, list_rect2 = self.clusters_to_rect(
            dbscan, np.array(list_seat))

        fig = plt.figure(figsize=(20, 40))
        ax = fig.add_subplot(111, aspect='equal')

        for rect in list_rect:
            ax.add_patch(
                patches.Rectangle(rect[0], rect[1], rect[2]))

        img_cv = cv.imread(path + img)
        #
        for obs in obstacle:
            A_point = obs[1], obs[0]
            B_point = obs[3], obs[2]

            img_cv = cv.rectangle(img_cv, A_point, B_point, (255, 0, 0), 2)

        ob = list(self.pipeline_zone.json.values())[0]['rectangles'][obs_number]
        t_obs = [(ob[0], ob[1]), (ob[2], ob[3])]

        img_cv = cv.line(img_cv, self.centroid_seat(
            list_seat[seat_number]), self.centroid_obstacle(t_obs),
                          (255, 255, 0), 2)
        plt.imshow(img_cv)
        plt.show()

    def to_json_simple_distance(self, json_seat: dict, json_zone: dict):
        """Documentation
        Parameters:
            pipeline_zone.json: json ???
            pipeline.json: json ???
        Out:
            dicimg: json final structure
        """

        dicimg = {}

        # for each image in the json
        for img in list(json_zone.keys()):
            dicimg[img] = {}
            # for each type seat
            for typeseat in json_seat[img].keys():

                dicimg[img][typeseat] = {}
                # for each coordinate in a type seat
                for coord_seat in json_seat[img][typeseat]:
                    # get the centroid position of the seat
                    coord_centroid_seat = self.centroid_seat(coord_seat)
                    dicimg[img][typeseat][str(coord_centroid_seat)] = {}
                    # for each obstacle type
                    for obstacle_type in json_zone[img].keys():
                        # if there is obstacles of that type of obstacle
                        if len(json_zone[img][obstacle_type]) > 0:
                            dicimg[img][typeseat][str(coord_centroid_seat)][
                                obstacle_type] = []
                            # for each coordinates in the obstacle type
                            for coord_obstacle_type in json_zone[img][
                                obstacle_type]:
                                # get the centroid position of the obstacle
                                coord_centroid_obstacle = self.centroid_obstacle(
                                    coord_obstacle_type)

                                # calcualate the distance etween the seat and the obstacle
                                distance = self.dist_crow_flies(
                                    coord_centroid_seat,
                                    coord_centroid_obstacle)

                                # save this distance in the dict
                                dicimg[img][typeseat][
                                    str(coord_centroid_seat)][
                                    obstacle_type].append(
                                    [coord_centroid_obstacle, distance])
        return dicimg



pipeline = Pipeline("/data/dataset/projetinterpromo/Interpromo2020/",
                    ["Aer_Lingus_Airbus_A330-300_A_plane6.jpg"])
#
pipeline.add_processes([BlackWhite, SeatFinder, RemoveDoubleSeat])
pipeline.run_pipeline(1)


def show_seats_find(image_rgb, json=None, img_name=None):
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

    for category in json[img_name]:
        for pos in json[img_name][category]:
            cv.rectangle(
                image_rgb, pos[0:2], (pos[0] + pos[3], pos[1] + pos[2]),
                color[category], 2)

    plt.imshow(image_rgb)
    plt.show()


import sklearn
from sklearn.cluster import DBSCAN


class AStarGraph(object):
    """Documentation
    Class to create the graph for the execution of the A* algorithm
    """

    # Define a class board like grid with two barriers
    def __init__(self, barriers: list):
        """Documentation
        Assign the barriers for the object graph
        Parameters:
            barriers: list of lists of tuples. Each list corresponds to the points that constitute the obstacles.
        """
        self.barriers = barriers

    def heuristic(self, start: tuple, goal: tuple):
        """Documentation
        Function to calculate the distance between the start and end point
        Parameters:
            start: Point of the start for the A* algorithm
            goal: Point of the end for the A* algorithm
        Out:
            Distance calculated between the start point and the end point
        """
        # Use Chebyshev distance heuristic if we can move one square either
        # adjacent or diagonal
        D: int = 1
        D2: int = 1
        dx: int = abs(start[0] - goal[0])
        dy: int = abs(start[1] - goal[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def get_vertex_neighbours(self, pos: tuple):
        """Documentation
        Returns the neighbouring points according to the four movements in front, right, left and back
        Parameters:
            pos: Current point
        Out:
            n: All neighbour points
        """
        n = []
        # Allowed movements are left, front, right and back
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x2 = pos[0] + dx
            y2 = pos[1] + dy
            #             if x2 < 0 or x2 > 7 or y2 < 0 or y2 > 7:
            #                 pass
            n.append((x2, y2))
        return n

    def move_cost(self, a: tuple, b: tuple):
        """Documentation
        Calculate the cost of a move to a neighbour
        Parameters:
            a: Current point
            b: Neighbour
        """
        for barrier in self.barriers:
            if b in barrier:
                return 9999999999999999999  # Extremely high cost to enter barrier squares
        return 1  # Normal movement cost




#     def to_json_complex_distance(self, json_seat: dict, json_zone: dict):
#         """Documentation
#         Parameters:
#             pipeline_zone.json: json ???
#             pipeline.json: json ???
#         Out:
#             dicimg: json final structure
#         """
#         dicimg = {}

#         # Get all obstacles coord
#         all_obstacles_coord = []
#         for img in list(json_zone.keys()):
#             for obstacle_type in json_zone[img].keys():
#                 for coord_obstacle_type in json_zone[img][obstacle_type]:
#                     all_obstacles_coord.append(coord_obstacle_type)

#         for img in list(json_zone.keys()):
#             dicimg[img] = {}
#             for typeseat in json_seat[img].keys():
#                 # Prepare the distance
#                 # Get the coord_seats merge
#                 # Find clusters for each seat position
#                 # Get the rectangles from the seats position
#                 coord_seats_merge= self.change_format(json_seat[img])[0]
#                 clusters = self.find_cluster(31, 3, coord_seats_merge)
#                 plot_rect, rectangles = self.clusters_to_rect(clusters, np.array(coord_seats_merge))

#                 dicimg[img][typeseat] = {}
#                 # for each coordinate in a type seat
#                 for coord_seat in json_seat[img][typeseat]:
#                     # get the centroid position of the seat
#                     coord_centroid_seat = self.centroid_seat(coord_seat)
#                     dicimg[img][typeseat][str(coord_centroid_seat)] = {}
#                     # for each obstacle type
#                     for obstacle_type in json_zone[img].keys():
#                         # if there is obstacles of that type
#                         if len(json_zone[img][obstacle_type]) > 0:
#                             dicimg[img][typeseat][str(coord_centroid_seat)][obstacle_type] = []
#                             for coord_obstacle_type in json_zone[img][obstacle_type]:
#                                 # get the centroid position of the obstacle
#                                 coord_centroid_obstacle = self.centroid_obstacle(coord_obstacle_type)

#                                 # get the distance with the obstacles
#                                 distance_jason = self.pathfinder(coord_centroid_seat, coord_centroid_obstacle, rectangles, all_obstacles_coord)
#                                 print(coord_centroid_seat, coord_centroid_obstacle, distance_jason)
#                     break
#                 break
#             break
#         return dicimg
#
#

"""

from gensim.parsing.preprocessing import strip_numeric, strip_non_alphanum
def merge_elements(json_zone):
    merge_dictio = {}
    for k in json_zone.keys():
        for el in json_zone[k].keys():
            merge_dictio[strip_non_alphanum(strip_numeric(el.split('.')[0]))] = []
    
    keys = merge_dictio.keys()
    for k in json_zone.keys():
        for el in json_zone[k].keys():
            for merge_key in merge_dictio.keys():
                if merge_key in el:
                    merge_dictio[merge_key]+= json_zone[k][el]
                merge_dictio[merge_key] = list( dict.fromkeys(merge_dictio[merge_key]) )
    return json_zone, merge_dictio
    
merge_elements(pipeline_zone.json)
"""
