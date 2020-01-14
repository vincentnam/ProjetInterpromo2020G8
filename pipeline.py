"""
Created on Fri Jan 3 13:28:13 CET 2019
Group 8
@authors: DANG Vincent-Nam
"""

from abc import ABCMeta, abstractmethod
from typing import Iterable

import inspect
import re

import os
import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np

from collections import defaultdict
from PIL import Image


COLOURS = {
    'LAYOUT_SEATGURU': {
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
    'LAYOUT_SEATMAESTRO': {
        'png': {
            "blue": [81, 101, 181],
            "exit": [1, 120, 175],
            "green": [120, 189, 198],
            "red_bad_seat": [207, 90, 150],
            "blue_seat_crew": [138, 165, 190]
        }
    }
}


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

# TODO :
#  - Tests unitaires et tests intégrations : test pipeline
#  (run_pipeline), levées d'erreur, etc...
#  - Traduire les commentaires en anglais (si besoin ?)
#  - Mettre à jour le pipeline pour prendre en compte des resultats
#    auxiliaires nécessaire pour le traitement suivant
#  - Gestion des hints plus formellement
#  - Gestion de l'héritage des docstrings


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


# Classe pour le pipeline
class Pipeline:
    """
    Classe permettant de définir un pipeline. Le pipeline execute dans
    l'ordre le pré_processing, le processing et le post_processing.
    Cette classe contient 3 numpy array contenant la liste des
    traitements pour chaque étapes à réaliser. Il faut utiliser les
    fonction add_pre_process
    """

    def __init__(self, data_path, list_images_name: Iterable[str], layouts: Iterable[str] = ['LAYOUT_SEATGURU', 'LAYOUT_SEATMAESTRO']) -> None:
        self.pre_process: Iterable[type] = np.array([])
        self.process: Iterable[type] = np.array([])
        self.post_process: Iterable[type] = np.array([])
        self.json = {}

        # definition of input path for images
        self.data_path = data_path
        self.layouts = layouts
        self.list_images_name = list_images_name
        self.input_path = self.data_path + self.layouts[0] + '/'

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
    def run_pipeline(self, nb_images: int, **kwargs) -> None:
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


        for image_name in os.listdir(self.input_path)[:nb_images]:
            # Create a Colour object
            col_obj = Colour(self.data_path, self.layouts[0], image_name)
            # Create a
            #util_obj = ImageUtil(self.data_path + self.layouts[0] + "/", image_name)
            temp_json = {}
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
                    print("Le pré-processing numéro " + str(num)
                          + "( " + pre_pro.process_desc
                          + " ) a levé une erreur.")
                    print(e)

            for num, process in enumerate(self.process):
                pro = process()
                try:
                    if image is not None:
                        print("Doing : " + pro.process_desc)
                        # data_image = le path de l'image
                        pro.run(image, temp_json,image_rgb = col_obj.image, data_image = self.data_path + self.layouts[0] + "/" + image_name ,**kwargs)
                    else:
                        raise ValueError("Image = None")
                except Exception as e:
                    print("Le processing numéro " + str(num)
                          + "( " + pro.process_desc + " ) a levé une erreur.")
                    print(e)

            for num, post_process in enumerate(self.post_process):
                post_pro = post_process()
                try:
                    print("Doing : " + post_pro.process_desc)
                    post_pro.run(temp_json, **kwargs)
                except Exception as e:
                    print("Le post_processing numéro " + str(num)
                          + " a levé une erreur.")
                    print(e)
            self.json[image_name] = temp_json


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
'''




