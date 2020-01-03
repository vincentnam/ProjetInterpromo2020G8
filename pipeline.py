"""
Created on Fri Jan 3 13:28:13 CET 2019
Group 8
@authors: DANG Vincent-Nam
"""

from abc import ABCMeta, abstractmethod
from typing import Iterable
import numpy as np


# TODO :
#  - Traduire les commentaires en anglais (si besoin ?)
#  - Mettre à jour le pipeline pour prendre en compte des resultats
#    auxiliaires nécessaire pour le traitement suivant
#  - Gestion des hints plus formellement
#  - Gestion de l'héritage des docstrings
#  -


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
        if self.process_desc is None or self.process_desc is "":
            raise NotImplementedError("Définissez une description pour "
                                      + "le process.")

    def __init__(self,verbose=1):
        self.verbose = verbose
        self.check_attributes()
        super().__init__()
        if self.verbose > 0 :
            print(self.__class__.__base__.__name__ + " : ", end=' ')
            print(self.process_desc)

    @property
    @abstractmethod
    def process_desc(self):
        """Name of the process to be able to identify the process"""
        return self.process_desc

    @abstractmethod
    def run(self, images: Iterable[Iterable]) -> None:
        """
        Réalise le traitement à effectuer sur la liste d'images. Ne
        retourne rien. Les modifications sont effectuées directement sur
        les images dans la liste.
        :param images: objet de array-like contenant les images à traiter
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def run(self, images: Iterable[Iterable]) -> None:
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
    class Preprocess_exemple(Proprocess):
        process_desc = "PyTesseract2.0-> recherche de caractères"
    ...
    """
    process_desc = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def run(self, images: Iterable[Iterable]) -> None:
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

    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

    @abstractmethod
    def run(self, images : Iterable[Iterable]) -> None:
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
    def __init__(self) -> None:
        self.pre_process: Iterable[Preprocess] = np.array([])
        self.process: Iterable[Process] = np.array([])
        self.post_process: Iterable[Postprocess] = np.array([])

    def add_processes(self, in_process: Iterable[MetaProcess]):
        for process in in_process:
            if isinstance(process, Preprocess):
                self.pre_process = np.append(self.pre_process,
                                             np.array([process]))
            if isinstance(process, Process):
                self.process = np.append(self.process,
                                             np.array([process]))
            if isinstance(process, Postprocess):
                self.post_process = np.append(self.post_process,
                                             np.array([process]))

    def print_process(self):
        for process in self.pre_process:
            print(process.process_desc)
        for process in self.process:
            print(process.process_desc)
        for process in self.post_process:
            print(process.process_desc)


    # Pas besoin de retourner les variables : on modifie directement les images
    def run_pipeline(self, images: Iterable[Iterable]) -> None:
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
        print("Début du pipeline : ")
        for num, pre_process in enumerate(self.pre_process):
            try:
                print("Doing : " + pre_process.process_desc)
                pre_process.run(images)
            except Exception as e :
                print("Le pré-processing numéro " + str(num)
                      + "( " + pre_process.process_desc + " ) a levé une erreur.")
                print(e)

        for num, process in enumerate(self.process):
            try:
                print("Doing : " + process.process_desc)
                process.run(images)
            except Exception as e:
                print("Le processing numéro " + str(num)
                      + "( " + process.process_desc + " ) a levé une erreur.")
                print(e)

        for num, post_process in enumerate(self.post_process):
            try :

                print("Doing : " + post_process.process_desc)
                post_process.run(images)
            except Exception as e:
                print("Le post_processing numéro " + str(num)
                      + " a levé une erreur.")
                print(e)


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

# Exemple of usage of the pipeline.

if __name__== '__name__':
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


    pipeline = Pipeline()
    pipeline.add_processes([Augmentation()])
    pipeline.add_processes([PyTesseract()])
    pipeline.add_processes([Alignement()])

    pipeline.run_pipeline([])
    print(Augmentation.run.__doc__)