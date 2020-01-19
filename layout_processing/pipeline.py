import numpy as np
from typing import Iterable
from .preprocess import Preprocess
from .process import Process
from .postprocess import Postprocess
from .metaprocess import MetaProcess, NotProcessClass
import traceback
import os
import sys
import matplotlib.pyplot as plt
from .colourtool import Colour, COLOURS


class Pipeline:
    """
    Documentation
    Pipeline class : define the process order (pre-process -> process
    -> post process) and informations exchanges between processes.
    To add a process, add_processes take a list of processes even if
    this list contain only 1 process.
    """

    def __init__(self, data_path, list_images_name: Iterable[str] = None,
                 layouts: Iterable[str] =
                 ['LAYOUT SEATGURU', 'LAYOUT SEATMAESTRO']) -> None:
        """
        Documentation
        Pipeline constructor
        Parameters:
            data_path: path to the base folder directly extracted from the .zip archive
            list_images_name: list of files images to make the pipeline on.
            layouts: list of layout folder to consider
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

    def add_processes(self, in_process: iter):
        """
        Documentation
        Add a list of processes in the pipeline. Processes are run in
        the same order that they are processed
        Parameter:
            in_process: list of process to add at the pipeline
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
        Documentation
        Print process that are in the pipeline. The order of printing is
        the same as the running order.
        """
        for process in self.pre_process:
            print(process.process_desc)
        for process in self.process:
            print(process.process_desc)
        for process in self.post_process:
            print(process.process_desc)

    def print_traceback(self, proc: MetaProcess, num: int,
                        e: Exception) -> None:
        """Documentation
        Print ???
        Parameters:
            proc:
            num:
            e:
        """
        traceback.print_tb(e.__traceback__)
        print("Proprocess number " + str(num)
              + "( " + proc.process_desc
              + " ) raised an error.")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[
            1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)

    def run_pipeline(self, nb_images: int, verbose: int = 1, **kwargs) -> None:
        """
        Documentation
        Run the pipeline. Compute, in order :
            - preprocessing
            - processing
            - postprocessing
        Each process is kept in a list for each type of process. Process
        are run in the order they are put in the list.
        Run the computations over 1 image at a time and are directly
        modified.
        The output are saved in the json class argument.
        Parameters:
            nb_images: number of images to compute the pipeline on
            param verbose : set visualisation on or off. Image are
            plot with matplotlib and process list are printed if
            verbose = True.
            **kwargs : allow argument passing by this dictionnary. It
            is used to give parameters to process in the run. Don't 
            forget to name parameter the same as in the process definition.
            :param nb_images: int : number of images to process, if nb_images
            < 0, the whole dataset is processed.
            :param verbose: int :
             if verbose = 0 : nothing is plot
             if verbose = 1 : images are plot
             if verbose > 2 : json are also plot (Warning : it can takes
             a lot of place in standard output)
        """
        if verbose is True:
            self.print_process()
        if self.list_images_name is None:
            if nb_images > 0:
                self.list_images_name = os.listdir(self.image_folder_path)[
                                        :nb_images]
            else:
                self.list_images_name = os.listdir(self.image_folder_path)
        for image_name in self.list_images_name:
            # Create a Colour object containing the image to process.
            col_obj = Colour(self.layout_folder_path, self.layouts[0],
                             image_name)
            if verbose > 0 :
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
                    if verbose > 1:
                        plt.figure(figsize=(4, 4))
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
                        if verbose > 2:
                            print(self.json)
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
