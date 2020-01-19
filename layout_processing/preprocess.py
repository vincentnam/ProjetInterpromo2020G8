from .metaprocess import MetaProcess, overrides
from abc import abstractmethod
from typing import Iterable
from .colourtool import Colour

class Preprocess(MetaProcess):
    """
    Documentation
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

    def __init__(self, col_obj: Colour, *args, **kwargs):
        """
        Documentation
        Constructor.
        Parameter:
            col_obj: color object to interact with the image
        """
        super().__init__(*args, **kwargs)
        # Colour object containing the image to preprocess and
        # may be already preprocessed.
        self.col_obj = col_obj

    @overrides
    @abstractmethod
    def run(self, **kwargs) -> Iterable:
        """
        Documentation
        Image is open in col_obj and preprocess are done on this image.
        The col_obj is the same during all the pipeline and all
        preprocess compute on this image.
        Out:
            image preprocessed to keep changes and set the new image in 
            the col_obj.
        """
        pass

