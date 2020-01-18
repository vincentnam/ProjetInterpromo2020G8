from .metaprocess import MetaProcess, overrides
from abc import abstractmethod
from typing import Iterable


class Process(MetaProcess):
    """
    Doumentation
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
        """
        Documentation
        Constructor
        """
        super().__init__(*args, **kwargs)

    @overrides
    @abstractmethod
    def run(self, image: iter, json: dict, **kwargs) -> None:
        """
        Documentation
        Parameters:
            image: image preprocessed
            json: input/output parameter, the dictionnary
            to fill with informations
        """
        pass

