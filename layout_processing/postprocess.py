from .metaprocess import MetaProcess, overrides
from abc import abstractmethod
from typing import Iterable


class Postprocess(MetaProcess):
    """
    Documentation
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
        """
        Documentation
        Constructor.
        """
        super().__init__()

    @overrides
    @abstractmethod
    def run(self, image: iter, json: dict, **kwargs) -> None:
        """
        Documentation
        
        """
        super()
        pass

