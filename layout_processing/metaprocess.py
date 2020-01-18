import inspect
import re
from abc import ABCMeta, abstractmethod
from typing import Iterable

def overrides(method):
    """
    Documentation
    Decorator implementation for overriding
    Attribute:
        method: the method to overrides
    Out:
        the proper version of the method
    Reference:
        1. https://stackoverflow.com/questions/1167617/
    in-python-how-do-i-indicate-im-overriding-a-method
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
    """
    Documentation
    Exception class to handle problem of object insertion in pipeline
    Attributes:
        process_desc -- process descrition to print
        process_class -- process class to print
    """
    def __init__(self, expression: str, message: str):
        """
        Documentation
        Constructor.
        Parameters:
            expression: expression to print
            message: message to print
        """
        self.expression = expression
        self.message = message


class MetaProcess(metaclass=ABCMeta):
    """
    Documentation
    Metaclass for process definition. Used to define a process behaviour
    to be able to make a pipeline of processes. (Cf. Pipeline class)
    """

    def check_attributes(self):
        """
        Documentation
        Check if attributes is defined and not empty. Raise an error
        if not defined.
        """
        if self.process_desc is None or self.process_desc == "":
            raise NotImplementedError("Définissez une description pour "
                                      + "le process.")

    def __init__(self, verbose: int=1, *args, **kwargs):
        """
        Documentation
        MetaProcess constructor. Check if process_desc is implemented.
        Parameters:
            verbose: >0 implies a printing of process_desc when called.
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
        Documentation
        Abstract attribute defining a process definition. The process
        description printing can be avoid by setting "verbose=0".
        This attribute is used to describe the computation and the
        major library used in the run (and the version).
        """
        return self.process_desc

    @abstractmethod
    def run(self, image: iter, **kwargs) -> None:
        """
        Documentation
        Run function and do the computation of the class. This function
        is used in pipeline and only this is launched in pipeline. Work
        as a main and parameters are filled with **kwargs dictionnary.
        Parameter:
            image: image to process, objet array-like
        """

