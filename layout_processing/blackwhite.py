from .preprocess import Preprocess
from typing import Iterable


class BlackWhite(Preprocess):
    """
    Transform colored image into a grey scale image.
    """
    process_desc = "OpenCV4.1.2.30 -> rgb to grey"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> Iterable:
        return self.col_obj.util_obj.to_gray()


