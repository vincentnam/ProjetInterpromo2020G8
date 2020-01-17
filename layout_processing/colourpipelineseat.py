from typing import Iterable
from .preprocess import Preprocess


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

