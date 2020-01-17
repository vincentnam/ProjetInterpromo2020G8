from .preprocess import Preprocess
from typing import Iterable

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

