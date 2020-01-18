import matplotlib.pyplot as plt
from .utiltool import ImageUtil


# Global variable : colors dictionnary for layout : used by colors
# preprocess - to determine wich colors has to be kept as element
# color in the image
COLOURS = {
    'LAYOUT SEATGURU': {
        'jpg': {
            "blue": [139, 168, 198],
            "yellow": [247, 237, 86],
            "exit": [222, 111, 100],
            "green": [89, 185, 71],
            "red_bad_seat": [244, 121, 123],
            "blue_seat_crew": [140, 169, 202]
        },
        'png': {
            "blue": [41, 182, 209],
            "yellow": [251, 200, 2],
            "exit": [190, 190, 190],
            "green": [41, 209, 135],
            "red_bad_seat": [226, 96, 82],
            "blue_seat_crew": [41, 182, 209]
        }
    },
    'LAYOUT SEATMAESTRO': {
        'png': {
            "blue": [81, 101, 181],
            "exit": [1, 120, 175],
            "green": [120, 189, 198],
            "red_bad_seat": [207, 90, 150],
            "blue_seat_crew": [138, 165, 190]
        }
    }
}




class Colour:
    """
    Documentation
    Tool class : used to make transformations over images
    """

    def __init__(self, csv_data_path: str, layout: iter[str], image_name: str):
        """
        Documentation
        Constructor for Colour class ; tool box for image preprocessing.
        Can be extended to add preprocesses easily.
        Parameters:
            csv_data_path: input path to the base folder
        containing the csv. Folder architecture is based on the archive
        given at the project beginning (i.e. "ProjetInterpromo2020"
        containing sub folder :  "All\ Data" -> "ANALYSE\ IMAGE/" and
        sub folder with the layout from website.
            layout: list of layout to process/list of str that are folders name.
            image_name: file name of the image to work on
        """

        # Path of the base folder containing all sub folder : sub-folder
        # created after unzipping the archive containing all CSV
        self.csv_data_path = csv_data_path
        # Path of the folder containing layout folder :
        # sub folder of "ALL DATA"
        self.layout_folder_path = csv_data_path + "ANALYSE IMAGE/"
        # Path to the seatguru folder containing all layouts
        self.seatguru_image_data_path = self.layout_folder_path \
                                        + "LAYOUT SEATGURU/"
        # Path to the seatmaestro folder containing all layouts
        self.seatmaestro_image_data_path = self.layout_folder_path \
                                           + "LAYOUT SEATMAESTRO/"

        # Redundant variable : only used to keep function calling
        self.input_path = self.csv_data_path
        # List of layout
        self.layout = layout
        # File image name preprocessed
        self.image_name = image_name
        # File image extension : images graphic charts are
        # differentiated by extension in our dataset.
        self.image_extension = image_name.split('.')[-1]
        # Reading of the image with matplotlib
        self.image = plt.imread(
            self.input_path + self.layout + '/' + self.image_name)
        # Creating an ImageUtil object used for preprocessing
        self.util_obj = ImageUtil(self.input_path + self.layout + '/',
                                  self.image_name)

    
    def set_image(self, image: iter):
        """
        Documentation
        Setter for image update on preprocess
        Parameter:
            image : image that we take for make a change
        """
        self.image = image
        self.util_obj.set_image(image)

    def colour_detection(self, colours: dict, epsilon: int, rgb_len: list, 
                         colour_mode: bool, default_colour: int):
        """
        Documentation
        This function will detect the colour and will do some pre-process
        on it params
        Parameters:
            colours : a dictionnary with a list of specified colours
            epsilon : threshold that allows to consider a colour from *
            another one as close
            rgb_len : only take the 3 first elements from pixel (RGB norm)
            colour_mode :
                if true : it means that if we consider a colour from
                the image close to a colour from the "colours" dict,
                then it will replace the colour by the one in the dict.
                if false : it means that if we consider a colour from
                the image close to a colour from the "colours" dict,
                then it will replace the colour by the default color value.
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

                    # for each colour we change the pixel value if we find
                    # the same colour
                    for colour in colours.values():
                        if sum([1 if abs(p - b) < epsilon else 0 for p, b in
                                zip(pixel, colour)]) == rgb_len:
                            img_copy[i][j] = colour

                # if we want to hide a colour by a default value
                else:
                    # default value
                    img_copy[i][j] = pixel

                    # for each recognized colour, we change it by the
                    # default value
                    for colour in colours.values():
                        if sum([1 if abs(p - b) < epsilon else 0 for p, b in
                                zip(pixel, colour)]) == rgb_len:
                            img_copy[i][j] = default_colour
        return img_copy

    def colour_pipeline(self, colours: dict={}, epsilon: int=20, colour_mode=True,
                        rgb_len=[0, 0, 0], default_color: int=3):
        """
        Documentation
        Call colour_detection function in order to pre-process
        colours in image.
        Parameters:
            colours: a dictionnary with a list of specified colours
            epsilon : threshold that allows to consider a colour
            from another one as close
            rgb_len : only take the 3 first elements from pixel
            (RGB norm)
            colour_mode: 
                if true (highlight colours in "colours" dict by
                standardize it) : it means that if we consider a
                colour from the image close to a colour from the
                "colours" dict, then it will replace the colour by
                the one in the dict.
                if false (remove colours in "colours" dict by the
                default one) : it means that if we consider a colour
                from the image close to a colour from the "colours" dict,
                then it will replace the colour by the default color value.
            default_color : default color value that a pixel has to take
        Out:
            image_res:
        """
        # if colours is empty we take the default value
        if not bool(colours):
            colours = COLOURS[self.layout][self.image_extension]

        # get the image result from colour detection pre-process wanted
        image_res = self.colour_detection(colours, epsilon, rgb_len,
                                          colour_mode, default_colour)

        return image_res
