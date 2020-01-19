from .process import Process
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2 as cv


class SeatFinder(Process):
    """
    Documentation
    
    """
    process_desc = "OpenCV4.1.2.30 -> Pattern Matching seat"

    def __init__(self, csv_data_path: str=None, *args, **kwargs):
        """
        Documentation
        Constructor.
        Parameter:
            csv_data_path: path of the csv containing the data
        """
        if csv_data_path is None:
            raise Exception("Data_path is empty in init function of " + str(
                self.__class__))
        super().__init__()
        self.csv_data_path = csv_data_path
        self.layout_folder_path = csv_data_path + "ANALYSE IMAGE/"
        self.seatguru_image_data_path = self.layout_folder_path + "LAYOUT SEATGURU/"
        self.seatmaestro_image_data_path = self.layout_folder_path + "LAYOUT SEATMAESTRO/"

    def hasNumbers(self, inputString: str):
        """
        Documentation
        Allow us to know if a string contains a number
        Parameter:
            inputString: input string
        Out :
            return a boolean, true if the string contains an int, else false
        """
        return any(char.isdigit() for char in inputString)

    def longestValue(self, inputList: list):
        """
        Documentation
        Takes the longest string in a list of strings
        Parameter:
            inputList: input list of strings
        Out :
            Return the longest string in the list of strings
        """
        if len(inputList) > 0:
            max_len = 0
            for i, el in enumerate(inputList):
                if len(el) > max_len:
                    max_len = i
            return inputList[max_len]
        return 0

    def get_relevant_aircraft_builders(self, image_names: list,
                                       proportion_min: float = 0.02,
                                       proportion_max: float = 0.75):
        """
        Documentation
        Find a list of relevant aircraft builder
        Parameters:
            image_names: input list of strings
            proportion_min : proportion minimum to consider an aircraft builder
            proportion_max : proportion maximum to consider an aircraft builder
        Out :
            Return a dictionnary containing relevant aircraft builders 
        """
        # retrieve aircraft builders types
        aircraft_builders = np.unique(
            [a_type.split(' ')[0].lower() for a_type in
             pd.read_csv(
                 self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv',
                 sep=';')['Aircraft_Type'].tolist() if
             len(a_type.split(' ')) > 1])

        # remove aircraft_builder that have Numerics in it
        aircraft_builders = [a_builder for a_builder in aircraft_builders if
                             not self.hasNumbers(a_builder) and len(
                                 a_builder) > 2]

        # remove aircraft with too low occurences in the "image_names" list
        relevant_aircraft_builders = {}
        for a_builder in aircraft_builders:
            proportion = sum(
                [1 for img_n in image_names if
                 a_builder in img_n.lower()]) / len(
                image_names)
            if proportion > proportion_min and proportion < proportion_max:
                relevant_aircraft_builders[a_builder] = proportion

        # remove non aircradft builder that remain
        del relevant_aircraft_builders['airlines']
        del relevant_aircraft_builders['irbus']
        del relevant_aircraft_builders['airways']
        del relevant_aircraft_builders['ays']

        # sort by proportion
        relevant_aircraft_builders = {k: v for k, v in
                                      sorted(
                                          relevant_aircraft_builders.items(),
                                          key=lambda item: item[1])}
        return relevant_aircraft_builders

    def get_image_information(self, image_names: iter, aircraft_builders: iter,
                              airlines: iter, aircraft_ref: iter, pattern: str,
                              layout: str = "LAYOUT SEATGURU/"):

        """
        Documentation
        Find complementary information with the image name
        Parameters
            image_names: input list of strings
            aircraft_builders : a list of aircraft builders
            airlines : a list of airlines
            aircraft_ref : a list of aircraft references
            pattern : determine which splitter we want to take for analysing the image name
            layout : a layout to consider only the right dataset
        Out :
            Return a dictionnary containing relevant information by image
        """
        # Initialisation of dict
        dictio_airlines_aircraft_builders = []

        for image_name in image_names:

            size = Image.open(
                self.layout_folder_path + layout + image_name).size

            dictio_airlines_aircraft_builders.append({
                'image_name': image_name,
                'aircraft_builder': 'not_relevant_aircraft_builders',
                'position_aircraft_builder': -1,
                'airline': '',
                'aircraft_ref': [],
                'x_size': int(size[0]),
                'y_size': int(size[1])
            })

            # a little pre-process to clean-up image name
            img_inf = image_name.lower().split('.')[0].split(pattern)

            for i, item in enumerate(img_inf):
                for a_builder in aircraft_builders:
                    # check if the image contains the aircraft builder
                    if a_builder == item:
                        # add the aircraft builder in the image name
                        dictio_airlines_aircraft_builders[-1][
                            'aircraft_builder'] = a_builder
                        dictio_airlines_aircraft_builders[-1][
                            'position_aircraft_builder'] = i

            # add airlines deduce by the image name
            for airline in airlines:
                if airline + pattern in image_name.lower():
                    dictio_airlines_aircraft_builders[-1][
                        'airline'] = airline

            # add aircraft_ref
            for a_ref in aircraft_ref:
                if a_ref in image_name.lower():
                    dictio_airlines_aircraft_builders[-1][
                        'aircraft_ref'].append(a_ref)
            # take the longest str element in the list of ref
            dictio_airlines_aircraft_builders[-1][
                'aircraft_ref'] = self.longestValue(
                dictio_airlines_aircraft_builders[-1]['aircraft_ref'])

        return dictio_airlines_aircraft_builders

    def main_layout_seatguru(self, layout: str = "LAYOUT SEATGURU/"):
        """
        Documentation
        Retrieve meta-data about each image containing in the layout folder
        Parameter:
            layout: a layout to consider only the right dataset
        Out :
            return a dataframe containing the meta-data about an image
        """
        image_name_list = [img for img in
                           os.listdir(self.layout_folder_path + layout)]

        relevant_aircraft_builders = self.get_relevant_aircraft_builders(
            image_name_list)
        airlines = \
            pd.read_csv(self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv',
                        sep=';')[
                'Airline_name'].unique()
        airlines = [airline.replace('-', '_') for airline in airlines]

        aircraft_ref = np.unique([a_type.lower() for a_type in pd.read_csv(
            self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv', sep=';')[
            'Aircraft_Type'].tolist()])
        aircraft_ref = np.unique(
            [t for text in aircraft_ref for t in text.split(' ') if
             self.hasNumbers(t)])

        dictio_airlines_aircraft_builders = self.get_image_information(
            image_name_list, relevant_aircraft_builders, airlines,
            aircraft_ref,
            '_')
        return pd.DataFrame(dictio_airlines_aircraft_builders)

    def get_image_extra_information(self, dataframe: pd.DataFrame):
        """
        Documentation
        Make a correspondance between the csv file and our image to get extra information
        Parameter:
            dataframe : dataframe containing image info
        Out :
            return a dataframe containing the other meta-data about an image
        """
        # retrieve aircraft builders
        aircraft_builders = dataframe['aircraft_builder'].tolist()
        # retrieve arcraft references
        aircraft_refs = dataframe['aircraft_ref'].tolist()
        # retrieves airline names
        airlines = dataframe['airline'].tolist()
        # retrieve image names
        image_names = dataframe['image_name'].tolist()

        df_seat_guru_info = pd.read_csv(
            self.csv_data_path + 'SEATGURU_INFO_AIRCRAFT.csv', sep=';')
        aircraft_type = df_seat_guru_info['Aircraft_Type']
        # dictionnary where the info will be stored
        dictio = {}
        # for each aircraft builder, aircraft ref, airline and image name,
        # we research the relevant information in df_seat_guru_info dataframe
        for a_b, a_r, airline, im in \
                zip(aircraft_builders, aircraft_refs, airlines, image_names):
            res = pd.DataFrame([df_seat_guru_info.loc[i] for i, a_t in
                                enumerate(aircraft_type) if
                                a_b in a_t.lower() and a_r in a_t.lower()]).reset_index()
            if 'Airline_name' in res.columns:  # res not empty right after the research
                res = pd.DataFrame([res.loc[i] for i, a_n in
                                    enumerate(res['Airline_name'].tolist()) if
                                    airline.replace('_', '-') in a_n.lower()])
                if 'Total_seat' in res.columns:  # res not empty right after the research
                    # get the maximum of seat number
                    nb_max_seats = max(res['Total_seat'].unique().tolist())
                    # select the seats with the help of nb_max_seats
                    res['selected'] = res['Total_seat'].apply(
                        lambda x: 1 if x == nb_max_seats else 0)
                    # select the column that we need
                    res = res[['Category', 'Count', 'Seat_Type', 'selected']]
                    res = res[res[
                                  'selected'] == 1].drop_duplicates()  # remove duplicated lines
                    dictio[im] = res.drop(columns=['selected'])
                else:
                    dictio[im] = pd.DataFrame(columns=['Category', 'Count', 'Seat_Type'])
            else:
                dictio[im] = pd.DataFrame(columns=['Category', 'Count', 'Seat_Type'])
        return dictio

    def retrieve_relevant_seat_info(self, dictio: dict, image_name: str):
        """
        Documentation
        Retrieve the relevant cross information between csv file and an image concerning a seat
        Parameters:
            dictio : dictionnary that contains correspondance information in link with a seat
            image_name : image name
        Out :
            Add seat information as Count, Category and Seat_Type to an image information
        """
        total_seat_info = []
        for i, row in dictio[image_name].iterrows():
            total_seat_info.append({
                'Category': row['Category'],
                'Seat_Type': row['Seat_Type'],
                'Count': row['Count']
            })
        return total_seat_info

    def coord_pattern_finder(self, image: iter, template: iter, threshold: float):
        """
        Documentation
        Find a position by pattern matching a template on an image
        Parameter:
            image : image plane cv.imread() black and white
            template : image pattern cv.imread() black and white
            threshold : threshold for this pattern
        Out :
            position : list right angle position for this pattern on the image
        """
        position = []  # Variable output
        # List of match
        res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

        for pos in zip(*np.where(res >= threshold)[::-1]):
            position.append(pos)

        return (position)

    def templ_category(self, path: str = './images/TEMPLATE/', category: str = 'BUSINESS',
                       seat_type: str = 'STANDARD', image_name: str = 'test.jpg'):
        """
        Documentation
        Create list of templates by category and seatType
        Parameters :
            path: directory path of templates
            category: name of category
            seatType: the seat type
            image_name: the image name
        Out :
            Return a list of templates
        """
        if '.png' in image_name:
            extension = 'PNG/'
        else:
            extension = 'JPG/'

        imagesTemp = os.listdir(path + category + '/' + extension)
        templates = []

        for i in imagesTemp:
            if seat_type in i:
                templates.append(cv.imread(
                    path + category + '/' + extension + i, 0))
        return (templates)

    def template_from_template(self, img: iter, template: iter, threshold_min: float = 0.70):
        """
        Documentation
        Moving the threshold from 1 to his mininum value, in order to select the relevant selected
        elements by pattern matching the template on the image
        Parameters:
            img : the image chosen (already read)
            template : the template chosen
            threshold_min : threshold minimum that we take to pattern match
            the template on the image and to stop searching
        Out :
            return the position of the an element recognized by pattern matching,
            or None if nothing was found
        """
        # default Threshold
        threshold = 1
        position = self.coord_pattern_finder(img, template, threshold)
        h, w = template.shape
        # Reduce Threshold while no template match
        while len(position) < 1 and threshold > threshold_min:
            threshold -= 0.005
            position = self.coord_pattern_finder(img, template,
                                                 threshold)

        if threshold > threshold_min:
            return (img[position[0][1]:position[0][1] + h,
                    position[0][0]:position[0][0] + w], True)
        return (None, False)

    def count_list(self, to_order: list):
        """
        Documentation
        Take a list in order to sort this list by number of time that an element appears
        Parameters :
            list : the list to order
        Out :
            return an ordered list with a single occurence
        """
        dictio_count = {}
        for el in to_order:
            dictio_count[el] = to_order.count(el)
        return {k: v for k, v in sorted(dictio_count.items(),
                                        key=lambda item: item[1],
                                        reverse=True)}

    def best_position(self, img: iter, template: iter, nb_seat: int,
                      step: float = 0.005, threshold_min: float = 0.65):
        """
        Documenation
        Find the best position elements for the seat by considering the template
        Parameters:
            img : image plane
            template : template find from this image
            nb_seat : number of seatt
            step : steps for threshold
            threshold_min : threshold min to keep template or not
        Out :
            coord for each match
        """
        position = []
        for threshold in np.arange(threshold_min, 1 + step, step):
            position += self.coord_pattern_finder(img, template, threshold)

        result = list(self.count_list(position).keys())
        if len(result) < nb_seat * 1.1:
            return (result)
        return (result[:int(nb_seat * 1.1)])

    def run(self, image: iter, json: dict, image_name: str=None,
            layout: list = ["LAYOUT SEATGURU/", "LAYOUT SEATMAESTRO/"],
            path: str = './images/', **kwargs):
        """
        Documentation
        
        Parameters :
            image : image plane
            json : json containing image information
            image_name : image name
            layout : layout type
            path : path for template directory
        Out :
            dictionnary
        """

        if not image_name in json.keys():
            json[image_name] = {}
        df_layout_seatguru = self.main_layout_seatguru(layout[0])
        dictio_correspondance = self.get_image_extra_information(
            df_layout_seatguru[df_layout_seatguru['image_name'] == image_name])
        nbObjectToFind = self.retrieve_relevant_seat_info(
            dictio_correspondance, image_name)

        for objet in nbObjectToFind:
            if not objet["Category"] in json[image_name].keys():
                json[image_name][objet['Category']] = []
            # Take all template name for this category
            templates = self.templ_category(
                category=objet['Category'],
                seat_type=objet['Seat_Type'],
                image_name=image_name)
            for template in templates:
                templateFind, found = self.template_from_template(image,
                                                                 template)
                if found:
                    position = self.best_position(image, templateFind,
                                                  objet['Count'])

                    h, w = template.shape
                    for i in range(len(position)):
                        position[i] = position[i] + (h, w)
                    json[image_name][objet['Category']] += position


