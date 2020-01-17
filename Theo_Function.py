import os
import numpy as np
import cv2
from pipeline import Pipeline, Process, Postprocess, Preprocess

data_path = '../All Data'


class SeatFinder(Process):
    process_desc = "OpenCV4.1.2.30 -> Pattern Matching seat"

    def __init__(self, csv_data_path=None, *args, **kwargs):
        if csv_data_path is None:
            raise Exception("Data_path is empty in init function of " + str(
                self.__class__))
        super().__init__()
        self.csv_data_path = csv_data_path
        self.layout_folder_path = csv_data_path + "ANALYSE IMAGE/"
        self.seatguru_image_data_path = self.layout_folder_path + "LAYOUT SEATGURU/"
        self.seatmaestro_image_data_path = self.layout_folder_path + "LAYOUT SEATMAESTRO/"

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def longestValue(self, inputList):
        if len(inputList) > 0:
            max_len = 0
            for i, el in enumerate(inputList):
                if len(el) > max_len:
                    max_len = i
            return inputList[max_len]
        return 0

    def get_relevant_aircraft_builders(self, image_names,
                                       proportion_min=0.02,
                                       proportion_max=0.75):
        # retrieve aircraft types

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

        # remove aircraft with too low occurences in the "IMAGES_NAMES" list
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

    def get_airline_aircraft_builder_pos(self, image_names, aircraft_builders,
                                         airlines, aircraft_ref, pattern,
                                         layout="LAYOUT SEATGURU/"):
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

    def main_layout_seatguru(self, layout="LAYOUT SEATGURU/"):
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

        dictio_airlines_aircraft_builders = self.get_airline_aircraft_builder_pos(
            image_name_list, relevant_aircraft_builders, airlines,
            aircraft_ref,
            '_')
        return pd.DataFrame(dictio_airlines_aircraft_builders)

    def get_correspondance(self, dataframe):
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
        return dictio

    def retrieve_relevant_seat_info(self, dictio, image_name):
        total_seat_info = []
        for i, row in dictio[image_name].iterrows():
            total_seat_info.append({
                'Category': row['Category'],
                'Seat_Type': row['Seat_Type'],
                'Count': row['Count']
            })
        return total_seat_info

    def coord_pattern_finder(self, image, template, threshold: float):
        """
        input:
            image : image plane cv2.imread() black and white
            template : image pattern cv2.imread() black and white
            threshold : threshold for this pattern
        output:
            position : list right angle position for this pattern on the image
        """
        position = []  # Variable output
        # List of match
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        for pos in zip(*np.where(res >= threshold)[::-1]):
            position.append(pos)
        return(position)

    def templ_category(self, path='./images/TEMPLATE/', category='BUSINESS',
                       seatType='STANDARD', planeName='test.jpg'):
        """
        Create list of template open with cv2 by category and seatType
        Input:
            Path: directory path of templates
            category: name of category
            seatType: Seat type
            planeName: plane name
        Output:
            templates: list of template name
        """
        if '.png' in planeName:
            extension = 'PNG/'
        else:
            extension = 'JPG/'

        imagesTemp = os.listdir(path + category + '/' + extension)
        templates = []

        for i in imagesTemp:
            if seatType in i:
                templates.append(cv2.imread(
                    path + category + '/' + extension + i, 0))
        return(templates)

    def template_from_template(self, img, template, thresholdMin=0.70):
        """
        intput:
            img : image plane
            template : template
            thresholdMin : threshold min to keep template or not
        output:
            template, boolean : true if found
        """
        # default Threshold
        threshold = 1
        position = self.coord_pattern_finder(img, template, threshold)
        h, w = template.shape
        # Reduce Threshold while no template match
        while len(position) < 1 and threshold > thresholdMin:
            threshold -= 0.005
            position = self.coord_pattern_finder(img, template,
                                                 threshold)
        if threshold > thresholdMin:
            return(img[position[0][1]:position[0][1] + h,
                       position[0][0]:position[0][0] + w], True)
        return(None, False)

    def count_list(self, list):
        """
        input:
            list : list
        output:
            ordored list with single occurence
        """
        dictio_count = {}
        for el in list:
            dictio_count[el] = list.count(el)
        return {k: v for k, v in sorted(dictio_count.items(),
                                        key=lambda item: item[1],
                                        reverse=True)}

    def best_position(self, img, template, nbSeat, step=0.005,
                      thresholdMin=0.65):
        """
        input:
            img : image plane
            template : template find from this image
            nbSeat : integrer - for this cat
            steps : steps for threshold
        output:
            position : coord for each match
        """
        position = []
        for threshold in np.arange(thresholdMin, 1 + step, step):
            position += self.coord_pattern_finder(img, template, threshold)
        result = list(self.count_list(position).keys())
        if len(result) < nbSeat*1.1:
            return(result)
        return(result[:int(nbSeat*1.1)])

    def run(self, img, nbObjectToFind, diction, planeName,
            path='./images/'):
        """
        input:
            img : image plane
            nbObjectToFind : Dictionnary : {
                                            'business': nbBusinessSeat,
                                            'bar': nbBar
                                            }
            diction : diction output
            planeName :
            path : path for template directory
        output:
            diction : dictionnary [{'Category': 'BUSINESS',
                                    'Seat_Type': 'FLAT_BED',
                                    'Count': 30},
                                   {'Category': 'ECONOMY',
                                    'Seat_Type': 'STANDARD',
                                    'Count': 287}]
        """
        for objet in nbObjectToFind:
            if not objet['Category'] in diction[planeName].keys():
                diction[planeName][objet['Category']] = []
            # Take all template name for this category
            templates = self.templ_category(category=objet['Category'],
                                            seatType=objet['Seat_Type'],
                                            planeName=planeName)
            for templ in templates:
                templateFind, find = self.template_from_template(img,
                                                                 templ)
                if find:
                    position = self.best_position(
                        img, templateFind, objet['Count'])
                    h, w = templ.shape
                    for i in range(len(position)):
                        position[i] = position[i] + (h, w)
                    diction[planeName][objet['Category']] += position

    def show_seats_find(self, image_rgb, json=None, img_name=None):
        """
        input:
            image : Opened image with color
            json : coordonate
            img_name : image name
        output:
            NONE
        """
        color = {'BUSINESS': (255, 0, 0),
                 'ECONOMY': (0, 0, 255),
                 'FIRST': (0, 255, 0),
                 'PREMIUM': (255, 255, 0)}

        for category in json[img_name].keys():
            for pos_seat in json[img_name][category]:
                cv2.rectangle(
                    image_rgb, pos_seat[0:2], (pos_seat[0] + pos_seat[3],
                                               pos_seat[1] + pos_seat[2]),
                    color[category], 2)

        cv2.imshow(img_name, image_rgb)
        cv2.waitKey()
        cv2.destroyAllWindows()


nbSeat = {'Aer_Lingus_Airbus_A330-300_A_plane6.jpg': [{'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 30},
                                                      {'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 287}],
          'Aer_Lingus_Airbus_A330-200_B_plane7.jpg': [{'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 248},
                                                      {'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 23}],
          'Aer_Lingus_Airbus_A320_plane9.jpg': [{'Category': 'ECONOMY',
                                                 'Seat_Type': 'STANDARD',
                                                 'Count': 174}],
          'Aer_Lingus_Airbus_A321_plane10.jpg': [{'Category': 'ECONOMY',
                                                  'Seat_Type': 'STANDARD',
                                                  'Count': 212}],
          'Aer_Lingus_Boeing_757-200_plane2.jpg': [{'Category': 'BUSINESS',
                                                    'Seat_Type': 'FLAT_BED',
                                                    'Count': 12},
                                                   {'Category': 'ECONOMY',
                                                    'Seat_Type': 'STANDARD',
                                                    'Count': 165}],
          'Aer_Lingus_Airbus_A330-200_plane6.jpg': [{'Category': 'ECONOMY',
                                                     'Seat_Type': 'STANDARD',
                                                     'Count': 248},
                                                    {'Category': 'BUSINESS',
                                                     'Seat_Type': 'FLAT_BED',
                                                     'Count': 23}],
          'Aer_Lingus_Airbus_A330-200_plane4.jpg': [{'Category': 'ECONOMY',
                                                     'Seat_Type': 'STANDARD',
                                                     'Count': 248},
                                                    {'Category': 'BUSINESS',
                                                     'Seat_Type': 'FLAT_BED',
                                                     'Count': 23}],
          'Aer_Lingus_Airbus_A330-300_A_plane8.jpg': [{'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 30},
                                                      {'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 287}],
          'Aer_Lingus_Airbus_A330-200_B_plane5.jpg': [{'Category': 'ECONOMY',
                                                       'Seat_Type': 'STANDARD',
                                                       'Count': 248},
                                                      {'Category': 'BUSINESS',
                                                       'Seat_Type': 'FLAT_BED',
                                                       'Count': 23}],
          'Aer_Lingus_Airbus_A321_plane1.jpg': [{'Category': 'ECONOMY',
                                                 'Seat_Type': 'STANDARD',
                                                 'Count': 212}],
          'Aer_Lingus_Boeing_757-200_plane11.jpg': [{'Category': 'BUSINESS',
                                                     'Seat_Type': 'FLAT_BED',
                                                     'Count': 12},
                                                    {'Category': 'ECONOMY',
                                                     'Seat_Type': 'STANDARD',
                                                     'Count': 165}]}

# nbSeat = {}
plane_name = 'Asiana_Boeing_777-200_ER_C_plane205.jpg'
img_gray = cv2.imread('../All Data/ANALYSE IMAGE/LAYOUT SEATGURU/' +
                      plane_name, 0)
img_rgb = cv2.imread('../All Data/ANALYSE IMAGE/LAYOUT SEATGURU/' +
                     plane_name, 1)

result = {plane_name: {}}
test = SeatFinder(csv_data_path=data_path)
test.run(img_gray, nbSeat[plane_name], result, plane_name)
test.show_seats_find(img_rgb, json=result, img_name=plane_name)
# print(result)

# for i in result[plane_name].values():
#     for j in i:
#         cv2.rectangle(img_rgb, j[0:2], (j[0] + j[3],
#                                         j[1] + j[2]), (0, 0, 255), 2)
# cv2.imshow('TEST', img_rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()
cv2.imwrite('test.jpg', img_rgb)
