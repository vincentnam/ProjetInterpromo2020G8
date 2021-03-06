from .postprocess import Postprocess


class RemoveDoubleSeat(Postprocess):
    """Documentation
    To remove the duplicated detected seats.
    """
    process_desc = "Standard Python >= 3.5 -> remove double point in list"

    def __init__(self, *args, **kwargs):
        """
        Documentation
        Constructor.
        """
        super().__init__()

    def remove_duplicate(self, coordinate: list):
        """Documentation
        To remove the duplicated detected seats.
        Parameters
            coordinate: original coordinates without treatment
        Out:
            list of coordinate without duplicated elements
        """
        dup = {}

        for category in coordinate:
            dup[category] = []
            for point1 in coordinate[category]:
                for point2 in coordinate[category]:
                    if point2 != point1 and point1 not in dup:
                        if ((abs(point1[0] - point2[0]) <= 5) and (
                                abs(point1[1] - point2[1]) <= 5)):
                            dup[category].append(point2)
        for d in dup:
            for category in coordinate:
                if d in coordinate[category]:
                    coordinate.remove(d)

        return coordinate

    def run(self, json: dict, **kwargs):
        """
        Documentation
        To run the removedoubleseat.
        """
        for seat_index in json:
            json[seat_index] = self.remove_duplicate(json[seat_index])
