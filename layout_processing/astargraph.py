class AStarGraph(object):
    """Documentation
    Class to create the graph for the execution of the A* algorithm
    """

    # Define a class board like grid with two barriers
    def __init__(self, barriers: list):
        """Documentation
        Assign the barriers for the object graph
        Parameters:
            barriers: list of lists of tuples. Each list corresponds to
            the points that constitute the obstacles.
        """
        self.barriers = barriers

    def heuristic(self, start: tuple, goal: tuple):
        """Documentation
        Function to calculate the distance between the start and end point
        Parameters:
            start: Point of the start for the A* algorithm
            goal: Point of the end for the A* algorithm
        Out:
            Distance calculated between the start point and the end point
        """
        # Use Chebyshev distance heuristic if we can move one square either
        # adjacent or diagonal
        D: int = 1
        D2: int = 1
        dx: int = abs(start[0] - goal[0])
        dy: int = abs(start[1] - goal[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def get_vertex_neighbours(self, pos: tuple):
        """Documentation
        Returns the neighbouring points according to the four movements
        in front, right, left and back
        Parameters:
            pos: Current point
        Out:
            n: All neighbour points
        """
        n = []
        # Allowed movements are left, front, right and back
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x2 = pos[0] + dx
            y2 = pos[1] + dy
            #             if x2 < 0 or x2 > 7 or y2 < 0 or y2 > 7:
            #                 pass
            n.append((x2, y2))
        return n

    def move_cost(self, a: tuple, b: tuple):
        """Documentation
        Calculate the cost of a move to a neighbour
        Parameters:
            a: Current point
            b: Neighbour
        """
        for barrier in self.barriers:
            if b in barrier:
                # Extremely high cost to enter barrier squares
                return 9999999999999999999
        return 1  # Normal movement cost

