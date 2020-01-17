import sklearn
import numpy as np
from matplotlib import patches

from .astargraph import AStarGraph
import matplotlib.pyplot as plt
import cv2 as cv
from gensim.parsing.preprocessing import strip_numeric, strip_non_alphanum
import pandas as pd
from sklearn.cluster import DBSCAN


class DistPipeline:

    def __init__(self, pipeline, pipeline_zone):
        self.pipeline = pipeline
        self.pipeline_zone = pipeline_zone
        self.pipeline_zone.json = self.merge_elements(self.pipeline_zone.json)

    def change_format(self, dict_seat: dict):
        """Documentation
        Parameters:
            epsilon: the maximum distance between two samples for one
            to be considered as in the neighborhood of the other
            min_sample: the number of samples in a neighborhood for a
             point to be considered as a core point
            list_wo_dup: list of seats coordinates not duplicated
        Out:
            list_wo_dup: list of seats coordinates
            height_width: height and width
        """
        list_wo_dup = []
        height_width = []
        for i in list(dict_seat.values()):
            for j in i:
                list_wo_dup.append((j[0], j[1]))
                height_width.append((j[2], j[3]))
        return list_wo_dup, height_width

    def find_cluster(self, epsilon: int, min_sample: int, list_wo_dup: list):
        """Documentation
        Parameters:
            epsilon: the maximum distance between two samples for one to be
            considered as in the neighborhood of the other
            min_sample: the number of samples in a neighborhood for a point
            to be considered as a core point
            list_wo_dup: list of seats coordinates not duplicated
        Out:
            dbscan: clustering result with DBSCAN
        """
        x_wo_dup = [a for a, b in list_wo_dup]
        y_wo_dup = [b for a, b in list_wo_dup]
        dbscan = DBSCAN(eps=epsilon, min_samples=min_sample).fit(list_wo_dup)
        plt.scatter(x_wo_dup, y_wo_dup, c=dbscan.labels_.astype(
            float), s=50, alpha=0.5)
        plt.show()
        return (dbscan)

    def merge_elements(self, json_zone):
        merge_dictio = {}
        for k in json_zone.keys():
            merge_dictio[k] = {}
            for el in json_zone[k].keys():
                merge_dictio[k][strip_non_alphanum(
                    strip_numeric(el.split('.')[0])).replace(' ', '')] = []

        keys = merge_dictio.keys()
        for k in json_zone.keys():
            for el in json_zone[k].keys():
                for merge_key in merge_dictio[k].keys():
                    if merge_key in el:
                        merge_dictio[k][merge_key] += json_zone[k][el]
                    merge_dictio[k][merge_key] = list(
                        dict.fromkeys(merge_dictio[k][merge_key]))
        return merge_dictio


    def clusters_to_rect(self, dbscan ,
                         array_wo_dup: np.array):
        """Documentation
        Parameters:
            dbscan: clustering result with DBSCAN
        Out:
            list_rect: list of rectangles representing each cluster
            list_rect2: list of rectangles representing each cluster
        """
        list_coord = array_wo_dup
        label_groups = pd.Series(dbscan.labels_).unique()
        list_rect = []  # to plot with plt.patches
        list_rect2 = []  # all corners of the rectangles
        HEIGHT: int = 30
        WIDTH: int = 20
        for group in label_groups:
            index = [i for i, x in enumerate(
                list(dbscan.labels_)) if x == group]
            points_cluster = list_coord[index]
            corner_bottom_right = (
            max(i[0] for i in points_cluster) + WIDTH, min(
                i[1] for i in points_cluster) - HEIGHT)
            corner_top_right = (max(i[0] for i in points_cluster) + WIDTH, max(
                i[1] for i in points_cluster))
            corner_top_left = (min(i[0] for i in points_cluster), max(
                i[1] for i in points_cluster))
            corner_bottom_left = (min(i[0] for i in points_cluster), min(
                i[1] for i in points_cluster) - HEIGHT)
            height = corner_top_right[1] - corner_bottom_right[1]
            width = corner_bottom_right[0] - corner_bottom_left[0]
            list_rect.append(((corner_bottom_left), width, height))
            list_rect2.append(
                (corner_bottom_left, corner_top_left, corner_top_right,
                 corner_bottom_right))
        return list_rect, list_rect2

    def centroid_obstacle(self, coord_obs: list):
        """Documentation
        Parameters:
            coord_obs: cooardinate of the obstacle (top left and bottom right)
        Out:
            coord_bar_obs: barycenter cooardinate of the obstacle
        """
        A_point = coord_obs[1], coord_obs[0]
        B_point = coord_obs[3], coord_obs[2]
        return int(np.mean([A_point[0], B_point[0]])), int(
            np.mean([A_point[1], B_point[1]]))

    def centroid_seat(self, coord_seat: tuple):
        """Documentation
        Parameters:
            coord_seat: cooardinate of the seat
        Out:
            coord_bar_seat: barycenter cooardinate of the seat
        """
        x, y = coord_seat[0], coord_seat[1]
        h, w = coord_seat[2], coord_seat[3]
        return (int(x + w / 2), int(y + h / 2))

    def dist_crow_flies(self, coord_bar_seat: tuple, coord_bar_obs: tuple):
        """Documentation
        Parameters:
            coord_bar_seat: barycenter coordinate of the seat
            coord_bar_obs: barycenter cooardinate of the obstacle
        Out:
            dist: distance between the two barycenter
        """
        dist = np.sqrt(((coord_bar_obs[0] - coord_bar_seat[0])
                        ** 2) + ((coord_bar_obs[1] - coord_bar_seat[1]) ** 2))
        return round(dist, 2)

    def AStarSearch(self, start: tuple, end: tuple, graph: AStarGraph):
        """Documentation
        A* algorithm to find the best path for from one point to another
        Parameters:
            start: Point of the start for the A* algorithm
            end: Point of the end for the A* algorithm
            graph: Graph for the execution of the A* algorithm
        Out:
            path: All points of the best path
            F[end]: Cost of the best path
        """
        # Actual movement cost to each position from the start position
        G: dict = {}
        # Estimated movement cost of start to end going via this position
        F: dict = {}
        # Initialize starting values
        G[start] = 0
        F[start] = graph.heuristic(start, end)  ###appeler class
        closedVertices: set = set()
        openVertices: set = set([start])
        cameFrom: dict = {}
        while len(openVertices) > 0:
            # Get the vertex in the open list with the lowest F score
            current = None
            current_fscore = None
            for pos in openVertices:
                if current is None or F[pos] < current_fscore:
                    current_fscore = F[pos]
                    current = pos
            # Check if we have reached the goal
            if current == end:
                # Retrace our route backward
                path = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    path.append(current)
                path.reverse()
                return path, F[end]  # Done!
            # Mark the current vertex as closed
            openVertices.remove(current)
            closedVertices.add(current)
            # Update scores for vertices near the current position
            for neighbour in graph.get_vertex_neighbours(current):
                if neighbour in closedVertices:
                    # We have already processed this node exhaustively
                    continue
                candidateG = G[current] + graph.move_cost(current, neighbour)
                if neighbour not in openVertices:
                    openVertices.add(neighbour)  # Discovered a new vertex
                elif candidateG >= G[neighbour]:
                    # This G score is worse than previously found
                    continue
                # Adopt this G score
                cameFrom[neighbour] = current
                G[neighbour] = candidateG
                H = graph.heuristic(neighbour, end)
                F[neighbour] = G[neighbour] + H

    def create_barriers_obs(self, coord_obstacle: iter, goal: tuple):
        """Documentation
        Return a list of lists representing the different obstacles
        Parameters:
            coord_obstacle: coordinates of the corners of each obstacles
        Out:
            list_barriers: List of lists representing the different obstacles with all their points
        """
        list_barriers: iter = []
        for coord in coord_obstacle:
            list_temp_1 = []
            list_temp_2 = []
            list_temp_3 = []
            list_temp_4 = []
            x_range = abs(coord[3] - coord[1])
            y_range = abs(coord[2] - coord[0])
            for x in range(x_range):
                list_temp_2.append((coord[1] + x, coord[2]))
                list_temp_4.append((coord[3] - x, coord[0]))
            for y in range(y_range):
                if coord[0] + y != goal[1]:
                    list_temp_1.append((coord[1], coord[0] + y))
                if coord[2] - y != goal[1]:
                    list_temp_3.append((coord[3], coord[2] - y))
            list_barriers.append(
                list_temp_1 + list_temp_2 + list_temp_3 + list_temp_4)
        return list_barriers

    def create_barriers_seat(self, corners_rect: iter, start: tuple):
        """Documentation
        Return a list of lists representing the different cluster of the seats
        Parameters:
            corners_rect: corners of the clusters representing the seats
        Out:
            list_corners: List of lists representing the different cluster with all the points of the outline
        """
        list_points = []
        for corners in corners_rect:
            x_range = corners[-1][0] - corners[0][0]
            y_range = corners[1][1] - corners[0][1]
            list_temp_1 = []
            list_temp_2 = []
            for x in range(x_range):
                list_temp_1.append((corners[1][0] + x, corners[0][1]))
                list_temp_2.append((corners[3][0] - x, corners[2][1]))
            list_temp_3 = []
            list_temp_4 = []
            for y in range(y_range):
                if corners[0][1] + y != start[1]:
                    list_temp_3.append((corners[0][0], corners[0][1] + y))
                if corners[2][1] - y != start[1]:
                    list_temp_4.append((corners[2][0], corners[2][1] - y))
            list_points.append(
                list_temp_1 + list_temp_2 + list_temp_3 + list_temp_4)
        return list_points

    def plane_contours(self, barriers_obs: iter, barriers_seat: iter):
        """Documentation
        Parameters:
            barriers_obs: List of lists representing the different obstacles with all their points
            barriers_seat: List of lists representing the different cluster with all the points of the outline
        Out:
            list_contours: List of the points of the outline representing the outline of the plane
        """
        x_min = y_min = np.inf
        x_max = y_max = -np.inf
        for barrier in barriers_seat:
            for point in barrier:
                if point[0] < x_min:
                    x_min = point[0]
                if point[0] > x_max:
                    x_max = point[0]
        for barrier in barriers_obs:
            for point in barrier:
                if point[1] < y_min:
                    y_min = point[1]
                if point[1] > y_max:
                    y_max = point[1]
        x_range = x_max - x_min
        y_range = y_max - y_min
        list_contours = []
        list_temp_1 = []
        list_temp_2 = []
        list_temp_3 = []
        list_temp_4 = []
        for y in range(y_range):
            list_temp_1.append((x_min, y_min + y))
            list_temp_3.append((x_max, y_max - y))
        for x in range(x_range):
            list_temp_2.append((x_min + x, y_max))
            list_temp_4.append((x_max - x, y_min))
        list_contours = list_temp_1 + list_temp_2 + list_temp_3 + list_temp_4
        return list_contours

    def pathfinder(self, start: tuple, goal: tuple, list_rect2: iter,
                   obstacles: iter):
        """Documentation
        Create the graph for the A* algorithm and calculate the best path
        Parameters:
            start: Start point for the A* algorithm
            goal: End point for the A* algorithm
            list_rect2: List of the corners of the cluster of the seat
            obstacles: List of the coordinates of the obstacles
        Out:
            path: Points of the best path
            cost: Cost of the path in pixel
        """
        barriers_seat = self.create_barriers_seat(list_rect2, start)
        barriers_obs = self.create_barriers_obs(obstacles, goal)
        outline = self.plane_contours(barriers_obs, barriers_seat)
        barriers = barriers_seat + barriers_obs + [outline]
        graph = AStarGraph(barriers)
        path, cost = self.AStarSearch(start, goal, graph)
        plt.figure(figsize=(40, 40))
        plt.plot([v[0] for v in path], [v[1] for v in path])
        for barrier in graph.barriers:
            plt.plot([v[0] for v in barrier], [v[1] for v in barrier],
                     color='red')
        plt.xlim(100, 400)
        plt.ylim(0, 1400)
        plt.show()
        return path, cost

    def draw_path(self, path: str, img: str, obs_number: int, seat_number: int,
                  obstacle: list, json_seat: dict):
        """Documentation
        Parameters:
            path: folder path
            img: image name
            obs_number: observation number
            seat_number: seat number
            obstacle: obstacles list, for each obstacle : oordinates of the top left and bottom right points
            json_seat: json
        """
        list_seat = []
        for i in list(json_seat.values()):
            list_seat += i

        dbscan = self.find_cluster(38, 3, list_seat)
        list_rect, list_rect2 = self.clusters_to_rect(
            dbscan, np.array(list_seat))

        fig = plt.figure(figsize=(20, 40))
        ax = fig.add_subplot(111, aspect='equal')

        for rect in list_rect:
            ax.add_patch(
                patches.Rectangle(rect[0], rect[1], rect[2]))

        img_cv = cv.imread(path + img)
        #
        for obs in obstacle:
            A_point = obs[1], obs[0]
            B_point = obs[3], obs[2]

            img_cv = cv.rectangle(img_cv, A_point, B_point, (255, 0, 0), 2)

        ob = list(self.pipeline_zone.json.values())[0]['rectangles'][obs_number]
        t_obs = [(ob[0], ob[1]), (ob[2], ob[3])]

        img_cv = cv.line(img_cv, self.centroid_seat(
            list_seat[seat_number]), self.centroid_obstacle(t_obs),
                          (255, 255, 0), 2)
        plt.imshow(img_cv)
        plt.show()

    def to_json_simple_distance(self, json_seat: dict, json_zone: dict):
        """Documentation
        Parameters:
            pipeline_zone.json: json ???
            pipeline.json: json ???
        Out:
            dicimg: json final structure
        """

        dicimg = {}

        # for each image in the json
        for img in list(json_zone.keys()):
            dicimg[img] = {}
            # for each type seat
            for typeseat in json_seat[img].keys():

                dicimg[img][typeseat] = {}
                # for each coordinate in a type seat
                for coord_seat in json_seat[img][typeseat]:
                    # get the centroid position of the seat
                    coord_centroid_seat = self.centroid_seat(coord_seat)
                    dicimg[img][typeseat][str(coord_seat)] = {}
                    # for each obstacle type
                    for obstacle_type in json_zone[img].keys():
                        # if there is obstacles of that type of obstacle
                        if len(json_zone[img][obstacle_type]) > 0:
                            dicimg[img][typeseat][str(coord_seat)][obstacle_type] = []
                            # for each coordinates in the obstacle type
                            for coord_obstacle_type in json_zone[img][obstacle_type]:
                                # get the centroid position of the obstacle
                                coord_centroid_obstacle = self.centroid_obstacle(
                                    coord_obstacle_type)

                                # calcualate the distance etween the seat and the obstacle
                                distance = self.dist_crow_flies(
                                    coord_centroid_seat,
                                    coord_centroid_obstacle)

                                obstacle_type_h_w_coord = (
                                    coord_obstacle_type[1], 
                                    coord_obstacle_type[0],
                                    abs(coord_obstacle_type[0]-coord_obstacle_type[2]),
                                    abs(coord_obstacle_type[1]-coord_obstacle_type[3])
                                )
                                    
                                # save this distance in the dict
                                dicimg[img][typeseat][str(coord_seat)][obstacle_type].append(
                                    [obstacle_type_h_w_coord, distance]
                                )
        return dicimg

