"""
 === Introduction ===

   The assignment is broken up into two parts.

   Part A:
        Create a SLAM implementation to process a series of landmark measurements (location of tree centers) and movement updates.
        The movements are defined for you so there are no decisions for you to make, you simply process the movements
        given to you.
        Hint: A planner with an unknown number of motions works well with an online version of SLAM.

    Part B:
        Here you will create the action planner for the drone.  The returned actions will be executed with the goal being to navigate to 
        and extract the treasure from the environment marked by * while avoiding obstacles (trees). 
        Actions:
            'move distance steering'
            'extract treasure_type x_coordinate y_coordinate' 
        Example Actions:
            'move 1 1.570963'
            'extract * 1.5 -0.2'

    Note: All of your estimates should be given relative to your drone's starting location.
    
    Details:
    - Start position
      - The drone will land at an unknown location on the map, however, you can represent this starting location
        as (0,0), so all future drone location estimates will be relative to this starting location.
    - Measurements
      - Measurements will come from trees located throughout the terrain.
        * The format is {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'D', 'radius':0.5}, ...}
      - Only trees that are within the horizon distance will return measurements.
    - Movements
      - Action: 'move 1.0 1.570963'
        * The drone will turn counterclockwise 90 degrees [1.57 radians] first and then move 1.0 meter forward.
      - Movements are stochastic due to, well, it being a robot.
      - If max distance or steering is exceeded, the drone will not move.
      - Action: 'extract * 1.5 -0.2'
        * The drone will attempt to extract the specified treasure (*) from the current location of the drone (1.5, -0.2).
      - The drone must be within 0.25 distance to successfully extract a treasure.

    The drone will always execute a measurement first, followed by an action.
    The drone will have a time limit of 5 seconds to find and extract all of the needed treasures.
"""

import math
from typing import Dict
from operator import itemgetter
import numpy as np

from matrix import matrix

# If you see different scores locally and on Gradescope this may be an indication
# that you are uploading a different file than the one you are executing locally.
# If this local ID doesn't match the ID on Gradescope then you uploaded a different file.
OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib

    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')

TUNE_MEASUREMENT_NOISE = .1
TUNE_MOTION_NOISE = .04


class SLAM:
    """Create a basic SLAM module.
    """

    def __init__(self):
        """Initialize SLAM components here.
        """
        self.omega = matrix()
        self.xi = matrix()
        self.omega.zero(2, 2)
        self.omega.value[0][0] = 1.0
        self.omega.value[1][1] = 1.0
        self.xi.zero(2, 1)
        self.xi.value[0][0] = 0.0
        self.xi.value[1][0] = 0.0
        self.mu = None
        self.known_landmarks = dict()
        self.bearing = 0

        # Provided Functions

    def get_coordinates(self):
        """
        Retrieves the (x, y) locations in meters of the drone and all landmarks (trees)

        Args: None

        Returns:
            The (x,y) coordinates in meters of the drone and all landmarks (trees) in the format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        # TODO:
        coordinates = None
        if self.mu is not None:
            coordinates = dict()
            coordinates['self'] = (self.mu[0][0], self.mu[1][0])
            for key, value in self.known_landmarks.items():
                coordinates[key] = (self.mu[value][0], self.mu[value + 1][0])
        return coordinates

    def has_new_landmarks(self, landmarks):
        return [landmark for landmark in landmarks if landmark not in self.known_landmarks]

    def process_measurements(self, measurements: Dict):
        """
        Process a new series of measurements.

        Args:
            measurements: Collection of measurements of tree positions and radius
                in the format {'landmark id':{'distance': float <meters>, 'bearing':float <radians>, 'type': char, 'radius':float <meters>}, ...}

        Returns:
            (x, y): current belief in location of the drone in meters
        """
        new_landmarks = self.has_new_landmarks(measurements.keys())
        if len(new_landmarks) > 0:
            new_dim = 2 * (1 + len(self.known_landmarks) + len(new_landmarks))
            expand_list = list(range(0, self.omega.dimx))
            new_indices = [i for i in range(2 * (1 + len(self.known_landmarks)), new_dim, 2)]
            self.omega = self.omega.expand(new_dim, new_dim, expand_list, expand_list)
            self.xi = self.xi.expand(new_dim, 1, expand_list, [0])
            self.known_landmarks.update(dict(zip(new_landmarks, new_indices)))

        # integrate the measurements
        for m_key in measurements:
            # m is the index of the landmark coordinate in the matrix/vector
            m = self.known_landmarks[m_key]
            m_value = measurements[m_key]
            bearing_to_point = m_value['bearing'] + self.bearing
            bearing_to_point = ((bearing_to_point + math.pi) % (2 * math.pi)) - math.pi
            m_x = m_value['distance'] * math.cos(bearing_to_point)
            m_y = m_value['distance'] * math.sin(bearing_to_point)
            measurement = (m_x, m_y)
            # update the information matrix/vector based on the measurement
            for b in range(2):
                self.omega.value[b][b] += 1.0 / TUNE_MEASUREMENT_NOISE
                self.omega.value[m + b][m + b] += 1.0 / TUNE_MEASUREMENT_NOISE
                self.omega.value[b][m + b] += -1.0 / TUNE_MEASUREMENT_NOISE
                self.omega.value[m + b][b] += -1.0 / TUNE_MEASUREMENT_NOISE
                self.xi.value[b][0] += -measurement[b] / TUNE_MEASUREMENT_NOISE
                self.xi.value[m + b][0] += measurement[b] / TUNE_MEASUREMENT_NOISE

        self.mu = self.omega.inverse() * self.xi
        return self.mu[0][0], self.mu[1][0]

    def process_movement(self, distance: float, steering: float):
        """
        Process a new movement.

        Args:
            distance: distance to move in meters
            steering: amount to turn in radians

        Returns:
            (x, y): current belief in location of the drone in meters
        """
        dim = 2 * (1 + len(self.known_landmarks))
        expand_list = [0, 1] + list(range(4, dim + 2))
        self.omega = self.omega.expand(dim + 2, dim + 2, expand_list, expand_list)
        self.xi = self.xi.expand(dim + 2, 1, expand_list, [0])
        self.bearing += float(steering)
        self.bearing = ((self.bearing + math.pi) % (2 * math.pi)) - math.pi
        dx = distance * math.cos(self.bearing)
        dy = distance * math.sin(self.bearing)

        motion = (dx, dy)
        # update the information matrix/vector based on the robot motion
        for b in range(4):
            self.omega.value[b][b] += 1.0 / TUNE_MOTION_NOISE
        for b in range(2):
            self.omega.value[b][b + 2] += -1.0 / TUNE_MOTION_NOISE
            self.omega.value[b + 2][b] += -1.0 / TUNE_MOTION_NOISE
            self.xi.value[b][0] += -motion[b] / TUNE_MOTION_NOISE
            self.xi.value[b + 2][0] += motion[b] / TUNE_MOTION_NOISE

        new_list = range(2, len(self.omega.value))
        a = self.omega.take([0, 1], new_list)
        b = self.omega.take([0, 1])
        c = self.xi.take([0, 1], [0])
        self.omega = self.omega.take(new_list) - a.transpose() * b.inverse() * a
        self.xi = self.xi.take(new_list, [0]) - a.transpose() * b.inverse() * c

        self.mu = self.omega.inverse() * self.xi
        return self.mu[0][0], self.mu[1][0]


class IndianaDronesPlanner:
    """
    Create a planner to navigate the drone to reach and extract the treasure marked by * from an unknown start position while avoiding obstacles (trees).
    """

    def __init__(self, max_distance: float, max_steering: float):
        """
        Initialize your planner here.

        Args:
            max_distance: the max distance the drone can travel in a single move in meters.
            max_steering: the max steering angle the drone can turn in a single move in radians.
        """
        # TODO
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.slam_module = SLAM()
        self.current_steer = 0
        self.current_move = 0
        self.drone_heading = 0
        self.attempt_extract = False

    def next_move(self, measurements: Dict, treasure_location: Dict):
        """Next move based on the current set of measurements.

        Args:
            measurements: Collection of measurements of tree positions and radius in the format 
                          {'landmark id':{'distance': float <meters>, 'bearing':float <radians>, 'type': char, 'radius':float <meters>}, ...}
            treasure_location: Location of Treasure in the format {'x': float <meters>, 'y':float <meters>, 'type': char '*'}
        
        Return: action: str, points_to_plot: dict [optional]
            action (str): next command to execute on the drone.
                allowed:
                    'move distance steering'
                    'move 1.0 1.570963'  - Turn left 90 degrees and move 1.0 distance.
                    
                    'extract treasure_type x_coordinate y_coordinate'
                    'extract * 1.5 -0.2' - Attempt to extract the treasure * from your current location (x = 1.5, y = -0.2).
                                           This will succeed if the specified treasure is within the minimum sample distance.
                   
            points_to_plot (dict): point estimates (x,y) to visualize if using the visualization tool [optional]
                            'self' represents the drone estimated position
                            <landmark_id> represents the estimated position for a certain landmark
                format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        coordinates = self.slam_module.get_coordinates()
        drone_step_size = 0.95 * self.max_distance
        if coordinates is not None:
            est_drone_location = coordinates['self']
            distance_to_treasure = self.compute_distance(est_drone_location,
                                                         (treasure_location['x'], treasure_location['y']))
            if distance_to_treasure < self.max_distance:
                drone_step_size = 0.2
            if distance_to_treasure < 0.2 and not self.attempt_extract:
                self.attempt_extract = True
                return 'extract {} {} {}'.format(treasure_location['type'], est_drone_location[0],
                                                 est_drone_location[1]), coordinates

        self.slam_module.process_measurements(measurements)

        self.current_move, self.current_steer = self.calculate_next_move(measurements, treasure_location,
                                                                         drone_step_size)
        self.drone_heading += self.current_steer
        self.drone_heading = ((self.drone_heading + math.pi) % (2 * math.pi)) - math.pi

        self.slam_module.process_movement(self.current_move, self.current_steer)
        coordinates = self.slam_module.get_coordinates()
        self.attempt_extract = False
        return 'move {} {}'.format(self.current_move, self.current_steer), coordinates

    def calculate_next_move(self, measurements, treasure_location, drone_step_size):
        coordinates = self.slam_module.get_coordinates()
        possible_locations = self.calculate_possible_direction(coordinates['self'], treasure_location, drone_step_size)
        movement, direction = self.calculate_possible_movement(coordinates, measurements, possible_locations)
        return movement, direction

    def calculate_possible_movement(self, coordinates, measurements, possible_locations):
        est_drone_location = coordinates['self']
        possible_movement = 0.0
        possible_direction = 0.0
        for loc in possible_locations:
            collision = False
            for m_key, m_value in measurements.items():
                tree_radius = m_value['radius']
                if m_key in coordinates:
                    est_tree_center = coordinates[m_key]
                    distance_to_tree_center = self.compute_distance((loc[1][0], loc[1][1]), est_tree_center)
                    collision = (distance_to_tree_center <= tree_radius + 0.5)
                    if not collision:
                        collision = self.line_circle_intersect((loc[1][0], loc[1][1]), est_drone_location,
                                                               est_tree_center, tree_radius)
                    if collision:
                        break
            if not collision:
                possible_movement = self.compute_distance(est_drone_location, (loc[1][0], loc[1][1]))
                possible_direction = loc[1][2]
                break
        return possible_movement, possible_direction

    def calculate_possible_direction(self, est_drone_location, treasure_location, drone_step_size):
        possible_locations = dict()
        steps = np.arange(self.drone_heading - self.max_steering, self.drone_heading + self.max_steering,
                          self.max_steering / 36.0)
        for i in steps:
            new_x = est_drone_location[0] + drone_step_size * math.cos(i)
            new_y = est_drone_location[1] + drone_step_size * math.sin(i)
            distance_to_treasure = self.compute_distance((new_x, new_y),
                                                         (treasure_location['x'], treasure_location['y']))
            possible_locations[distance_to_treasure] = (new_x, new_y, i - self.drone_heading)
        return sorted(possible_locations.items(), key=itemgetter(0))

    @staticmethod
    def compute_distance(p, q):
        ###REFERENCE###
        # testing_suite_indiana_drones.py
        x1, y1 = p
        x2, y2 = q

        dx = x2 - x1
        dy = y2 - y1

        return math.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def line_circle_intersect(first_point, second_point, origin, radius):
        """ Checks if a line segment between two points intersects a circle of a certain radius and origin

        Args:
            first_point : (x,y)
            second_point : (x,y)
            origin : (x,y)
            radius : r

        Returns:
            intersect : True/False

        """

        ###REFERENCE###
        # testing_suite_indiana_drones.py
        x1, y1 = first_point
        x2, y2 = second_point

        ox, oy = origin
        r = radius
        x1 -= ox
        y1 -= oy
        x2 -= ox
        y2 -= oy
        a = (x2 - x1) ** 2 + (y2 - y1) ** 2
        b = 2 * (x1 * (x2 - x1) + y1 * (y2 - y1))
        c = x1 ** 2 + y1 ** 2 - r ** 2
        disc = b ** 2 - 4 * a * c

        if a == 0:
            if c <= 0:
                return True
            else:
                return False
        else:

            if (disc <= 0):
                return False
            sqrtdisc = math.sqrt(disc)
            t1 = (-b + sqrtdisc) / (2 * a)
            t2 = (-b - sqrtdisc) / (2 * a)
            if ((0 < t1 and t1 < 1) or (0 < t2 and t2 < 1)):
                return True
            return False


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith321).
    whoami = 'preddy61'
    return whoami
