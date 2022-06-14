######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

# Optional: You may use deepcopy to help prevent aliasing
# from copy import deepcopy

# You may use either the numpy library or Sebastian Thrun's matrix library for
# your matrix math in this project; uncomment the import statement below for
# the library you wish to use.
# import numpy as np
# from matrix import matrix

# If you see different scores locally and on Gradescope this may be an indication
# that you are uploading a different file than the one you are executing locally.
# If this local ID doesn't match the ID on Gradescope then you uploaded a different file.
from matrix import matrix
import numpy as np
import math

OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib

    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')


class Turret(object):
    """The laser used to defend against invading Meteorites."""

    def __init__(self, init_pos, max_angle_change, dt):
        """Initialize the Turret."""
        self.x_pos = init_pos['x']
        self.y_pos = init_pos['y']
        self.max_angle_change = max_angle_change
        self.dt = dt
        self.estimates = dict()
        self.uncertainties = dict()
        self.alive_meteors = list()
        self.attempted_to_shoot = dict()
        self.turret_meteor_distance = dict()

    def observe_and_estimate(self, noisy_meteorite_observations):
        """Observe the locations of the Meteorites.

        self is a reference to the current object, the Turret.
        noisy_meteorite_observations is a list of observations of meteorite
        locations.  Each observation in noisy_meteorite_observations is a tuple
        (i, x, y), where i is the unique ID for an meteorite, and x, y are the
        x, y locations (with noise) of the current observation of that
        meteorite at this timestep. Only meteorites that are currently
        'in-bounds' will appear in this list, so be sure to use the meteorite
        ID, and not the position/index within the list to identify specific
        meteorites. (The list may change in size as meteorites move in and out
        of bounds.) In this function, return the estimated meteorite locations
        (one timestep into the future) as a tuple of (i, x, y) tuples, where i
        is a meteorite's ID, x is its x-coordinate, and y is its y-coordinate.
        """
        # TODO: Update the Turret's estimate of where the meteorites are
        # located at the current timestep and return the updated estimates
        u = matrix([[0.], [0.], [0.], [0.], [0.]])
        F = matrix([[1., 0., 0.1, 0., 0.00167],
                    [0., 1., 0., 0.1, 0.005],
                    [0., 0., 1., 0., 0.0333],
                    [0., 0., 0., 1., 0.1],
                    [0., 0., 0., 0., 1.]])
        H = matrix([[1., 0., 0., 0., 0.],
                    [0., 1, 0., 0., 0.]])
        I = matrix([[1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 1.]])
        R = matrix([[0.0055, 0.], [0., 0.0081]])
        estimated_pos = []
        self.alive_meteors.clear()
        for entry in noisy_meteorite_observations:
            m_id = entry[0]
            Z = matrix([[entry[1], entry[2]]])
            if m_id in self.estimates:
                x = matrix([[self.estimates[m_id][0]], [self.estimates[m_id][1]], [self.estimates[m_id][2]],
                            [self.estimates[m_id][3]], [self.estimates[m_id][4]]])
                P = self.uncertainties[m_id]
            else:
                x = matrix([[entry[1]], [entry[2]], [0.], [0.], [0.]])
                var_x = (entry[1] + .25) - (entry[1] - .25)
                var_y = (entry[2] + .25) - (entry[2] - .25)
                P = matrix([[var_x, 0., 0., 0., 0.],
                            [0., var_y, 0., 0., 0.],
                            [0., 0., var_x, 0., 0.],
                            [0., 0., 0., var_y, 0.],
                            [0., 0., 0., 0., var_x]])

            est_x, est_P = self.filter(x, F, P, H, R, I, u, Z)
            self.uncertainties[m_id] = est_P
            self.estimates[m_id] = (est_x[0][0], est_x[1][0], est_x[2][0], est_x[3][0], est_x[4][0])
            if m_id != -1 and m_id not in self.alive_meteors:
                self.alive_meteors.append(m_id)
            estimated_pos.append((m_id, est_x[0][0], est_x[1][0]))

        return tuple(estimated_pos)

    @staticmethod
    def filter(x, F, P, H, R, I, u, Z):
        # prediction
        x = (F * x) + u
        P = (F * P) * F.transpose()

        # measurement update
        y = Z.transpose() - (H * x)
        S = (H * P) * H.transpose() + R
        K = (P * H.transpose()) * S.inverse()
        x = x + (K * y)
        P = (I - (K * H)) * P
        return x, P

    def get_laser_action(self, current_aim_rad):
        """Return the laser's action; it can change its aim angle or fire.

        self is a reference to the current object, the Turret.
        current_aim_rad is the laser turret's current aim angle, in radians,
        provided by the simulation.

        The laser can aim in the range [0.0, pi].
        The maximum amount the laser's aim angle can change in a given timestep
        is self.max_angle_change radians. Larger change angles will be
        clamped to self.max_angle_change, but will keep the same sign as the
        returned desired angle change (e.g. an angle change of -3.0 rad would
        be clamped to -self.max_angle_change).
        If the laser is aimed at 0.0 rad, it will point horizontally to the
        right; if it is aimed at pi rad, it will point to the left.
        If the value returned from this function is the string 'fire' instead
        of a numerical angle change value, the laser will fire instead of
        moving.
        Returns: Float (desired change in laser aim angle, in radians), OR
        String 'fire' to fire the laser
        """
        # TODO: Update the change in the laser aim angle, in radians, based
        # on where the meteorites are currently, OR return 'fire' to fire the
        # laser at a meteorite
        action = 0.0
        if len(self.alive_meteors) == 0:
            return action

        closest_meteor = self.find_closest_meteor(current_aim_rad)
        if closest_meteor is None:
            return action

        return self.generate_action(closest_meteor, current_aim_rad)  # angle or 'fire'

    def generate_action(self, meteor_to_shoot, current_aim_rad):
        action = current_aim_rad - math.atan2((meteor_to_shoot[2] - -1), (meteor_to_shoot[1] - 0))
        if abs(action) > self.max_angle_change:
            action = -self.max_angle_change if action > 0 else self.max_angle_change
        else:
            if abs(action) <= 0.025:
                if meteor_to_shoot[0] not in self.attempted_to_shoot:
                    self.attempted_to_shoot[meteor_to_shoot[0]] = 1
                else:
                    self.attempted_to_shoot[meteor_to_shoot[0]] = self.attempted_to_shoot[meteor_to_shoot[0]] + 1
                action = 'fire'
        return action

    def find_closest_meteor(self, current_aim_rad):
        filtered_meteors = [(k, self.estimates[k][0], self.estimates[k][1]) for k in self.estimates if
                            k in self.alive_meteors and self.is_meteor_in_range(self.estimates[k][0],
                                                                                self.estimates[k][1])]
        # ignore fast moving meteors
        meteors_to_attack = [k for k in filtered_meteors if k[2] < -0.5 and
                             (k[0] not in self.turret_meteor_distance or
                             abs(self.turret_meteor_distance[k[0]] - k[2]) > 0.0001)]
        angle_closest_meteor_rad = 4.0
        closest_meteor = None
        for meteor in meteors_to_attack:
            self.turret_meteor_distance[meteor[0]] = meteor[2]
            angle_2_turret = math.atan2((meteor[2] - -1), (meteor[1] - 0))
            angle_diff = current_aim_rad - angle_2_turret
            if abs(angle_diff) < angle_closest_meteor_rad and (meteor[0] not in self.attempted_to_shoot or
                                                               self.attempted_to_shoot[meteor[0]] <= 7):
                angle_closest_meteor_rad = abs(angle_diff)
                closest_meteor = meteor
        return closest_meteor

    @staticmethod
    def get_distance(pos_x, posy):
        return ((-1 - posy) ** 2
                + (0 - pos_x) ** 2) ** 0.5

    @staticmethod
    def is_meteor_in_range(pos_x, posy):
        return 0.9 > ((-1 - posy) ** 2
                      + (0 - pos_x) ** 2) ** 0.5


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith321).
    whoami = 'preddy61'
    return whoami
