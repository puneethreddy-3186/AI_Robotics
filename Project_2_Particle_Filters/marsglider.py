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

# These import statements give you access to library functions which you may
# (or may not?) want to use.
from math import *
from glider import *
import numpy as np
import math

# If you see different scores locally and on Gradescope this may be an indication
# that you are uploading a different file than the one you are executing locally.
# If this local ID doesn't match the ID on Gradescope then you uploaded a different file.
OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib

    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')


# This is the function you will have to write for part A.
# -The argument 'height' is a floating point number representing
# the number of meters your glider is above the average surface based upon 
# atmospheric pressure. (You can think of this as height above 'sea level'
# except that Mars does not have seas.) Note that this sensor may be
# slightly noisy.
# This number will go down over time as your glider slowly descends.
#
# -The argument 'radar' is a floating point number representing the
# number of meters your glider is above the specific point directly below
# your glider based off of a downward facing radar distance sensor. Note that
# this sensor has random Gaussian noise which is different for each read.

# -The argument 'mapFunc' is a function that takes two parameters (x,y)
# and returns the elevation above "sea level" for that location on the map
# of the area your glider is flying above.  Note that although this function
# accepts floating point numbers, the resolution of your map is 1 meter, so
# that passing in integer locations is reasonable.
#
#
# -The argument OTHER is initially None, but if you return an OTHER from
# this function call, it will be passed back to you the next time it is
# called, so that you can use it to keep track of important information
# over time.
#

def estimate_next_pos(height, radar, mapFunc, OTHER=None):
    """Estimate the next (x,y) position of the glider."""

    p = initialize_particles(height) if OTHER is None else OTHER['particles']
    time_step = 0 if OTHER is None else OTHER['time_step'] + 1

    weights = calculate_weights(p, height, radar, mapFunc)
    p = resample(p, weights)
    p = fuzz_particles(p)
    p = move_particles(p, 5.0)

    # example of how to find the actual elevation of a point of ground from the map:
    # actualElevation = mapFunc(5, 4)

    # You must return a tuple of (x,y) estimate, and OTHER (even if it is NONE)
    # in this order for grading purposes.
    #
    xy_estimate = (0, 0)  # Sample answer, (X,Y) as a tuple.

    # TODO - remove this canned answer which makes this template code
    # pass one test case once you start to write your solution....
    if p.shape[0] > 0:
        xy_estimate = (np.median(p[:, 0]), np.median(p[:, 1]))

    # You may optionally also return a list of (x,y,h) points that you would like
    # the PLOT_PARTICLES=True visualizer to plot for visualization purposes.
    # If you include an optional third value, it will be plotted as the heading
    # of your particle.
    optionalPointsToPlot = p[:, [0, 1, 3]]
    if OTHER is None:
        OTHER = dict()
    OTHER['particles'] = p
    OTHER['time_step'] = time_step
    return xy_estimate, OTHER, optionalPointsToPlot


def initialize_particles(height):
    tune_no_particles = 25000
    particles = np.empty((tune_no_particles, 4))
    particles[:, 0] = np.random.uniform(-250, 250, tune_no_particles)
    particles[:, 1] = np.random.uniform(-250, 250, tune_no_particles)
    particles[:, 2] = height
    particles[:, 3] = np.random.normal(0, pi / 4, tune_no_particles)
    return particles


def gaussian(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(-((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))


def calculate_weights(particles, height, radar, mapFunc):
    tune_weights_sigma = 75
    weights = np.zeros(particles.shape[0])
    for i in range(particles.shape[0]):
        p_radar = particles[i, 2] - mapFunc(particles[i, 0], particles[i, 1])
        weights[i] = 1 * gaussian(p_radar, tune_weights_sigma, radar)
    return weights


def move_particles(particles, speed):
    particles[:, 0] += speed * np.cos(particles[:, 3])
    particles[:, 1] += speed * np.sin(particles[:, 3])
    particles[:, 2] -= 1.0
    return particles


def resample(particles, weights):
    tune_resample_size = 800
    new_particles = np.empty((tune_resample_size, 4))
    index = int(random.random() * tune_resample_size) % tune_resample_size
    beta = 0.0
    mw = max(weights)
    for i in range(tune_resample_size):
        beta += random.random() * 2.0 * mw
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % tune_resample_size
        new_particles[i] = particles[index, :]
    return new_particles


def fuzz_particles(particles):
    tune_particle_step = 3
    tune_particle_heading = np.pi / 32
    fuzz_array = np.concatenate((
        np.random.normal(0, tune_particle_step, (particles.shape[0], 1)),
        np.random.normal(0, tune_particle_step, (particles.shape[0], 1)),
        np.random.normal(0, 0, (particles.shape[0], 1)),
        np.random.normal(0, tune_particle_heading, (particles.shape[0], 1)),
    ),
        axis=1
    )
    particles += fuzz_array
    particles[:, 3] = ((particles[:, 3] + pi) % (pi * 2)) - pi
    return particles


# This is the function you will have to write for part B. The goal in part B
# is to navigate your glider towards (0,0) on the map steering # the glider 
# using its rudder. Note that the Z height is unimportant.

#
# The input parameters are exactly the same as for part A.

def next_angle(height, radar, mapFunc, OTHER=None):
    # How far to turn this timestep, limited to +/-  pi/8, zero means no turn.
    p = initialize_particles(height) if OTHER is None else OTHER['particles']
    time_step = 0 if OTHER is None else OTHER['time_step'] + 1
    previous_dist = -1.0 if OTHER is None else OTHER['previous_target_distance']
    u_turn_point_dist = -1.0 if OTHER is None or 'u_turn_point_distance' not in OTHER.keys() else OTHER[
        'u_turn_point_distance']

    weights = calculate_weights(p, height, radar, mapFunc)
    p = resample(p, weights)
    p = fuzz_particles(p)
    p = move_particles(p, 5.0)
    # You may optionally also return a list of (x,y)  or (x,y,h) points that
    # you would like the PLOT_PARTICLES=True visualizer to plot.
    #
    # optionalPointsToPlot = [ (1,1), (20,20), (150,150) ]  # Sample plot points
    # return steering_angle, OTHER, optionalPointsToPlot
    steering_angle = 0.0
    current_dist = 0.0
    if time_step > 40:
        heading = 0.0
        xy_estimate = (0, 0)
        if p.shape[0] > 0:
            xy_estimate = (np.median(p[:, 0]), np.median(p[:, 1]))
            heading = np.median(p[:, 3])

        current_dist = np.sqrt((xy_estimate[1] - 0) ** 2 + (xy_estimate[0] - 0) ** 2)
        slope_to_estimate = math.atan2((xy_estimate[1] - 0), (xy_estimate[0] - 0))
        dge = np.degrees(heading)
        if current_dist > previous_dist > 0:
            # u turn necessary
            u_turn_point_dist = current_dist
        if current_dist > u_turn_point_dist - 8 > 0:
            steering_angle = -pi / 8 if slope_to_estimate > 0 else pi / 8
        elif heading < 0.0:
            heading_prime = heading + pi
            slope_to_estimate = math.atan2((xy_estimate[1] - 0), (xy_estimate[0] - 0))
            if heading_prime < pi / 2:
                steering_angle = slope_to_estimate - heading_prime
                steering_angle = pi / 8 if steering_angle > (pi / 8) else steering_angle
            else:
                steering_angle = heading_prime - slope_to_estimate
                steering_angle = -pi / 8 if steering_angle > (pi / 8) else -steering_angle

        p[:, 3] += steering_angle
        p[:, 3] = ((p[:, 3] + pi) % (pi * 2)) - pi

    # You may optionally also return a list of (x,y,h) points that you would like
    # the PLOT_PARTICLES=True visualizer to plot for visualization purposes.
    # If you include an optional third value, it will be plotted as the heading
    # of your particle.
    # optionalPointsToPlot = p[:, [0, 1, 3]]
    if OTHER is None:
        OTHER = dict()
    OTHER['particles'] = p
    OTHER['time_step'] = time_step
    OTHER['previous_target_distance'] = current_dist
    if u_turn_point_dist > 0.0:
        OTHER['u_turn_point_distance'] = u_turn_point_dist
    return steering_angle, OTHER


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith321).
    whoami = 'preddy61'
    return whoami
