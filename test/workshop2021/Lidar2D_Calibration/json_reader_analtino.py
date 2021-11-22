#!/usr/bin/python3

import json
import pprint
import math
import sys

import matplotlib.pyplot as plt
import numpy as np


def jsonImporter(path):
    """
    Imports .json file and divides the data into the one from the left LIDAR and the right LIDAR
    Args:
        path: String

    Returns:

    """
    # Opening JSON file
    f = open(path, )
    data = json.load(f)

    # Dividing the dictionary in two
    data_left = data['collections']['0']['data']['left_laser']
    data_right = data['collections']['0']['data']['right_laser']

    return data_left, data_right


def pol2cart(rho, phi):
    """
    Converts polar coordinates into cartesian coordinates
    Args:
        rho: Int
        phi: Int

    Returns:

    """
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return x, y


def dataConverter(data_left, data_right):
    """
    From the data of both LIDARs, plots them to give a visual aid
    Args:
        data_left: Dict
        data_right: Dict

    Returns: left_xs, left_ys, right_xs, right_ys

    """


    # Retrieving data from the dictionary
    minangle_l = data_left['angle_min']
    incangle_l = data_left['angle_increment']
    minangle_r = data_right['angle_min']
    incangle_r = data_right['angle_increment']
    left_ranges = data_left['ranges']
    right_ranges = data_right['ranges']

    # Defining variables
    angle_left = minangle_l
    angle_right = minangle_r
    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []

    # Converting from polar coordinates to cartesian coordinates
    for laser_range in left_ranges:
        x, y = pol2cart(laser_range, angle_left)
        angle_left += incangle_l
        left_xs.append(x)
        left_ys.append(y)

    for laser_range in right_ranges:
        x, y = pol2cart(laser_range, angle_right)
        angle_right += incangle_r
        right_xs.append(x)
        right_ys.append(y)

    return left_xs, left_ys, right_xs, right_ys

    # # Initializing and viewing the plot
    # plt.plot(0, 0)
    # plt.grid()
    # plt.axis([-20, 20, -20, 20])
    #
    # plt.plot(left_xs, left_ys, 'bo')
    # plt.plot(right_xs, right_ys, 'ro')
    # plt.show()

class LineModel():
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def getY(self, x):
        if type(self.m) is list:
            m = self.m[0]
        else:
            m = self.m

        if type(self.b) is list:
            b = self.b[0]
        else:
            b = self.b

        return m * x + b

    def getYs(self, xs):
        return [self.getY(item) for item in xs]



def main():
    # -----------------------------------------------------
    # INITIALIZATIONxs_right
    # -----------------------------------------------------
    # -----------------------------------------------------

    # Create line model
    line_model = LineModel(1, 0.5)
    best_line_model_left = LineModel(1, 0.5)
    best_line_model_right = LineModel(1, 0.5)

    total_best_error = sys.float_info.max

    # Visualization
    # create a figure
    plt.plot(0, 0)
    plt.grid()
    plt.axis([-20, 20, -20, 20])

    # Retrieve info from dictionary and plot it
    data_left, data_right = jsonImporter('data/data_collected.json')
    xs_observations_left = dataConverter(data_left, data_right)[0]
    ys_observations_left = dataConverter(data_left, data_right)[1]
    xs_observations_right = dataConverter(data_left, data_right)[2]
    ys_observations_right = dataConverter(data_left, data_right)[3]

    # draw observations
    plt.plot(xs_observations_left, ys_observations_left, 'bo')
    plt.plot(xs_observations_right, ys_observations_right, 'co')

    handle_error_anchors_left = plt.plot(xs_observations_left, ys_observations_left, '.r')
    handle_error_anchors_right = plt.plot(xs_observations_right, ys_observations_right, '.r')

    # draw line model
    xs_left = list(np.linspace(-20, 20, 200))
    ys = line_model.getYs(xs_left)
    plt.plot(xs_left, ys, '-.k')
    handle_model_left_plot = plt.plot(xs_left, ys, '-.k')

    # draw parabola model
    xs_right = list(np.linspace(-20, 20, 200))
    ys = line_model.getYs(xs_right)
    handle_model_right_plot = plt.plot(xs_right, ys, '-.k')

    # draw best line model left
    handle_best_model_plot_left = plt.plot(xs_observations_left, best_line_model_left.getYs(xs_observations_left), '-r')

    # draw best line model right
    handle_best_model_plot_right = plt.plot(xs_observations_right, best_line_model_right.getYs(xs_observations_right), '-m')
 # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------
    while True:
        # randomize model parameters
        line_model.m = np.random.uniform(-2, 2)
        line_model.b = np.random.uniform(-20, 20)

        # compute error as vertical distance
        ys_observations_from_model_left = line_model.getYs(xs_observations_left)
        ys_observations_from_model_right = line_model.getYs(xs_observations_right)

       # errors = [abs(y_o - y_ofm) for y_o, y_ofm in zip(ys_observations, ys_observations_from_model)]
        errors = []

        # errors from the line model left
        for y_o, y_ofm in zip(ys_observations_left, ys_observations_from_model_left):
            error = abs(y_o - y_ofm)
            errors.append(error)

        # errors from the line model right
        for y_o, y_ofm in zip(ys_observations_right, ys_observations_from_model_right):
            error = abs(y_o - y_ofm)
            errors.append(error)

        total_error = sum(errors)
        print(total_error)

        # update best model (if needed)
        if total_error < total_best_error:  # found better line parameters
            total_best_error = total_error

            best_line_model_left.m = line_model.m
            best_line_model_left.b = line_model.b

            best_line_model_right.m = line_model.m
            best_line_model_right.b = line_model.b


        # Visualization
        # line left
        plt.setp(handle_model_left_plot, data=(xs_left, line_model.getYs(xs_left)))  # update the line draw
        plt.setp(handle_best_model_plot_left, data=(xs_left, best_line_model_left.getYs(xs_left)))  # update the best line draw
        plt.setp(handle_error_anchors_left, data=(xs_observations_left, ys_observations_from_model_left))  # update the anchor erros left model

        # line right
        plt.setp(handle_model_right_plot, data=(xs_right, line_model.getYs(xs_right)))  # update the line draw
        plt.setp(handle_best_model_plot_right, data=(xs_right, best_line_model_right.getYs(xs_right)))  # update the best line draw
        plt.setp(handle_error_anchors_right, data=(xs_observations_right, ys_observations_from_model_right))  # update the anchor erros left model

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not key is None:
            print('Pressed a key, terminating')
            break

if __name__ == "__main__":
    main()