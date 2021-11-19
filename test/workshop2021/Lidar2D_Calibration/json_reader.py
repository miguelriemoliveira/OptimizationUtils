#!/usr/bin/python3

import json
import pprint
import math
import matplotlib.pyplot as plt


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


def dataViewer(data_left, data_right):
    """
    From the data of both LIDARs, plots them to give a visual aid
    Args:
        data_left: Dict
        data_right: Dict

    Returns:

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

    # Initializing and viewing the plot
    plt.plot(0, 0)
    plt.grid()
    plt.axis([-20, 20, -20, 20])

    plt.plot(left_xs, left_ys, 'bo')
    plt.plot(right_xs, right_ys, 'ro')
    plt.show()


def main():
    # Retrieve info from dictionary and plot it
    data_left, data_right = jsonImporter('data/data_collected.json')
    dataViewer(data_left, data_right)

if __name__ == "__main__":
    main()