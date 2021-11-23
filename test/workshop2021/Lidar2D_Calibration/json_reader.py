#!/usr/bin/python3

import json
import pprint
import math
import matplotlib.pyplot as plt


def jsonImporter(path):
    """
    Imports .json file 
    Args:
        path: String

    Returns:

    """
    # Opening JSON file
    f = open(path, )
    data = json.load(f)
    return data


def divideDict(data, col):
    """
    Divides the data into the one from the left LIDAR and the right LIDAR
    """
    
    # Dividing the dictionary in two
    data_left = data['collections'][col]['data']['left_laser']
    data_right = data['collections'][col]['data']['right_laser']
    
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


def dataTreatment(data_left, data_right):
    """
    From the data of both LIDARs, saves them on two lists of tuples
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
        left_xs.append(round(x,2))
        left_ys.append(round(y,2))

    for laser_range in right_ranges:
        x, y = pol2cart(laser_range, angle_right)
        angle_right += incangle_r
        right_xs.append(round(x,2))
        right_ys.append(round(y,2))

    return left_xs, left_ys, right_xs, right_ys
