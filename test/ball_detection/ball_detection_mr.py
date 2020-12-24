#!/usr/bin/env python
"""
This code shows how the optimizer class may be used for changing the colors of a set of images so that the average
color in all images is very similar. This is often called color correction.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import argparse  # to read command line arguments
import json
import math
import sys

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.KeyPressManager as KeyPressManager
import cv2
import copy

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities
from numpy import inf

# class Ball:
#     def __init__(self, radius, x, y):
#         self.radius = radius
#         self.x = x
#         self.y = y
#
#         self.xs = []
#         self.ys = []
#
#         self.nxs = []
#         self.nys = []
#
#
# class Canny:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    args = vars(ap.parse_args())
    print(args)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    eps = 2 * sys.float_info.epsilon
    eps = 0.1
    image = cv2.imread('images/4balls_clean.jpg')
    # image = cv2.imread('images/4balls.jpg')

    data = {'balls': {'0': {'radius': 70, 'center_x': image.shape[1] / 4, 'center_y': image.shape[0] / 4,
                            'points': [], 'neighbors': [], 'color': (255, 0, 0)},
                      '1': {'radius': 70, 'center_x': image.shape[1] / 4, 'center_y': image.shape[0] * 3 / 4,
                            'points': [], 'neighbors': [], 'color': (0, 0, 255)},
                      '2': {'radius': 70, 'center_x': image.shape[1] * 3 / 4, 'center_y': image.shape[0] * 3 / 4,
                            'points': [], 'neighbors': [], 'color': (0, 255, 0)},
                      '3': {'radius': 70, 'center_x': image.shape[1] * 3 / 4, 'center_y': image.shape[0] / 4,
                            'points': [], 'neighbors': [], 'color': (0, 0, 0)}
                      },
            'angles': (np.linspace(0, np.pi * 2, 100)).tolist(),
            'canny': {'x': 100, 'y': 200, 'edges': None},
            }

    with open('data.json', 'w') as fh:
        json.dump(data, fh)

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('data', data)

    # Push parameters related to the balls
    def getterBall(data, ball_key):
        d = data['balls'][ball_key]
        return [d['center_x'], d['center_y'], d['radius']]

    def setterBall(data, values, ball_key):
        d = data['balls'][ball_key]
        d['center_x'] = values[0]
        d['center_y'] = values[1]
        d['radius'] = values[2]

    for ball_key, ball in data['balls'].items():
        print(ball_key)
        opt.pushParamVector('ball' + '_' + ball_key + '_', 'data', partial(getterBall, ball_key=ball_key),
                            partial(setterBall, ball_key=ball_key),
                            bound_max=None, bound_min=None,
                            suffix=['center_x', 'center_y', 'radius'])

    # Push canny related parameters
    def getterCanny(data):
        return [data['canny']['x'], data['canny']['y']]


    def setterCanny(data, values):
        data['canny']['t1'] = values[0]
        data['canny']['t2'] = values[1]


    opt.pushParamVector('canny_', 'data', getterCanny, setterCanny,
                        bound_max=None, bound_min=None,
                        suffix=['t1', 't2'])

    opt.printParameters()


    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def find_nearest_white(img, target):
        nonzero = np.argwhere(img == 255)
        distances = np.sqrt((nonzero[:, 0] - target[0]) ** 2 + (nonzero[:, 1] - target[1]) ** 2)
        nearest_index = np.argmin(distances)
        return nonzero[nearest_index]


    def objectiveFunction(models):
        data = models['data']

        # Recompute the edge image using the new canny parameters
        data['canny']['edges'] = cv2.Canny(image, data['canny']['t1'], data['canny']['t2'])

        # initialize errors dictionary
        errors = {}

        for ball_key, ball in data['balls'].items():
            ball['points'] = []
            ball['neighbors'] = []
            for angle_idx, angle in enumerate(data['angles']):
                x = ball['center_x'] + ball['radius'] * math.cos(angle)
                y = ball['center_y'] + ball['radius'] * math.sin(angle)

                ball['points'].append({'x': x, 'y': y})
                ny, nx = find_nearest_white(data['canny']['edges'], [y, x])
                ball['neighbors'].append({'x': nx, 'y': ny})

                residual_name = ball_key + '_a' + str(angle_idx)
                errors[residual_name] = math.sqrt(pow(x - nx, 2) + pow(y - ny, 2))

        residual_name = 'canny_total_whites'
        errors[residual_name] = cv2.countNonZero(data['canny']['edges']) / 100.

        return errors


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for ball_key, ball in data['balls'].items():
        for angle_idx, angle in enumerate(data['angles']):
            residual_name = ball_key + '_a' + str(angle_idx)
            params = opt.getParamsContainingPattern('ball' + '_' + ball_key + '_')
            opt.pushResidual(name=residual_name, params=params)

    residual_name = 'canny_total_whites'
    params = opt.getParamsContainingPattern('canny_')
    opt.pushResidual(name=residual_name, params=params)

    print('residuals = ' + str(opt.residuals))
    opt.computeSparseMatrix()
    opt.printSparseMatrix()

    # opt.callObjectiveFunction()
    opt.printParameters()
    opt.printResiduals()
    # exit(0)
    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------
    fig = plt.figure()

    cv2.namedWindow('ball_detection')
    cv2.imshow('ball_detection', image)

    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0.01, verbose=False):
        exit(0)


    def visualizationFunction(models):
        data = models['data']
        gui_image = copy.deepcopy(image)

        # draw ball perimeter
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]
        for ball_key, ball in data['balls'].items():
            for point_idx, point in enumerate(ball['points']):
                x, y = int(point['x']), int(point['y'])
                cv2.line(gui_image, (x, y), (x, y), ball['color'], 2)

        # draw line to closest white point
        for ball_key, ball in data['balls'].items():
            for point_idx, point in enumerate(ball['points']):
                x, y = int(point['x']), int(point['y'])
                nx = int(ball['neighbors'][point_idx]['x'])
                ny = int(ball['neighbors'][point_idx]['y'])
                cv2.line(gui_image, (x, y), (nx, ny), (255, 128, 0), 2)

        # Draw detected edges as greens pixels
        h, w, nc = gui_image.shape
        for pix_x in range(0, w):
            for pix_y in range(0, h):
                if data['canny']['edges'][pix_y, pix_x] == 255:
                    gui_image[pix_y, pix_x] = (0, 255, 0)
        cv2.imshow('ball_detection', gui_image)
        cv2.waitKey(20)

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False) == 'x':
            print("exiting")
            exit(0)


    opt.setVisualizationFunction(visualizationFunction, True)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # opt.fromXToData()
    # opt.callObjectiveFunction()
    # wm = KeyPressManager.KeyPressManager.WindowManager()
    # if wm.waitForKey():
    #     exit(0)
    #
    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-3})

    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0.01, verbose=False) == 'x':
        print("exiting")
        exit(0)
