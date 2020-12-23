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


class Ball:
    def __init__(self, radius, x, y):
        self.radius = radius
        self.x = x
        self.y = y

        self.xs = []
        self.ys = []

        self.nxs = []
        self.nys = []


class Canny:
    def __init__(self, x, y):
        self.x = x
        self.y = y


if __name__ == "__main__":

    image = cv2.imread('4balls_clean.jpg')

    # cv2.waitKey(0)
    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    # It will be the number of polynomial degree
    # for now, i will work with 4

    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    args = vars(ap.parse_args())
    print(args)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    # ball = Ball(radius=500,x=image.shape[1]/2,y=image.shape[0]/2)
    ball1 = Ball(radius=70, x=image.shape[1] / 4, y=image.shape[0] / 4)
    ball2 = Ball(radius=70, x=image.shape[1] / 4, y=3 * image.shape[0] / 4)
    ball3 = Ball(radius=70, x=3 * image.shape[1] / 4, y=3 * image.shape[0] / 4)
    ball4 = Ball(radius=70, x=3 * image.shape[1] / 4, y=image.shape[0] / 4)
    canny = Canny(x=100, y=200)

    edges = cv2.Canny(image, canny.x, canny.y)
    # cv2.imshow('edges', edges)

    angles = (np.linspace(0, np.pi * 2, 100)).tolist()

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('ball1', ball1)
    opt.addDataModel('ball2', ball2)
    opt.addDataModel('ball3', ball3)
    opt.addDataModel('ball4', ball4)
    opt.addDataModel('edges', edges)
    opt.addDataModel('canny', canny)


    # Create specialized getter and setter functions
    def setter(ball, value, field):
        if field == 'radius':
            ball.radius = value
        elif field == 'x':
            ball.x = value
        elif field == 'y':
            ball.y = value


    def getter(ball, field):
        if field == 'radius':
            return [ball.radius]
        elif field == 'x':
            return [ball.x]
        elif field == 'y':
            return [ball.y]


    eps = 2 * sys.float_info.epsilon
    eps = 0.1
    balls = [ball1, ball2, ball3, ball4]
    for idx, ball in enumerate(['ball1', 'ball2', 'ball3', 'ball4']):
        for field, bound_min in zip(['radius', 'x', 'y'], [50, 0, 0]):
            # for field, bound_min,bound_max in zip(['radius', 'x', 'y'], [balls[idx].radius-eps, balls[idx].x-eps, balls[idx].y-eps],[balls[idx].radius+eps, balls[idx].x+eps, balls[idx].y+eps]):
            opt.pushParamScalar(group_name=ball + '_' + field, data_key=ball, getter=partial(getter, field=field),
                                setter=partial(setter, field=field), bound_min=bound_min)

    opt.pushParamScalar(group_name='canny_x', data_key='canny', getter=partial(getter, field='x'),
                        setter=partial(setter, field='x'))
    opt.pushParamScalar(group_name='canny_y', data_key='canny', getter=partial(getter, field='y'),
                        setter=partial(setter, field='y'))
    opt.printParameters()


    # exit(0)

    def find_nearest_white(img, target):
        nonzero = np.argwhere(img == 255)
        distances = np.sqrt((nonzero[:, 0] - target[0]) ** 2 + (nonzero[:, 1] - target[1]) ** 2)
        nearest_index = np.argmin(distances)
        return nonzero[nearest_index]
        # ---------------------------------------


    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(models):
        ball1 = models['ball1']
        ball2 = models['ball2']
        ball3 = models['ball3']
        ball4 = models['ball4']
        # edges = models['edges']
        canny = models['canny']

        edges = cv2.Canny(image, canny.x[0], canny.y[0])
        models['edges'] = edges

        xs_1 = []
        ys_1 = []
        xs_2 = []
        ys_2 = []
        xs_3 = []
        ys_3 = []
        xs_4 = []
        ys_4 = []

        nxs_1 = []
        nys_1 = []
        nxs_2 = []
        nys_2 = []
        nxs_3 = []
        nys_3 = []
        nxs_4 = []
        nys_4 = []

        error_1 = []
        error_2 = []
        error_3 = []
        error_4 = []
        for angle in angles:
            x_1 = ball1.x[0] + ball1.radius[0] * math.cos(angle)
            y_1 = ball1.y[0] + ball1.radius[0] * math.sin(angle)
            x_2 = ball2.x[0] + ball2.radius[0] * math.cos(angle)
            y_2 = ball2.y[0] + ball2.radius[0] * math.sin(angle)
            x_3 = ball3.x[0] + ball3.radius[0] * math.cos(angle)
            y_3 = ball3.y[0] + ball3.radius[0] * math.sin(angle)
            x_4 = ball4.x[0] + ball4.radius[0] * math.cos(angle)
            y_4 = ball4.y[0] + ball4.radius[0] * math.sin(angle)

            xs_1.append(x_1)
            ys_1.append(y_1)
            xs_2.append(x_2)
            ys_3.append(y_2)
            xs_3.append(x_3)
            ys_3.append(y_3)
            xs_4.append(x_4)
            ys_4.append(y_4)

            ny_1, nx_1 = find_nearest_white(edges, [y_1, x_1])
            ny_2, nx_2 = find_nearest_white(edges, [y_2, x_2])
            ny_3, nx_3 = find_nearest_white(edges, [y_3, x_3])
            ny_4, nx_4 = find_nearest_white(edges, [y_4, x_4])

            nxs_1.append(nx_1)
            nys_1.append(ny_1)
            distance_1 = math.sqrt(pow(x_1 - nx_1, 2) + pow(y_1 - ny_1, 2))
            error_1.append(distance_1)

            nxs_2.append(nx_2)
            nys_2.append(ny_2)
            distance_2 = math.sqrt(pow(x_2 - nx_2, 2) + pow(y_2 - ny_2, 2))
            error_2.append(distance_2)

            nxs_3.append(nx_3)
            nys_3.append(ny_3)
            distance_3 = math.sqrt(pow(x_3 - nx_3, 2) + pow(y_3 - ny_3, 2))
            error_3.append(distance_3)

            nxs_4.append(nx_4)
            nys_4.append(ny_4)
            distance_4 = math.sqrt(pow(x_4 - nx_4, 2) + pow(y_4 - ny_4, 2))
            error_4.append(distance_4)

        # visualization
        ball1.xs = xs_1
        ball1.ys = ys_1
        ball1.nxs = nxs_1
        ball1.nys = nys_1

        ball2.xs = xs_2
        ball2.ys = ys_2
        ball2.nxs = nxs_2
        ball2.nys = nys_2

        ball3.xs = xs_3
        ball3.ys = ys_3
        ball3.nxs = nxs_3
        ball3.nys = nys_3

        ball4.xs = xs_4
        ball4.ys = ys_4
        ball4.nxs = nxs_4
        ball4.nys = nys_4

        return error_1 + error_2 + error_3 + error_4


    opt.setObjectiveFunction(objectiveFunction)
    opt.callObjectiveFunction()
    opt.printParameters()

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for ball in ['ball1', 'ball2', 'ball3', 'ball4']:
        for idx, angle in enumerate(angles):
            params = opt.getParamsContainingPattern(ball + '_')
            # print(params,'\n')
            opt.pushResidual(name=ball + '_r' + str(idx), params=params)

    print('residuals = ' + str(opt.residuals))
    opt.computeSparseMatrix()
    opt.printSparseMatrix()
    # exit(0)
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
        opt.printParameters()
        # return None
        ball1 = models['ball1']
        ball2 = models['ball2']
        ball3 = models['ball3']
        ball4 = models['ball4']
        edges = models['edges']

        gui_image = copy.deepcopy(image)
        # # draw line to closest white point
        # for x1, y1, nx1, ny1, x2, y2, nx2, ny2, x3, y3, nx3, ny3, x4, y4, nx4, ny4 in zip(ball1.xs, ball1.ys, ball1.nxs,
        #                                                                                   ball1.nys, ball2.xs, ball2.ys,
        #                                                                                   ball2.nxs, ball2.nys,
        #                                                                                   ball3.xs, ball3.ys, ball3.nxs,
        #                                                                                   ball3.nys, ball4.xs, ball4.ys,
        #                                                                                   ball4.nxs, ball4.nys):
        #     x1, y1, nx1, ny1 = int(x1), int(y1), int(nx1), int(ny1)
        #     cv2.line(gui_image, (x1, y1), (nx1, ny1), (0, 0, 255), 1)
        #
        #     x2, y2, nx2, ny2 = int(x2), int(y2), int(nx2), int(ny2)
        #     cv2.line(gui_image, (x2, y2), (nx2, ny2), (0, 0, 255), 1)
        #
        #     x3, y3, nx3, ny3 = int(x3), int(y3), int(nx3), int(ny3)
        #     cv2.line(gui_image, (x3, y3), (nx3, ny3), (0, 0, 255), 1)
        #
        #     x4, y4, nx4, ny4 = int(x4), int(y4), int(nx4), int(ny4)
        #     cv2.line(gui_image, (x4, y4), (nx4, ny4), (0, 0, 255), 1)

        ball_instances = [ball1, ball2, ball3, ball4]
        colors=[(0,  0,255),(219,  83,236),(166,  140,0),(102,0,255)]
        # draw ball perimeter
        for idx, ball_names in enumerate(['ball1', 'ball2', 'ball3', 'ball4']):
            # for x1, y1, x2, y2, x3, y3, x4, y4 in zip(ball1.xs, ball1.ys, ball2.xs, ball2.ys, ball3.xs, ball3.ys, ball4.xs,
            #                                       ball4.ys):
            # print(ball_instances[idx].xs)
            for x, y in zip(ball_instances[idx].xs, ball_instances[idx].ys):
                x, y = int(x), int(y)
                cv2.line(gui_image, (x, y), (x, y), colors[idx], 2)
                # print(x,y)
            # x1, y1 = int(x1), int(y1)
            # cv2.line(gui_image, (x1, y1), (x1, y1), (255, 0, 0), 2)
            #
            # x2, y2 = int(x2), int(y2)
            # cv2.line(gui_image, (x2, y2), (x2, y2), (236, 219, 83), 2)
            #
            # x3, y3 = int(x3), int(y3)
            # cv2.line(gui_image, (x3, y3), (x3, y3), (0, 166, 140), 2)
            #
            # x4, y4 = int(x4), int(y4)
            # cv2.line(gui_image, (x4, y4), (x4, y4), (0, 166, 140), 2)

        h, w, nc = gui_image.shape
        for pix_x in range(0, w):
            for pix_y in range(0, h):
                if edges[pix_y, pix_x] == 255:
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
        optimization_options={'x_scale': 'jac', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-4})

    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0.01, verbose=False) == 'x':
        print("exiting")
        exit(0)
