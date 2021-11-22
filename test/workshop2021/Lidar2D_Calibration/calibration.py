#!/usr/bin/python3

import OptimizationUtils.OptimizationUtils as OptimizationUtils
from json_reader import *
import matplotlib.pyplot as plt
import math
import numpy as np
from functools import partial
import sys


class LaserModel():
    def __init__(self, tx, ty, ang):
        self.tx = tx
        self.ty = ty
        self.ang = ang

    def getCoord(self, x, y):
        if type(self.tx) is list:
            tx = self.tx[0]
        else:
            tx = self.tx

        if type(self.ty) is list:
            ty = self.ty[0]
        else:
            ty = self.ty

        if type(self.ang) is list:
            ang = self.ang[0]
        else:
            ang = self.ang

        matrix = np.array([[math.cos(ang), -math.sin(ang), tx],
                           [math.sin(ang), math.cos(ang), ty],
                           [0, 0, 1]])

        in_matrix = np.array([[x], [y], [1]])

        out_matrix = np.matmul(matrix, in_matrix)

        x_out = out_matrix[0]
        y_out = out_matrix[1]

        return x_out[0], y_out[0]

    def getCoords(self, xs, ys):
        x_ol = []
        y_ol = []
        for idx, y in enumerate(ys):
            x = xs[idx]
            (x_o, y_o) = self.getCoord(x, y)
            x_ol.append(x_o)
            y_ol.append(y_o)
        return x_ol, y_ol


def main():
    # Initiating module
    laser_model = LaserModel(0, 0, 0)
    best_laser_model = LaserModel(0, 0, 0)

    # Calling json_reader functions
    data_left, data_right = jsonImporter('data/data_collected.json')
    left_xs, left_ys, right_xs, right_ys = dataViewer(data_left, data_right)

    # Initializing and viewing the plot
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(0, 0)
    ax.grid()
    ax.axis([-20, 20, -20, 20])
    handle_left_laser = ax.plot(left_xs, left_ys, 'bo')
    handle_right_laser = ax.plot(right_xs, right_ys, 'ro')
    plt.draw()
    plt.waitforbuttonpress(1)

    opt = OptimizationUtils.Optimizer()

    # Add data models
    opt.addDataModel('laser_model', laser_model)

    # -----------------------------------------------------
    # Define parameters
    # -----------------------------------------------------

    # Laser parameters
    def getterLaser(data, prop):
        if prop == 'tx':
            return [data.tx]
        elif prop == 'ty':
            return [data.ty]
        elif prop == 'ang':
            return [data.ang]

    def setterLaser(data, value, prop):
        if prop == 'tx':
            data.tx = value
        elif prop == 'ty':
            data.ty = value
        elif prop == 'ang':
            data.ang = value

    opt.pushParamScalar(group_name='laser_tx', data_key='laser_model',
                        getter=partial(getterLaser, prop='tx'),
                        setter=partial(setterLaser, prop='tx'))
    opt.pushParamScalar(group_name='laser_ty', data_key='laser_model',
                        getter=partial(getterLaser, prop='ty'),
                        setter=partial(setterLaser, prop='ty'))
    opt.pushParamScalar(group_name='laser_ang', data_key='laser_model',
                        getter=partial(getterLaser, prop='ang'),
                        setter=partial(setterLaser, prop='ang'))

    opt.printParameters()

    # -----------------------------------------------------
    # Define objective function
    # -----------------------------------------------------
    def objectiveFunction(data_models):
        # retrieve data models
        laser_model = data_models['laser_model']

        # Initialize the residuals
        errors = []

        # Compute observations from model
        right_xs_model, right_ys_model = laser_model.getCoords(right_xs, right_ys)

        counter = 0
        # Compute error
        # errors from the laser model
        # for x, x_m, y, y_m in zip(left_xs, right_xs_model, left_ys, right_ys_model):
        for idx, x in enumerate(left_xs):
            y = left_ys[idx]
            error_min = sys.float_info.max
            for idx2, x_m in enumerate(right_xs_model):
                y_m = right_ys_model[idx2]
                counter += 1
                error = abs(x - x_m) + abs(y - y_m)
                if error < error_min:
                    error_min = error
            errors.append(error_min)
        return errors

    opt.setObjectiveFunction(objectiveFunction)

    # -----------------------------------------------------
    # Define residuals
    # -----------------------------------------------------

    for idx, x in enumerate(right_xs):
        opt.pushResidual(name='laser_r' + str(idx), params=['laser_tx', 'laser_ty', 'laser_ang'])

    opt.printResiduals()

    # -----------------------------------------------------
    # Compute sparse matrix
    # -----------------------------------------------------
    opt.computeSparseMatrix()
    opt.printSparseMatrix()

    # -----------------------------------------------------
    # Define visualization function
    # -----------------------------------------------------
    def visualizationFunction(data_models):
        # retrieve data models
        laser_model = data_models['laser_model']

        print('Visualization function called ...')
        print('tx=' + str(laser_model.tx))
        print('ty=' + str(laser_model.ty))
        print('ang=' + str(laser_model.ang))

        right_xs_model, right_ys_model = laser_model.getCoords(right_xs, right_ys)

        # laser visualization
        handle_right_laser[0].set_xdata(right_xs_model)
        handle_right_laser[0].set_ydata(right_ys_model)

        wm = OptimizationUtils.KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)

    opt.setVisualizationFunction(visualizationFunction, True)

    # -----------------------------------------------------
    # Start optimization
    # -----------------------------------------------------
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6, 'diff_step': None})


if __name__ == "__main__":
    main()
