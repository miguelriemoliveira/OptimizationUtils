#!/usr/bin/env python3

import argparse
import random
import sys

import colorama
import matplotlib.pyplot as plt
import pickle

import numpy as np
from functools import partial


# Use a line model
# y = m x + b
import OptimizationUtils.KeyPressManager


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


class ParabolaModel():
    # y = a * (x - b)** 2 + c
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def getY(self, x):
        if type(self.a) is list:
            a = self.a[0]
        else:
            a = self.a
        if type(self.b) is list:
            b = self.b[0]
        else:
            b = self.b
        if type(self.c) is list:
            c = self.c[0]
        else:
            c = self.c
        return a * ((x - b) ** 2) + c

    def getYs(self, xs):
        return [self.getY(item) for item in xs]


def main():
    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------
    # -----------------------------------------------------
    # Command line arguments
    parser = argparse.ArgumentParser(description='Regression example 1')
    parser.add_argument('-i', '--input', type=str, default='points.pkl', help='Filename to read.')
    args = vars(parser.parse_args())
    print(args)

    # -----------------------------------------------------
    # Load file with data points
    print('Loading file to ' + str(args['input']))
    with open(args['input'], 'rb') as file_handle:
        points = pickle.load(file_handle)

    # print(points)

    # Create line model
    line_model = LineModel(1, 0.5)
    best_line_model = LineModel(1, 0.5)
    total_best_error = sys.float_info.max

    # Create parabola model
    parabola_model = ParabolaModel(1, 0.5, 1)
    best_parabola_model = ParabolaModel(1, 0.5, 0)

    # initialize optimizer
    import OptimizationUtils.OptimizationUtils as OptimizationUtils
    opt = OptimizationUtils.Optimizer()

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------
    # create a figure
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(0, 0)
    ax.grid()
    ax.axis([-5, 5, -5, 5])

    # draw observations
    xs_observations_left = [item['x'] for item in points if item['x'] < 0]
    ys_observations_left = [item['y'] for item in points if item['x'] < 0]
    ax.plot(xs_observations_left, ys_observations_left, 'bo')

    xs_observations_right = [item['x'] for item in points if item['x'] >= 0]
    ys_observations_right = [item['y'] for item in points if item['x'] >= 0]
    ax.plot(xs_observations_right, ys_observations_right, 'co')

    handle_error_anchors_left = ax.plot(xs_observations_left, ys_observations_left, '.r')

    # draw line model
    xs_left = list(np.linspace(-5, 0, 100))
    ys = line_model.getYs(xs_left)
    handle_model_left_plot = ax.plot(xs_left, ys, '-.k')

    # draw parabola model
    xs_right = list(np.linspace(0, 5, 100))
    ys = parabola_model.getYs(xs_right)
    handle_model_right_plot = ax.plot(xs_right, ys, '-.k')

    # draw best line model
    handle_best_model_plot = ax.plot(xs_observations_left, best_line_model.getYs(xs_observations_left), '-m')

    # draw best line model
    handle_best_model_plot_right = ax.plot(xs_right, best_parabola_model.getYs(xs_right), '-r')

    plt.draw()
    plt.waitforbuttonpress(1)

    # Add data models
    opt.addDataModel('line_model', line_model)
    opt.addDataModel('parabola_model', parabola_model)

    # -----------------------------------------------------
    # Define parameters
    # -----------------------------------------------------

    # Line parameters
    def getterLine(data, prop):
        if prop == 'm':
            return [data.m]
        elif prop == 'b':
            return [data.b]

    def setterLine(data, value, prop):
        if prop == 'm':
            data.m = value
        elif prop == 'b':
            data.b = value

    opt.pushParamScalar(group_name='line_m', data_key='line_model',
                        getter=partial(getterLine, prop='m'),
                        setter=partial(setterLine, prop='m'))
    opt.pushParamScalar(group_name='line_b', data_key='line_model',
                        getter=partial(getterLine, prop='b'),
                        setter=partial(setterLine, prop='b'))

    # Parabola parameters
    def getterParabola(data, prop):
        if prop == 'a':
            return [data.a]
        elif prop == 'b':
            return [data.b]
        elif prop == 'c':
            return [data.c]

    def setterParabola(data, value, prop):
        if prop == 'a':
            data.a = value
        elif prop == 'b':
            data.b = value
        elif prop == 'c':
            data.c = value

    opt.pushParamScalar(group_name='parabola_a', data_key='parabola_model',
                        getter=partial(getterParabola, prop='a'),
                        setter=partial(setterParabola, prop='a'))
    opt.pushParamScalar(group_name='parabola_b', data_key='parabola_model',
                        getter=partial(getterParabola, prop='b'),
                        setter=partial(setterParabola, prop='b'))
    opt.pushParamScalar(group_name='parabola_c', data_key='parabola_model',
                        getter=partial(getterParabola, prop='c'),
                        setter=partial(setterParabola, prop='c'))

    opt.printParameters()

    # -----------------------------------------------------
    # Define objective function
    # -----------------------------------------------------
    def objectiveFunction(data_models):
        # retrieve data models
        line_model = data_models['line_model']
        parabola_model = data_models['parabola_model']

        # Initialize the residuals
        # residuals = {} # TODO lets try it with dictionaries also
        errors = []
        # errors = {}

        # Compute observations from model
        ys_observations_from_model_left = line_model.getYs(xs_observations_left)
        ys_observations_from_model_right = parabola_model.getYs(xs_observations_right)

        # Compute error
        # errors from the line model
        for y_o, y_ofm in zip(ys_observations_left, ys_observations_from_model_left):
            error = abs(y_o - y_ofm)
            errors.append(error)

        # errors from the parabola
        for y_o, y_ofm in zip(ys_observations_right, ys_observations_from_model_right):
            error = abs(y_o - y_ofm)
            errors.append(error)

        return errors

    opt.setObjectiveFunction(objectiveFunction)

    # -----------------------------------------------------
    # Define residuals
    # -----------------------------------------------------
    # params = opt.getParamsContainingPattern('weight')  # get all weight related parameters

    for idx, x in enumerate(xs_observations_left):
        opt.pushResidual(name='line_r' + str(idx), params=['line_m', 'line_b'])

    for idx, x in enumerate(xs_observations_right):
        opt.pushResidual(name='parabola_r' + str(idx), params=['parabola_a', 'parabola_b', 'parabola_c'])

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
        line_model = data_models['line_model']
        parabola_model = data_models['parabola_model']

        print('Visualization function called ...')
        print('m=' + str(line_model.m))
        print('b=' + str(line_model.b))

        # line visualization
        handle_model_left_plot[0].set_ydata(line_model.getYs(xs_left))

        # parabola visualization
        handle_model_right_plot[0].set_ydata(parabola_model.getYs(xs_right))

        wm = OptimizationUtils.KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)
        # plt.draw()
        # plt.waitforbuttonpress(1)

    opt.setVisualizationFunction(visualizationFunction, True)



    # -----------------------------------------------------
    # Start optimization
    # -----------------------------------------------------
    opt.startOptimization(optimization_options={'x_scale': 'jac', 'ftol': 1e-3,'xtol': 1e-3, 'gtol': 1e-3, 'diff_step': None})


if __name__ == '__main__':
    main()