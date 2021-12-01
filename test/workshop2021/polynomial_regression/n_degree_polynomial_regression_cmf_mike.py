#!/usr/bin/env python3

import argparse
import math
import random
import sys
import colorama
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import OptimizationUtils.KeyPressManager as KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils


class nDegreePolynomialModel():
    def __init__(self, initial_parameters):
        self.parameters = initial_parameters  # a list [a,b,c, ... z] for a polynomial such as a x + b x**2 + ... z x**n
        self.degree = len(initial_parameters) - 1

    def getY(self, x):
        y = 0
        for idx, parameter in enumerate(self.parameters):
            y += parameter * math.pow(x, idx)
        return y

    def getYs(self, xs):
        return [self.getY(item) for item in xs]


def main():
    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------
    # Command line arguments
    parser = argparse.ArgumentParser(description='Regression example 1')
    parser.add_argument('-inp', '--input_numpy', type=str, required=True,
                        help='Filename to read the data points JIH observations (numpy).')
    parser.add_argument('-mi', '--monotonically_increasing', action='store_true', default=False,
                        help='Force function to be monotonically increasing')
    parser.add_argument('-deg', '--degree', type=int, required=True, help='Degree of the polynomial regression')
    args = vars(parser.parse_args())
    print(args)

    # -----------------------------------------------------
    # Load file with data points
    print('Loading file to ' + str(args['input_numpy']))
    with open(args['input_numpy'], 'rb') as file_handle:
        jih = np.load(file_handle)

    jih_b = jih[0, :, :]

    # Convert histogram into observations
    tgt_colors = []
    src_colors = []
    for tgt_color in range(jih_b.shape[1]):
        for src_color in range(jih_b.shape[0]):
            if jih_b[src_color, tgt_color] > 0:
                src_colors.extend(
                    [src_color] * jih_b[src_color, tgt_color])  # create observations from histogram
                tgt_colors.extend(
                    [tgt_color] * jih_b[src_color, tgt_color])  # create observations from histogram

    xs_obs = tgt_colors
    ys_obs = src_colors

    # subsample xs_obs and ys_obs
    subsample = 1
    xs_obs = xs_obs[::subsample]
    ys_obs = ys_obs[::subsample]

    # Create n Degree Polynomial model
    n_degree_polynomial_model = nDegreePolynomialModel([100] + [0] * args['degree'] )

    # initialize optimizer
    opt = OptimizationUtils.Optimizer()
    # Add data models
    opt.addDataModel('n_degree_polynomial_model', n_degree_polynomial_model)

    # -----------------------------------------------------
    # Define parameters
    # -----------------------------------------------------
    def getterParameters(data):
        return data.parameters

    def setterParameters(data, values):
        data.parameters = values

    opt.pushParamVector(group_name='polynomial_', data_key='n_degree_polynomial_model',
                        getter=getterParameters,
                        setter=setterParameters,
                        suffix=[str(item) for item in range(0, n_degree_polynomial_model.degree + 1)])

    opt.printParameters()

    # -----------------------------------------------------
    # Define objective function
    # -----------------------------------------------------
    def objectiveFunction(data_models):
        # retrieve data models
        n_degree_polynomial_model = data_models['n_degree_polynomial_model']

        # Initialize the residuals
        residuals = {}

        # Compute observations from model
        ys_obs_from_model = n_degree_polynomial_model.getYs(xs_obs)

        # Compute error
        for idx, (y_o, y_ofm) in enumerate(zip(ys_obs, ys_obs_from_model)):
            residual = (y_o - y_ofm) ** 2  # TODO try without the modulus
            # residual = y_o - y_ofm
            residual_name = 'r' + str(idx)
            residuals[residual_name] = residual

        return residuals

    opt.setObjectiveFunction(objectiveFunction)

    # -----------------------------------------------------
    # Define residuals
    # -----------------------------------------------------
    params = opt.getParameters() # use all parameters. The matrix is completely dense.
    for idx, _ in enumerate(ys_obs):
        residual_name = 'r' + str(idx)
        opt.pushResidual(name=residual_name, params=params)

    opt.printResiduals()

    # -----------------------------------------------------
    # Compute sparse matrix
    # -----------------------------------------------------
    opt.computeSparseMatrix()
    opt.printSparseMatrix()

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------
    # create a figure
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(0, 0)
    ax.grid()
    ax.axis([0, 255, 0, 255])

    # draw observations
    # jih_b = jih[0, :, :]

    colormap3 = matplotlib.cm.Blues(np.linspace(0, 1, np.amax(jih_b, axis=None) + 1))

    xys_jih_b = np.argwhere(jih_b > 0)  # get coordinates of counts of jih > 0
    xs_jih_b = xys_jih_b[:, 1]  # extract x coordinate
    ys_jih_b = xys_jih_b[:, 0]  # extract y coordinate
    counts = jih_b[ys_jih_b, xs_jih_b]  # get counts at the extracted coordinates
    colors = colormap3[counts, :-1]  # get colors as a function of the counts
    plt.scatter(xs_jih_b, ys_jih_b, c=colors)  # draw the histogram
    # plt.xlim([0, jih.shape[1] - 1])
    # plt.ylim([0, jih.shape[0] - 1])

    # ax.plot(xs_obs, ys_obs, 'bo') # before ...
    # ---------------------------

    # draw parabola model
    xs = list(np.linspace(0, 255, 256))
    ys = n_degree_polynomial_model.getYs(xs)
    handle_model_plot = ax.plot(xs, ys, '-g')

    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0.1, verbose=False):
        exit(0)

    # -----------------------------------------------------
    # Define visualization function
    # -----------------------------------------------------
    def visualizationFunction(data_models):
        # retrieve data models
        n_degree_polynomial_model = data_models['n_degree_polynomial_model']

        # n_degree_polynomial_model visualization
        handle_model_plot[0].set_ydata(n_degree_polynomial_model.getYs(xs))

        # opt.printParameters()
        # opt.printResiduals()
        if wm.waitForKey(0.01, verbose=False):
            exit(0)
        # plt.draw()
        # plt.waitforbuttonpress(1)

    opt.setVisualizationFunction(visualizationFunction, True)

    # -----------------------------------------------------
    # Start optimization
    # -----------------------------------------------------
    # optimization_options = {'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6, 'diff_step': 1e-15}
    x_scale = [1/math.pow(100, item) for item in range(0,n_degree_polynomial_model.degree+1)]
    print(opt.x0)
    print(x_scale)
    optimization_options = {'x_scale': x_scale, 'ftol': 1e-16, 'xtol': 1e-25, 'gtol': 1e-6, 'diff_step': 1e-15}
    opt.startOptimization(optimization_options=optimization_options)
    opt.printParameters()

    # -----------------------------------------------------
    # Create cmf and save figure
    # -----------------------------------------------------
    # Given a target color (x value), the function returns source color (y value).
    xs_cmf = list(np.linspace(0, 255, 256).astype(int))
    ys_cmf = n_degree_polynomial_model.getYs(xs_cmf, args['monotonically_increasing'])
    ys_cmf = [int(max(0, min(round(y_cmf), 255))) for y_cmf in ys_cmf]  # round, undersaturate and oversaturate
    polynomial_cmf = {'cmf': {'x': xs_cmf, 'y': ys_cmf}}

    filename = '{}_polynomial'.format(n_degree_polynomial_model.degree)
    fig.savefig(filename)

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)


if __name__ == '__main__':
    main()
