#!/usr/bin/env python3

import argparse
import random
import sys
import colorama
import matplotlib.pyplot as plt
import pickle
import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import OptimizationUtils.KeyPressManager as KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils


class nDegreePolynomialModel():
    def __init__(self, list_parameters):
        dict_parameters = {}
        for idx, parameter in enumerate(list_parameters):
            dict_parameters['a{}'.format(idx)] = parameter

        for parameter_key, parameter_value in dict_parameters.items():
            setattr(self, parameter_key, parameter_value)

        self.degree = len(list_parameters) - 1
        self.parameters_names = list(dict_parameters.keys())

    def getY(self, x):

        final_parameters = []
        for att in self.parameters_names:
            if type(getattr(self, att)) is list:
                final_parameters.append(getattr(self, att)[0])
            else:
                final_parameters.append(getattr(self, att))

        y = 0
     
        for idx, final_parameter in enumerate(final_parameters):
            y += final_parameter * x ** (self.degree - idx)

        return y


    def getYs(self, xs, mi = False):
        
        ys_old = [self.getY(item) for item in xs]
        if mi:
            
            ys = []
            for idx, y_old in enumerate(ys_old):
                if idx == 0:
                    ys.append(y_old)
                elif y_old < ys[-1]:
                    ys.append(ys[-1])
                else:
                    ys.append(y_old)
            
            return ys
        else:
            return ys_old

        


def main():
    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------
    # -----------------------------------------------------
    # Command line arguments
    parser = argparse.ArgumentParser(description='Regression example 1')
    parser.add_argument('-inp', '--input_numpy', type=str, required=True,
                        help='Filename to read the data points JIH observations (numpy).')
    parser.add_argument('-mi', '--monotonically_increasing', action='store_true', default=False,
                        help='Force function to be monotonically increasing')
    parser.add_argument('-deg', '--degree', type=int, required=True, help = 'Degree of the polynomial regression')
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

    # Create n Degree Polynomial model
    n_degree_polynomial_model = nDegreePolynomialModel([1] * (args['degree'] + 1))
    # n_degree_polynomial_model = nDegreePolynomialModel([-0.000027, 0.013832, -1.829113, 128.440552])
    initial_n_degree_polynomial_model = nDegreePolynomialModel([1] * (args['degree'] + 1))
    # initial_n_degree_polynomial_model = nDegreePolynomialModel([-0.000027, 0.013832, -1.829113, 128.440552])
    xs = np.linspace(0, 255, 256)
    print(len(xs))


    print(n_degree_polynomial_model.parameters_names)


    # exit(0)
    # initialize optimizer
    opt = OptimizationUtils.Optimizer()

    # Add data models
    opt.addDataModel('n_degree_polynomial_model', n_degree_polynomial_model)

    # -----------------------------------------------------
    # Define parameters
    # -----------------------------------------------------

    # Parabola parameters
    def getternDegreePolynomial(data, prop):
        # data is our class instance

        for parameters_name in n_degree_polynomial_model.parameters_names:
            if prop == parameters_name:
                return [getattr(data, parameters_name)]

    def setternDegreePolynomial(data, value, prop):

        for parameters_name in n_degree_polynomial_model.parameters_names:
            if prop == parameters_name:
                setattr(data, parameters_name, value)

    for parameters_name in n_degree_polynomial_model.parameters_names:
        opt.pushParamVector(group_name='{}_polynomial_{}'.format(n_degree_polynomial_model.degree, parameters_name),
                            data_key='n_degree_polynomial_model',
                            getter=partial(getternDegreePolynomial, prop=parameters_name),
                            setter=partial(setternDegreePolynomial, prop=parameters_name),
                            suffix=[''])

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
        ys_obs_from_model = n_degree_polynomial_model.getYs(xs_obs, args['monotonically_increasing'])

        # Compute error
        # errors from the parabola --> (ys_obs, ys_obs_from_model)
        for idx, (y_o, y_ofm) in enumerate(zip(ys_obs, ys_obs_from_model)):
            residual = (y_o - y_ofm) ** 2
            # residual = abs(y_o - y_ofm)
            residuals['{}_polynomial_r'.format(n_degree_polynomial_model.degree) + str(idx)] = residual

        return residuals

    opt.setObjectiveFunction(objectiveFunction)

    # -----------------------------------------------------
    # Define residuals
    # -----------------------------------------------------
    params = ['{}_polynomial_'.format(n_degree_polynomial_model.degree) + parameters_name for parameters_name in
              n_degree_polynomial_model.parameters_names]

    for idx, x in enumerate(xs_obs):
        opt.pushResidual(name='{}_polynomial_r'.format(n_degree_polynomial_model.degree) + str(idx), params=params)

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

    ax.plot(xs_obs, ys_obs, 'bo')

    # draw parabola model
    xs = list(np.linspace(0, 255, 256))
    print(xs)
    ys = n_degree_polynomial_model.getYs(xs)
    handle_model_plot = ax.plot(xs, ys, '-g')

    # draw best parabola model
    # TODO Professor used xs_obs
    handle_best_model_plot = ax.plot(xs, initial_n_degree_polynomial_model.getYs(xs,args['monotonically_increasing']), '--k')

    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0., verbose=False):
        exit(0)

    # -----------------------------------------------------
    # Define visualization function
    # -----------------------------------------------------
    def visualizationFunction(data_models):
        # retrieve data models
        n_degree_polynomial_model = data_models['n_degree_polynomial_model']
        # print('Visualization function called ...')
        # print('a=' + str(parabola_model.a))
        # print('b=' + str(parabola_model.b))
        # print('c=' + str(parabola_model.c))
        # opt.printParameters()

        # n_degree_polynomial_model visualization
        handle_model_plot[0].set_ydata(n_degree_polynomial_model.getYs(xs,args['monotonically_increasing']))

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)
        # plt.draw()
        # plt.waitforbuttonpress(1)

    opt.setVisualizationFunction(visualizationFunction, True)

    # -----------------------------------------------------
    # Start optimization
    # -----------------------------------------------------
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-3, 'xtol': 1e-3, 'gtol': 1e-3, 'diff_step': None})
    opt.printParameters()

    # -----------------------------------------------------
    # Create cmf and save figure
    # -----------------------------------------------------
    # Given a target color (x value), the function returns source color (y value).
    xs_cmf = list(np.linspace(0, 255, 256).astype(int))
    ys_cmf = n_degree_polynomial_model.getYs(xs_cmf,args['monotonically_increasing'])
    ys_cmf = [int(max(0, min(round(y_cmf), 255))) for y_cmf in ys_cmf]  # round, undersaturate and oversaturate
    polynomial_cmf = {'cmf': {'x': xs_cmf, 'y': ys_cmf}}

    filename = '{}_polynomial'.format(n_degree_polynomial_model.degree)
    fig.savefig(filename)

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)


if __name__ == '__main__':
    main()
