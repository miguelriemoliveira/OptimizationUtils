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




class ParabolaModel():
    # TODO add a polinomyal degree as parameter
    # y = a * ((x - b) ** 2) + c
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
    parser.add_argument('-inp', '--input_numpy', type=str,required=True, help='Filename to read the data points JIH observations (numpy).')
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

    
    # Create parabola model
    parabola_model = ParabolaModel(1/100, 125, 0)
    initial_parabola_model = ParabolaModel(1/100, 125, 0)

    # initialize optimizer
    opt = OptimizationUtils.Optimizer()
    
    # Add data models
    opt.addDataModel('parabola_model', parabola_model)

    # -----------------------------------------------------
    # Define parameters
    # -----------------------------------------------------

    # Parabola parameters
    def getterParabola(data, prop):
        # data is our class instance
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

    for param in 'abc':
        opt.pushParamVector(group_name='parabola_' + param, data_key='parabola_model',
                            getter=partial(getterParabola, prop=param),
                            setter=partial(setterParabola, prop=param),
                            suffix=[''])

    opt.printParameters()

    
    # -----------------------------------------------------
    # Define objective function
    # -----------------------------------------------------
    def objectiveFunction(data_models):
        # retrieve data models
        parabola_model = data_models['parabola_model']

        # Initialize the residuals
        residuals = {}

        # Compute observations from model
        ys_obs_from_model = parabola_model.getYs(xs_obs)

        # Compute error
        # errors from the parabola --> (ys_obs, ys_obs_from_model)
        for idx, (y_o, y_ofm) in enumerate(zip(ys_obs, ys_obs_from_model)):
            residual = (y_o - y_ofm)**2
            residuals['parabola_r' + str(idx)] = residual

        
        return residuals
    
    opt.setObjectiveFunction(objectiveFunction)

    # -----------------------------------------------------
    # Define residuals
    # -----------------------------------------------------

    for idx, x in enumerate(xs_obs):
        opt.pushResidual(name='parabola_r' + str(idx), params=['parabola_a', 'parabola_b', 'parabola_c'])

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
    xs = list(np.linspace(0, 255, 1000))
    ys = parabola_model.getYs(xs)
    handle_model_plot = ax.plot(xs, ys, '-g')

    # draw best parabola model
    # TODO Professor used xs_obs
    handle_best_model_plot = ax.plot(xs, initial_parabola_model.getYs(xs), '--k')

    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0., verbose=False):
        exit(0)

    # -----------------------------------------------------
    # Define visualization function
    # -----------------------------------------------------
    def visualizationFunction(data_models):
        # retrieve data models
        parabola_model = data_models['parabola_model']
        # print('Visualization function called ...')
        # print('a=' + str(parabola_model.a))
        # print('b=' + str(parabola_model.b))
        # print('c=' + str(parabola_model.c))


        # parabola visualization
        handle_model_plot[0].set_ydata(parabola_model.getYs(xs))

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)
        # plt.draw()
        # plt.waitforbuttonpress(1)

    opt.setVisualizationFunction(visualizationFunction,True)



    # -----------------------------------------------------
    # Start optimization
    # -----------------------------------------------------
    opt.startOptimization(optimization_options={'x_scale': 'jac', 'ftol': 1e-3,'xtol': 1e-3, 'gtol': 1e-3, 'diff_step': None})
    opt.printParameters()

    # -----------------------------------------------------
    # Create cmf
    # -----------------------------------------------------
    # Given a target color (x value), the function returns source color (y value).
    xs_cmf = list(np.linspace(0, 255, 256).astype(int))
    ys_cmf = parabola_model.getYs(xs_cmf)
    ys_cmf = [int(max(0, min(round(y_cmf), 255))) for y_cmf in ys_cmf]  # round, undersaturate and oversaturate
    polynomial_cmf = {'cmf': {'x': xs_cmf, 'y': ys_cmf}}

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)



if __name__ == '__main__':
    main()