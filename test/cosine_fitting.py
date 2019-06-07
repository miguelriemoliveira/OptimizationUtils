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

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import KeyPressManager.KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities


class Polynomial:

    def __init__(self):
        self.param0 = 0
        self.param1 = 0
        self.param2 = 0
        self.params_3_and_4 = [0, 0]


if __name__ == "__main__":

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
    polynomial = Polynomial()
    x = (np.linspace(-1 * np.pi / 2, np.pi / 2, 100)).tolist()

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('polynomial', polynomial)


    # Create specialized getter and setter functions
    def setter(polynomial, value, i):
        if i == 0:
            polynomial.param0 = value
        elif i == 1:
            polynomial.param1 = value
        elif i == 2:
            polynomial.param2 = value
        elif i == 3:
            polynomial.params_3_and_4[0] = value
        elif i == 4:
            polynomial.params_3_and_4[1] = value


    def getter(polynomial, i):
        if i == 0:
            return [polynomial.param0]
        elif i == 1:
            return [polynomial.param1]
        elif i == 2:
            return [polynomial.param2]
        elif i == 3:
            return [polynomial.params_3_and_4[0]]
        elif i == 4:
            return [polynomial.params_3_and_4[1]]


    for idx in range(0, 5):
        opt.pushParamScalar(group_name='p' + str(idx), data_key='polynomial', getter=partial(getter, i=idx),
                            setter=partial(setter, i=idx))


    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        polynomial = model['polynomial']

        def pol(u):
            y = polynomial.param0[0] + (polynomial.param1[0] * u) + (polynomial.param2[0] * u ** 2) + (
                        polynomial.params_3_and_4[0][0] * u ** 3) + (polynomial.params_3_and_4[1][0] * u ** 4)
            return y

        error = []
        for a in x:
            error.append(abs(pol(a) - np.cos(a)))

        return error


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for a in range(0, len(x)):
        opt.pushResidual(name='x' + str(a), params=['p0', 'p1', 'p2', 'p3', 'p4'])

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()

    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig = plt.figure()
    ax = fig.gca()

    ax.set_xlabel('X'), ax.set_ylabel('Y'),
    ax.set_xticklabels([]), ax.set_yticklabels([])
    ax.set_xlim(-math.pi/2, math.pi/2), ax.set_ylim(-5, 5)

    # Draw cosine fucntion
    f = np.cos(x)
    ax.plot(x, f, label="cosine")
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

    y = 0 + \
        np.multiply(0, np.power(x, 1)) + \
        np.multiply(0, np.power(x, 2)) + \
        np.multiply(0, np.power(x, 3)) + \
        np.multiply(0, np.power(x, 4))

    handle_plot = ax.plot(x, y, label="polynomial")
    print(type(handle_plot))
    print((handle_plot))

    wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0., verbose=False):
        exit(0)

    # handles_out = {}
    # handles_out['point'] = ax.plot([pt_origin[0, 0], pt_origin[0, 0]], [pt_origin[1, 0], pt_origin[1, 0]],
    #                                [pt_origin[2, 0], pt_origin[2, 0]], 'k.')[0]
    # handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black',
    #                               fontsize=fontsize)
    # else:
    #     handles['point'].set_xdata([pt_origin[0, 0], pt_origin[0, 0]])
    #     handles['point'].set_ydata([pt_origin[1, 0], pt_origin[1, 0]])
    #     handles['point'].set_3d_properties(zs=[pt_origin[2, 0], pt_origin[2, 0]])
    #
    #     handles['text'].set_position((pt_origin[0, 0], pt_origin[1, 0]))
    #     handles['text'].set_3d_properties(z=pt_origin[2, 0], zdir='x')

    def visualizationFunction(model):

        polynomial = model['polynomial']

        y = polynomial.param0[0] + \
            np.multiply(polynomial.param1[0], np.power(x, 1)) + \
            np.multiply(polynomial.param2[0], np.power(x, 2)) + \
            np.multiply(polynomial.params_3_and_4[0][0], np.power(x, 3)) + \
            np.multiply(polynomial.params_3_and_4[1][0], np.power(x, 4))

        handle_plot[0].set_ydata(y)


        wm = KeyPressManager.KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
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
        optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-12, 'gtol': 1e-8, 'diff_step': 1e-4})

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)
