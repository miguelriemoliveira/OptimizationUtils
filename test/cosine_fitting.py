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
import numpy as np
import pylab as pl
from functools import partial
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
        self.param1 = 0
        self.param2 = 0
        self.params_3_and_4 = [0, 0]



if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    # It will be the number of polynomial degree
    # for now, i will work with 4

    # ap = argparse.ArgumentParser()
    # ap = OCDatasetLoader.addArguments(ap) # Dataset loader arguments
    # ap = OptimizationUtils.addArguments(ap) # OptimizationUtils arguments
    # args = vars(ap.parse_args())
    # print(args)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    polynomial = Polynomial()

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('polynomial', polynomial)

    def setter(polynomial, value, i):
        if i == 1:
            polynomial.param1 = value
        elif i == 2:
            polynomial.param2 = value
        elif i == 3:
            polynomial.params_3_and_4[0] = value
        elif i == 4:
            polynomial.params_3_and_4[1] = value


    def getter(polynomial, i):
        if i == 1:
            return [polynomial.param1]
        elif i == 2:
            return [polynomial.param2]
        elif i == 3:
            return [polynomial.params_3_and_4[0]]
        elif i == 4:
            return [polynomial.params_3_and_4[1]]

    #
    #
    # # Create specialized getter and setter functions
    # for idx_camera, camera in enumerate(dataset.cameras):
    #     if idx_camera == 0:# First camera with static color
    #         bound_max = camera.rgb.bias + 0.00001
    #         bound_min = camera.rgb.bias - 0.00001
    #     else:
    #         bound_max = camera.rgb.bias + 250
    #         bound_min = camera.rgb.bias - 250
    #
    for idx in range(1, 5):
        opt.pushParamScalar(group_name='p' + str(idx), data_key='polynomial', getter=partial(getter, i=idx),
                            setter=partial(setter, i=idx))

    # # ---------------------------------------
    # # --- Define THE OBJECTIVE FUNCTION
    # # ---------------------------------------
    def objectiveFunction(model):

        polynomial = model['polynomial']

        x = np.arange(-1*np.pi/2, np.pi/2, np.pi/30)
        def pol(u):
            y = (polynomial.param1 * u) + (polynomial.param2 * u**2) + (polynomial.params_3_and_4[0] * u**3) + (polynomial.params_3_and_4[1] * u**4)
            return y

        error = []
        for a in x:
            error.append(abs(pol(a) - np.cos(a)))

        return error
    opt.setObjectiveFunction(objectiveFunction)

    # # ---------------------------------------
    # # --- Define THE RESIDUALS
    # # ---------------------------------------
    for a in range(1,31):
        opt.pushResidual(name='x' + str(a), params=['p1', 'p2', 'p3', 'p4'])

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()

    #
    # # ---------------------------------------
    # # --- Define THE VISUALIZATION FUNCTION
    # # ---------------------------------------
    def visualizationFunction(polynomial):
        x = np.arange(-1 * np.pi / 2, np.pi / 2, np.pi / 30)
        y = (polynomial.param1 * x) + (polynomial.param2 * x**2) + (polynomial.params_3_and_4[0] * x**3) + (polynomial.params_3_and_4[1] * x**4)
        f = np.cos(x)

        fig = pl.plt.figure()
        ax = fig.add_subplot(111)

        pl.plt.xlabel("30 points beetween -pi/2 to pi/2")
        pl.plt.ylabel("value of polynomial and cos")

        pl.plt.plot(x, y, label="polynomial")
        pl.plt.plot(x, f, label="cosine")
        legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')


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
    opt.startOptimization(optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-8, 'gtol': 1e-8, 'diff_step': 1e-0})

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)
