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
from scipy.stats import norm # to generate gaussian functions
import random

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

class Gmm:

    # Each parameter is a vector with each Gaussian:
    # - mixing coefficient
    # - means
    # - standard deviation
    def init(self, gt_means, gt_stds):
        n = len(means)
        self.param0 = [float(1/float(n))] * n
        self.param1 = [x + y for x, y in zip(means, [random.random() / 3] * n)]
        self.param2 = [x + y for x, y in zip(stds, [random.random() / 3] * n)]



if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments

    ap.add_argument('--pis',  nargs='+', type=float)
    ap.add_argument('--means',  nargs='+', type=float)
    ap.add_argument('--stds',  nargs='+', type=float)

    args = ap.parse_args()

    # Save input arguments
    pis   = args.pis 
    means = args.means 
    stds  = args.stds 
    # Check input arguments
    if len(pis) <> len(means) or len(pis) <> len(stds) or len(means) <> len(stds):
        print('Invalid input arguments\n')
        exit(0)
    
    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    gmm = Gmm()
    gmm.init(means, stds)
    xs = (np.linspace(-2, 2, 100)).tolist()

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('gmm', gmm)


    # Create specialized getter and setter functions
    def setter(gmm, value, i):
        gmm.param0[i] = value[0]        
        gmm.param1[i] = value[1]        
        gmm.param2[i] = value[2]        

    def getter(gmm, i):
        return [gmm.param0[i], gmm.param1[i], gmm.param2[i]]


    for idx in range(0, len(means)):
        opt.pushParamVector(group_name='g' + str(idx) + '_', data_key='gmm',
                            getter=partial(getter, i=idx),
                            setter=partial(setter, i=idx),
                            bound_min=[0, -np.inf, 1e-200],
                            suffix=['pis', 'means', 'stds'])

    opt.printParameters()

    # ---------------------------------------
    # --- Define THE GROUND TRUTH GMM FUNCTION
    # ---------------------------------------
    def gaussian_mm(pis, means, stds, xs):
        ys = []
        for x in xs:
            y = 0
            for idx in range(0, len(means)):
                y += pis[idx] * norm.pdf(x, means[idx], stds[idx])
            ys.append(y)
        return ys

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        gmm = model['gmm']
        errors = []
        gt = gaussian_mm(pis, means, stds, xs)
        ys = gaussian_mm(gmm.param0, gmm.param1, gmm.param2, xs)
        errors = np.array(gt) - np.array(ys)
        errors = (errors * errors).tolist()

        return errors

    opt.setObjectiveFunction(objectiveFunction)
    opt.callObjectiveFunction()

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for x in range(0, len(xs)):
        opt.pushResidual(name='x' + str(x), params=['g0_pis', 'g0_means', 'g0_stds',
        'g1_pis', 'g1_means', 'g1_stds', 'g2_pis', 'g2_means', 'g2_stds'])

    opt.computeSparseMatrix()

    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------

    fig1 = plt.figure(1)

    ax1 = plt.subplot(211)
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'),
    ax1.set_xticklabels([]), ax1.set_yticklabels([])
    ax1.set_xlim(-2, 2), ax1.set_ylim(0, 5)

    # Draw gound truth gmm function
    f = gaussian_mm(pis, means, stds, xs)

    ax1.plot(xs, f, label="Ground Truth GMM")
    legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')

    # Draw estimated gmm
    y = gaussian_mm(gmm.param0, gmm.param1, gmm.param2, xs)
    handle_plot = ax1.plot(xs, y, label="Estimated GMM")
    legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')

    # Draw individual gaussians
    ax2 = plt.subplot(212)
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'),
    ax2.set_xticklabels([]), ax2.set_yticklabels([])
    ax2.set_xlim(-2, 2), ax2.set_ylim(0, 5)
    gauss_plot = [None] * len(gmm.param0)
    print(len(gauss_plot))
    for x in range(0, len(gmm.param0)):
        gt_gauss = pis[x] * norm.pdf(xs, means[x], stds[x])
        es_gauss = gmm.param0[x] * norm.pdf(xs, gmm.param1[x], gmm.param2[x])
        ax2.plot(xs, gt_gauss, label="Ground Truth Gaussian " + str(x))
        gauss_plot[x] = ax2.plot(xs, es_gauss, label="Estimated Gaussian " + str(x))
        legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-small')

    wm = KeyPressManager.KeyPressManager.WindowManager(fig1)
    if wm.waitForKey(0., verbose=False):
        exit(0)

    def visualizationFunction(model):
        gmm = model['gmm']
        y = gaussian_mm(gmm.param0, gmm.param1, gmm.param2, xs)
        handle_plot[0].set_ydata(y)

        for x in range(0, len(gmm.param0)):
            y = gmm.param0[x] * norm.pdf(xs, gmm.param1[x], gmm.param2[x])
            gauss_plot[x][0].set_ydata(y)

        wm = KeyPressManager.KeyPressManager.WindowManager(fig1)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)

    opt.setVisualizationFunction(visualizationFunction, True)

    # ---------------------------------------
    # --- Call the Optimizer
    # ---------------------------------------

    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-8, 'xtol': 1e-8, 'gtol':1e-8,
    'diff_step': None})

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)
