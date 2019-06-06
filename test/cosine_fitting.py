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
from itertools import combinations
import numpy as np
import cv2
from functools import partial
import random
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import KeyPressManager.KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities

if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap = OCDatasetLoader.addArguments(ap) # Dataset loader arguments
    ap = OptimizationUtils.addArguments(ap) # OptimizationUtils arguments
    args = vars(ap.parse_args())
    print(args)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    dataset_loader = OCDatasetLoader.Loader(args)
    dataset = dataset_loader.loadDataset()
    num_cameras = len(dataset.cameras)
    print(num_cameras)

    # Change camera's colors just to better see optimization working
    for i, camera in enumerate(dataset.cameras):
        # if i>0:
        dataset.cameras[i].rgb.image = utilities.addSafe(dataset.cameras[i].rgb.image, random.randint(-170, 170))

    # lets add a bias variable to each camera.rgb. This value will be used to change the image and optimize
    for i, camera in enumerate(dataset.cameras):
        # camera.rgb.bias = random.randint(-30, 30)
        camera.rgb.bias = 0

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('dataset', dataset)
    opt.addModelData('another_thing', [])

    def setter(dataset, value, i):
        dataset.cameras[i].rgb.bias = value

    def getter(dataset, i):
        return [dataset.cameras[i].rgb.bias]


    # Create specialized getter and setter functions
    for idx_camera, camera in enumerate(dataset.cameras):
        if idx_camera == 0:# First camera with static color
            bound_max = camera.rgb.bias + 0.00001
            bound_min = camera.rgb.bias - 0.00001
        else:
            bound_max = camera.rgb.bias + 250
            bound_min = camera.rgb.bias - 250

        opt.pushParamScalar(group_name='bias_' + camera.name, data_key='dataset', getter=partial(getter, i=idx_camera),
                            setter=partial(setter, i=idx_camera), bound_max=bound_max, bound_min=bound_min)

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        # Get the dataset from the model dictionary
        # dataset = model['dataset']


        error = []

        # compute the errors ...


        return error


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    # The error is computed from the difference of the average color of one image with another
    # Thus, we will use all pairwise combinations of available images
    # For example, if we have 3 cameras c0, c1 and c2, the residuals should be:
    #    c0-c1, c0-c2, c1-c2

    for cam_a, cam_b in combinations(dataset.cameras, 2):
        opt.pushResidual(name='c' + cam_a.name + '-c' + cam_b.name, params=['bias_' + cam_a.name, 'bias_' + cam_b.name])

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()


    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------
    def visualizationFunction(model):
        # Get the dataset from the model dictionary
        dataset = model['dataset']

        for i, camera in enumerate(dataset.cameras):
            cv2.namedWindow('Initial Cam ' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('Initial Cam ' + str(i), camera.rgb.image)

        for i, camera in enumerate(dataset.cameras):
            cv2.namedWindow('Changed Cam ' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('Changed Cam ' + str(i), camera.rgb.image_changed)
        cv2.waitKey(20)


    opt.setVisualizationFunction(visualizationFunction, True)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # opt.fromXToData()
    # opt.callObjectiveFunction()
    # wm = KeyPressManager.KeyPressManager.WindowManager()
    # if wm.waitForKey():
    #     exit(0)

    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization(optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-8, 'gtol': 1e-8, 'diff_step': 1e-0})

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)
