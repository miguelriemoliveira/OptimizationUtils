#!/usr/bin/env python
"""
This example shows an optimizer working with a set of n cameras, changing their pose so that the reprojection error is
minimized.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import argparse  # to read command line arguments
import random
from copy import deepcopy
from itertools import combinations
from math import floor
import numpy as np
import numpy.linalg as npl
import gtk
import cv2
from functools import partial
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import KeyPressManager.KeyPressManager
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OCDatasetLoader.OCArucoDetector as OCArucoDetector
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.utilities as utilities

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap = OCDatasetLoader.addArguments(ap) # Dataset loader arguments
    ap = OptimizationUtils.addArguments(ap) # OptimizationUtils arguments
    args = vars(ap.parse_args())

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    dataset_loader = OCDatasetLoader.Loader(args)
    dataset_cameras = dataset_loader.loadDataset()
    print("dataset_cameras contains " + str(len(dataset_cameras.cameras)) + " cameras")

    # aruco_detector = OCArucoDetector.ArucoDetector(args)
    # dataset_arucos, dataset_cameras = aruco_detector.detect(dataset_cameras)

    # Change camera's colors just to better see optimization working
    # for i, camera in enumerate(dataset_cameras.cameras):
    #     # if i>0:
    #     dataset_cameras.cameras[i].rgb.image = utilities.addSafe(dataset_cameras.cameras[i].rgb.image,
    #                                                              random.randint(-80, 80))

    # lets add a set of six parameters per image. This value will be used to change the image and optimize
    for i, camera in enumerate(dataset_cameras.cameras):
        # camera.rgb.gamma = random.randint(0.0001, 5.0)
        camera.rgb.bias_l = 0.0  # luminance channel bias
        camera.rgb.bias_a = 0.0  # alpha channel bias
        camera.rgb.bias_b = 0.0  # beta channel bias
        camera.rgb.scale_l = 1.0  # luminance channel scale
        camera.rgb.scale_a = 1.0  # alpha channel scale
        camera.rgb.scale_b = 1.0  # beta channel scale

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('data_cameras', dataset_cameras)


    # ------------  Cameras -----------------
    def setterCameraBias(dataset, value, i):
        dataset.cameras[i].rgb.bias_l = value[0]
        dataset.cameras[i].rgb.bias_a = value[1]
        dataset.cameras[i].rgb.bias_b = value[2]


    def getterCameraBias(dataset, i):
        return [dataset.cameras[i].rgb.bias_l,
                dataset.cameras[i].rgb.bias_a,
                dataset.cameras[i].rgb.bias_b]

    def setterCameraScale(dataset, value, i):
        dataset.cameras[i].rgb.scale_l = value[0]
        dataset.cameras[i].rgb.scale_a = value[1]
        dataset.cameras[i].rgb.scale_b = value[2]


    def getterCameraScale(dataset, i):
        return [dataset.cameras[i].rgb.scale_l,
                dataset.cameras[i].rgb.scale_a,
                dataset.cameras[i].rgb.scale_b]


    # Add parameters related to the cameras
    for idx_camera, camera in enumerate(dataset_cameras.cameras):
        # if idx_camera == 0:  # First camera with static color
        #     bound_max = camera.rgb.gamma + 0.00001
        #     bound_min = camera.rgb.gamma - 0.00001
        # else:
        bound_max = 3.5
        bound_min = 0.001

        # opt.pushParamScalar(group_name='bias_C' + camera.name, data_key='data_cameras',
        #                     getter=partial(getterGamma, i=idx_camera),
        #                     setter=partial(setterGamma, i=idx_camera),
        #                     bound_max=bound_max, bound_min=bound_min)

        opt.pushParamVector3(group_name='C' + camera.name + '_bias_', data_key='data_cameras',
                             getter=partial(getterCameraBias, i=idx_camera),
                             setter=partial(setterCameraBias, i=idx_camera),
                             sufix=['lum', 'alf', 'bet'])

        opt.pushParamVector3(group_name='C' + camera.name + '_scale_', data_key='data_cameras',
                             getter=partial(getterCameraScale, i=idx_camera),
                             setter=partial(setterCameraScale, i=idx_camera),
                             sufix=['lum', 'alf', 'bet'])

    # # ------------  Arucos -----------------
    # # Each aruco will only have the position (tx,ty,tz)
    # # thus, the getter should return a list of size 3
    # def getterArucoTranslation(data, aruco_id):
    #     return data.arucos[aruco_id][0:3, 3]
    #
    #
    # def setterArucoTranslation(data, value, aruco_id):
    #     data.arucos[aruco_id][0:3, 3] = value
    #
    #
    # # Add parameters related to the arucos
    # for aruco_id, aruco in dataset_arucos.arucos.items():
    #     opt.pushParamVector3(group_name='A' + str(aruco_id), data_key='data_arucos',
    #                          getter=partial(getterArucoTranslation, aruco_id=aruco_id),
    #                          setter=partial(setterArucoTranslation, aruco_id=aruco_id),
    #                          sufix=['_tx', '_ty', '_tz'])

    opt.printParameters()

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    first_time = True


    def objectiveFunction(data):
        """
        Computes the vector of errors. Each error associated with a pairwise combination of available cameras. For each,
        pair of cameras, the average color difference between valid projections of measured 3D points.
        :return: a vector of residuals
        """
        # Get the data
        data_cameras = data['data_cameras']
        # data_arucos = data['data_arucos']
        errors = []

        # Apply changes to all camera images using parameter vector
        for camera in data_cameras.cameras:
            # camera.rgb.image_changed = utilities.adjustGamma(camera.rgb.image, camera.rgb.gamma)
            camera.rgb.image_changed = utilities.adjustLAB(camera.rgb.image,
                     l_bias=camera.rgb.bias_l, a_bias=camera.rgb.bias_a, b_bias=camera.rgb.bias_b,
                     l_scale=camera.rgb.scale_l, a_scale=camera.rgb.scale_a, b_scale=camera.rgb.scale_b)

        # Compute all the pair wise combinations of the set of cameras
        # Each element in the vector of errors is the difference of the average color for the combination
        for cam_a, cam_b in combinations(data_cameras.cameras, 2):
            # print(cam_a.name + ' with ' + cam_b.name)

            ci_a = cam_a.rgb.camera_info
            ci_b = cam_b.rgb.camera_info

            # Option1: get a list of 3D points in the map frame by concatenating the 3D measurements of cam_a and cam_b
            pts3D_in_map = np.concatenate([
                np.dot(cam_a.depth.matrix, cam_a.depth.vertices[:, 0::args['skip_vertices']]),
                np.dot(cam_b.depth.matrix, cam_b.depth.vertices[:, 0::args['skip_vertices']])], axis=1)

            # Option2: use the vertices of the mesh as the list of 3D points to project
            # pts3D_in_map = data_cameras.pts_map[:, 0::args['skip_vertices']] # use only a subset of the points

            pts2D_a, pts2D_b, valid_mask = utilities.projectToCameraPair(
                ci_a.K, ci_a.D, ci_a.width, ci_a.height, npl.inv(cam_a.rgb.matrix), cam_a.rgb.image,
                cam_a.rgb.range_dense,
                ci_b.K, ci_b.D, ci_b.width, ci_b.height, npl.inv(cam_b.rgb.matrix), cam_b.rgb.image,
                cam_b.rgb.range_dense,
                pts3D_in_map, z_inconsistency_threshold=args['z_inconsistency_threshold'],
                visualize=args['view_projected_vertices'])

            # Compute the error with the valid projections
            colors_a = cam_a.rgb.image_changed[pts2D_a[1, valid_mask], pts2D_a[0, valid_mask]]
            colors_b = cam_b.rgb.image_changed[pts2D_b[1, valid_mask], pts2D_b[0, valid_mask]]
            error = np.linalg.norm(colors_a.astype(np.float) - colors_b.astype(np.float), ord=2, axis=1)
            # utilities.printNumPyArray({'colors_a': colors_a, 'colors_b': colors_b, 'error': error})

            utilities.drawProjectionErrors(cam_a.rgb.image_changed, pts2D_a[:, valid_mask], cam_b.rgb.image_changed,
                                           pts2D_b[:, valid_mask],
                                           error, cam_a.name + '_' + cam_b.name, skip=10)

            errors.append(np.mean(error))

        # print('errors is = ' + str(errors))
        # Return the errors
        return errors


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for cam_a, cam_b in combinations(dataset_cameras.cameras, 2):
        params = opt.getParamsContainingPattern('C' + cam_a.name)
        params.extend(opt.getParamsContainingPattern('C' + cam_b.name))
        opt.pushResidual(name='C' + cam_a.name + '_C' + cam_b.name, params=params)

    opt.printResiduals()

    # ---------------------------------------
    # --- Compute the SPARSE MATRIX
    # ---------------------------------------
    opt.computeSparseMatrix()

    # ---------------------------------------
    # --- SETUP THE VISUALIZATION FUNCTION
    # ---------------------------------------
    print('view optimization = ' + str(args['view_optimization']))
    if args['view_optimization']:
        wm = KeyPressManager.KeyPressManager.WindowManager()
        W = gtk.gdk.screen_width()
        H = gtk.gdk.screen_height()

        # position the windows in the proper place
        wpw = 4
        start_row = 30
        for i, camera in enumerate(dataset_cameras.cameras):
            aspect = float(camera.rgb.image.shape[1]) / float(camera.rgb.image.shape[0])
            w = int(W / wpw)
            h = int(w / aspect)
            name = 'C' + camera.name + '_original'
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(name, w, h)
            cv2.imshow(name, camera.rgb.image)
            cv2.waitKey(10)
            cv2.moveWindow(name, w * i, start_row)
        cv2.waitKey(200)

        wpw = 4
        start_row = 30 + H / 4
        for i, camera in enumerate(dataset_cameras.cameras):
            aspect = float(camera.rgb.image.shape[1]) / float(camera.rgb.image.shape[0])
            w = int(W / wpw)
            h = int(w / aspect)
            name = 'C' + camera.name + '_current'
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(name, w, h)
            cv2.imshow(name, camera.rgb.image)
            cv2.waitKey(10)
            cv2.moveWindow(name, w * i, start_row)
        cv2.waitKey(200)

        i = 0
        wpw = 3
        start_row = 2 * (30 + H / 4)
        window_header_height = 15
        for cam_a, cam_b in combinations(dataset_cameras.cameras, 2):
            aspect = float(2 * camera.rgb.image.shape[1]) / float(camera.rgb.image.shape[0])
            w = int(W / wpw)
            h = int(w / aspect)
            name = cam_a.name + '_' + cam_b.name
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.waitKey(10)
            cv2.resizeWindow(name, w, h)
            utilities.drawProjectionErrors(cam_a.rgb.image, [], cam_b.rgb.image, [], [], name)
            cv2.waitKey(10)
            cv2.moveWindow(name, int(w * (i % wpw)), int(start_row + floor(i / wpw) * (h + window_header_height)))
            i = i + 1

        cv2.waitKey(50)

        wm.waitForKey(time_to_wait=None, verbose=True)


    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    def visualizationFunction(data):
        # Get the data
        data_cameras = data['data_cameras']

        for i, camera in enumerate(dataset_cameras.cameras):
            cv2.imshow('C' + camera.name + '_current', camera.rgb.image_changed)

        wm.waitForKey(0.01, verbose=False)


    opt.setVisualizationFunction(visualizationFunction, args['view_optimization'], niterations=5)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # opt.x = opt.addNoiseToX(noise=0.01)
    opt.fromXToData()
    # opt.callObjectiveFunction()
    # exit(0)

    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-8, 'gtol': 1e-8, 'diff_step': 1e-2})
