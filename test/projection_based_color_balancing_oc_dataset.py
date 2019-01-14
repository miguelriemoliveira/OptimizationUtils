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

import numpy as np
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

    # Dataset loader arguments
    ap.add_argument("-p", "--path_to_images", help="path to the folder that contains the OC dataset", required=True)
    ap.add_argument("-ext", "--image_extension", help="extension of the image files, e.g., jpg or png", default='jpg')
    ap.add_argument("-m", "--mesh_filename", help="full filename to input obj file, i.e. the 3D model", required=True)
    ap.add_argument("-i", "--path_to_intrinsics", help="path to intrinsics yaml file", required=True)
    ap.add_argument("-ucci", "--use_color_corrected_images", help="Use previously color corrected images",
                    action='store_true', default=False)
    ap.add_argument("-si", "--skip_images", help="skip images. Useful for fast testing", type=int, default=1)
    ap.add_argument("-vri", "--view_range_image", help="visualize sparse and dense range images", action='store_true',
                    default=False)
    ap.add_argument("-ms", "--marker_size", help="Size in meters of the aruco markers in the images", type=float,
                    required=True)
    ap.add_argument("-vad", "--view_aruco_detections", help="visualize aruco detections in the camera images",
                    action='store_true',
                    default=False)
    ap.add_argument("-va3d", "--view_aruco_3d", help="visualize aruco detections in a 3d window", action='store_true',
                    default=False)
    ap.add_argument("-va3dpc", "--view_aruco_3d_per_camera",
                    help="visualize aruco detections in a 3d window showing all the aruco detections (plot becomes quite dense)",
                    action='store_true',
                    default=False)
    # OptimizationUtils arguments
    ap.add_argument("-sv", "--skip_vertices", help="skip vertices. Useful for fast testing", type=int, default=1)
    ap.add_argument("-z", "--z_inconsistency_threshold", help="threshold for max z inconsistency value", type=float,
                    default=0.05)
    ap.add_argument("-vpv", "--view_projected_vertices", help="visualize projections of vertices onto images",
                    action='store_true', default=False)
    ap.add_argument("-vo", "--view_optimization", help="...", action='store_true', default=False)

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
    for i, camera in enumerate(dataset_cameras.cameras):
        # if i>0:
        dataset_cameras.cameras[i].rgb.image = utilities.addSafe(dataset_cameras.cameras[i].rgb.image, random.randint(-70, 70))

    # lets add a bias variable to each camera.rgb. This value will be used to change the image and optimize
    for i, camera in enumerate(dataset_cameras.cameras):
        camera.rgb.bias = random.randint(-30, 30)
        # camera.rgb.bias = 0

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('data_cameras', dataset_cameras)


    # opt.addModelData('data_arucos', dataset_arucos)

    # ------------  Cameras -----------------
    # Each camera will have a bias value which will be added to all pixels

    def setterBias(dataset, value, i):
        dataset.cameras[i].rgb.bias = value


    def getterBias(dataset, i):
        return [dataset.cameras[i].rgb.bias]


    # Add parameters related to the cameras
    for idx_camera, camera in enumerate(dataset_cameras.cameras):
        # if idx_camera == 0:  # First camera with static color
        #     bound_max = camera.rgb.bias + 0.00001
        #     bound_min = camera.rgb.bias - 0.00001
        # else:
        bound_max = camera.rgb.bias + 250
        bound_min = camera.rgb.bias - 250

        opt.pushParamScalar(group_name='bias_C' + camera.name, data_key='data_cameras',
                            getter=partial(getterBias, i=idx_camera),
                            setter=partial(setterBias, i=idx_camera),
                            bound_max=bound_max, bound_min=bound_min)

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
            camera.rgb.image_changed = utilities.addSafe(camera.rgb.image, camera.rgb.bias)
            camera.rgb.avg_changed = np.average(camera.rgb.image_changed)

        # Compute all the pair wise combinations of the set of cameras
        # Each element in the vector of errors is the difference of the average color for the combination
        for cam_a, cam_b in combinations(data_cameras.cameras, 2):
            # print(cam_a.name + ' with ' + cam_b.name)

            ci_a = cam_a.rgb.camera_info
            ci_b = cam_b.rgb.camera_info

            # get a list of 3D points in the map frame by concatenating the 3D measurements of cam_a and cam_b
            pts3D_in_map = np.concatenate([
                np.dot(cam_a.depth.matrix, cam_a.depth.vertices[:, 0::args['skip_vertices']]),
                np.dot(cam_b.depth.matrix, cam_b.depth.vertices[:, 0::args['skip_vertices']])], axis=1)

            pts2D_a, pts2D_b, valid_mask = utilities.projectToCameraPair(
                ci_a.K, ci_a.D, ci_a.width, ci_a.height, np.linalg.inv(cam_a.rgb.matrix), cam_a.rgb.image,
                cam_a.rgb.range_dense,
                ci_b.K, ci_b.D, ci_b.width, ci_b.height, np.linalg.inv(cam_b.rgb.matrix), cam_b.rgb.image,
                cam_b.rgb.range_dense,
                pts3D_in_map, z_inconsistency_threshold=args['z_inconsistency_threshold'],
                visualize=args['view_projected_vertices'])

            colors_a = cam_a.rgb.image_changed[pts2D_a[1, :], pts2D_a[0, :]]
            colors_b = cam_b.rgb.image_changed[pts2D_b[1, :], pts2D_b[0, :]]
            error = np.linalg.norm(colors_a.astype(np.float) - colors_b.astype(np.float), ord=2, axis=1)
            # utilities.printNumPyArray({'colors_a': colors_a, 'colors_b': colors_b, 'error': error})

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
    # position the windows in the proper place
    for i, camera in enumerate(dataset_cameras.cameras):
        cv2.namedWindow('C' + camera.name + '_original', cv2.WINDOW_NORMAL)
        cv2.moveWindow('C' + camera.name + '_original', 420 * i, 70)
        cv2.imshow('C' + camera.name + '_original', camera.rgb.image)

    for i, camera in enumerate(dataset_cameras.cameras):
        cv2.namedWindow('C' + camera.name + '_current', cv2.WINDOW_NORMAL)
        cv2.moveWindow('C' + camera.name + '_current', 420 * i, 450)
        cv2.imshow('C' + camera.name + '_current', camera.rgb.image)

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey(time_to_wait=None, verbose=True):
        exit(0)


    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    def visualizationFunction(data):
        # Get the data
        data_cameras = data['data_cameras']

        for i, camera in enumerate(dataset_cameras.cameras):
            cv2.imshow('C' + camera.name + '_current', camera.rgb.image_changed)

        wm = KeyPressManager.KeyPressManager.WindowManager()
        if wm.waitForKey(0.01, verbose=False):
            exit(0)


    opt.setVisualizationFunction(visualizationFunction, n_iterations=0)

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
    print("\n\nStarting optimization")
    opt.startOptimization(optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-8, 'gtol': 1e-8, 'diff_step': 1e-0})

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey(time_to_wait=None, verbose=True):
        exit(0)
