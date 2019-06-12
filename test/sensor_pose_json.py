#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import json
import math

from tf import transformations

import OptimizationUtils.OptimizationUtils as OptimizationUtils
import KeyPressManager.KeyPressManager as KeyPressManager
import OptimizationUtils.utilities as utilities
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plyfile as plyfile
import cv2
import argparse
import subprocess
import os
import shutil
from copy import deepcopy
from functools import partial
from matplotlib import cm
from scipy.spatial.distance import euclidean
from open3d import *

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
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-json", "--json_file", help="Json file containing input dataset.", type=str, required=True)
    ap.add_argument("-csize", "--chess_size", help="Size in meters of the side of the chessboard's squares.",
                    type=float, required=True)
    ap.add_argument("-cnumx", "--chess_num_x", help="Chessboard's number of squares in horizontal dimension.",
                    type=int, required=True)
    ap.add_argument("-cnumy", "--chess_num_y", help="Chessboard's number of squares in vertical dimension.",
                    type=int, required=True)
    args = vars(ap.parse_args())
    print("\nArgument list=")
    print(args)
    print('\n')

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    """ Loads a json file containing the detections"""
    f = open(args['json_file'], 'r')
    dataset_sensors = json.load(f)

    # Remove some sensors if desired. Should be done here according to the examples bellow.
    del dataset_sensors['sensors']['frontal_laser']
    del dataset_sensors['sensors']['left_laser']
    del dataset_sensors['sensors']['right_laser']

    print('Loaded dataset containing ' + str(len(dataset_sensors['sensors'].keys())) + ' sensors and ' + str(
        len(dataset_sensors['collections'].keys())) + ' collections.')

    # Load images from files into memory. Images in the json file are stored in separate png files and in their place
    # a field "data_file" is saved with the path to the file. We must load the images from the disk.
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not sensor['msg_type'] == 'Image':  # nothing to do here.
                continue

            filename = os.path.dirname(args['json_file']) + '/' + collection['data'][sensor_key]['data_file']
            collection['data'][sensor_key]['data'] = cv2.imread(filename)

    # ---------------------------------------
    # --- CREATE CHESSBOARD DATASET
    # ---------------------------------------
    dataset_chessboard = {}

    objp = np.zeros((args['chess_num_x'] * args['chess_num_y'], 3), np.float32)
    objp[:, :2] = args['chess_size'] * np.mgrid[0:args['chess_num_x'], 0:args['chess_num_y']].T.reshape(-1, 2)
    chessboard_points = np.transpose(objp)
    chessboard_points = np.vstack((chessboard_points, np.ones((1, args['chess_num_x'] * args['chess_num_y']), dtype=np.float)))

    for sensor_key, sensor in dataset_sensors['sensors'].items():
        # if sensor['msg_type'] == 'Image' and sensor_key == 'top_right_camera':
        if sensor['msg_type'] == 'Image' and sensor_key == 'frontal_camera':

            for collection_key, collection in dataset_sensors['collections'].items():

                print collection['data'][sensor_key]
                image_rgb = collection['data'][sensor_key]['data']

                mtx = np.ndarray((3, 3), dtype=np.float,
                                 buffer=np.array(sensor['camera_info']['K']))

                dist = np.ndarray((5, 1), dtype=np.float,
                                  buffer=np.array(sensor['camera_info']['D']))


                def draw(img, corners, imgpts):
                    corner = tuple(corners[0].ravel())
                    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
                    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
                    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)
                    return img


                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                objp = np.zeros((args['chess_num_x'] * args['chess_num_y'], 3), np.float32)
                objp[:, :2] = args['chess_size'] * np.mgrid[0:args['chess_num_x'], 0:args['chess_num_y']].T.reshape(-1, 2)

                axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)

                gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (args['chess_num_x'], args['chess_num_y']), None)
                if ret == True:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    # Find the rotation and translation vectors.
                    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
                    print("First guess is:\n" + str(rvecs) + "\n" + str(tvecs))

                    # project 3D points to image plane
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                    image_rgb = draw(image_rgb, corners2, imgpts)
                    # cv2.imshow('img', image_rgb)

                    root_T_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                    print('root_T_sensor=\n' + str(root_T_sensor) + '\n\n')

                    sensor_T_chessboard = utilities.traslationRodriguesToTransform(tvecs, rvecs)
                    print('sensor_T_chessboard =\n ' + str(sensor_T_chessboard))

                    root_T_chessboard = np.dot(root_T_sensor, sensor_T_chessboard)
                    print('root_T_chessboard =\n ' + str(root_T_chessboard))

                    d = {}
                    d['trans'] = list(root_T_chessboard[0:3, 3])

                    T = deepcopy(root_T_chessboard)
                    T[0:3, 3] = 0  # remove translation component from 4x4 matrix
                    d['quat'] = list(transformations.quaternion_from_matrix(T))

                    dataset_chessboard[collection_key] = d

                    # cv2.waitKey(10)

    # print(dataset_chessboard)

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('\nInitializing optimizer...')
    opt = OptimizationUtils.Optimizer()

    opt.addModelData('dataset_sensors', dataset_sensors)
    opt.addModelData('dataset_chessboard', dataset_chessboard)


    # ------------  Sensors -----------------
    # Each sensor will have a position (tx,ty,tz) and a rotation (r1,r2,r3)

    def getterSensorTranslation(data, sensor_name):
        calibration_parent = data['sensors'][sensor_name]['calibration_parent']
        calibration_child = data['sensors'][sensor_name]['calibration_child']
        transform_key = calibration_parent + '-' + calibration_child
        # We use collection "0" and assume they are all the same
        return data['collections']['0']['transforms'][transform_key]['trans']


    def setterSensorTranslation(data, value, sensor_name):
        assert len(value) == 3, "value must be a list with length 3."

        calibration_parent = data['sensors'][sensor_name]['calibration_parent']
        calibration_child = data['sensors'][sensor_name]['calibration_child']
        transform_key = calibration_parent + '-' + calibration_child

        for _collection_key in data['collections']:
            data['collections'][_collection_key]['transforms'][transform_key]['trans'] = value


    def getterSensorRotation(data, sensor_name):
        calibration_parent = data['sensors'][sensor_name]['calibration_parent']
        calibration_child = data['sensors'][sensor_name]['calibration_child']
        transform_key = calibration_parent + '-' + calibration_child

        # We use collection "0" and assume they are all the same
        quat = data['collections']['0']['transforms'][transform_key]['quat']
        hmatrix = transformations.quaternion_matrix(quat)
        matrix = hmatrix[0:3, 0:3]

        return utilities.matrixToRodrigues(matrix)


    def setterSensorRotation(data, value, sensor_name):
        assert len(value) == 3, "value must be a list with length 3."

        matrix = utilities.rodriguesToMatrix(value)
        hmatrix = np.identity(4)
        hmatrix[0:3, 0:3] = matrix
        quat = transformations.quaternion_from_matrix(hmatrix)

        calibration_parent = data['sensors'][sensor_name]['calibration_parent']
        calibration_child = data['sensors'][sensor_name]['calibration_child']
        transform_key = calibration_parent + '-' + calibration_child

        for collection_key in data['collections']:
            data['collections'][collection_key]['transforms'][transform_key]['quat'] = quat


    # Add parameters related to the sensors
    for sensor_key, sensor in dataset_sensors['sensors'].items():
        opt.pushParamV3(group_name='S_' + sensor_key + '_t', data_key='dataset_sensors',
                        getter=partial(getterSensorTranslation, sensor_name=sensor_key),
                        setter=partial(setterSensorTranslation, sensor_name=sensor_key),
                        sufix=['x', 'y', 'z'])

        # Add the rotation
        opt.pushParamV3(group_name='S' + sensor_key + '_r', data_key='dataset_sensors',
                        getter=partial(getterSensorRotation, sensor_name=sensor_key),
                        setter=partial(setterSensorRotation, sensor_name=sensor_key),
                        sufix=['1', '2', '3'])


    # ------------  Chessboard -----------------
    # Each Chessboard will have the position (tx,ty,tz) and rotation (r1,r2,r3)

    def getterChessBoardTranslation(data, collection):
        return data[collection]['trans']


    def setterChessBoardTranslation(data, value, collection):
        assert len(value) == 3, "value must be a list with length 3."

        data[collection]['trans'] = value


    def getterChessBoardRotation(data, collection):
        quat = data[collection]['quat']
        hmatrix = transformations.quaternion_matrix(quat)
        matrix = hmatrix[0:3, 0:3]
        return utilities.matrixToRodrigues(matrix)


    def setterChessBoardRotation(data, value, collection):
        assert len(value) == 3, "value must be a list with length 3."

        matrix = utilities.rodriguesToMatrix(value)
        hmatrix = np.identity(4)
        hmatrix[0:3, 0:3] = matrix
        quat = transformations.quaternion_from_matrix(hmatrix)
        data[collection]['quat'] = quat


    # Add translation and rotation parameters related to the Chessboards
    for collection_key in dataset_chessboard:
        opt.pushParamV3(group_name='C_' + collection_key + '_t', data_key='dataset_chessboard',
                        getter=partial(getterChessBoardTranslation, collection=collection_key),
                        setter=partial(setterChessBoardTranslation, collection=collection_key),
                        sufix=['x', 'y', 'z'])

        opt.pushParamV3(group_name='C_' + collection_key + '_r', data_key='dataset_chessboard',
                        getter=partial(getterChessBoardRotation, collection=collection_key),
                        setter=partial(setterChessBoardRotation, collection=collection_key),
                        sufix=['1', '2', '3'])

    opt.printParameters()

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    first_time = True

    def objectiveFunction(data):
        """
        Computes the vector of errors. There should be an error for each stamp, sensor and chessboard tuple.
        The computation of the error varies according with the modality of the sensor:
            - Reprojection error for camera to chessboard
            - Point to plane distance for 2D laser scanners
            - (...)
            :return: a vector of residuals
        """

        # Get the data from the model
        print('\n\nObjective function call:')
        dataset_sensors = data['dataset_sensors']
        dataset_chessboard = data['dataset_chessboard']

        errors = []

        for collection_key, collection in dataset_sensors['collections'].items():
            # print('for collection_key ' + str(collection_key))
            for sensor_key, sensor in dataset_sensors['sensors'].items():

                # print('for sensor_key ' + str(sensor_key))
                if not collection['labels'][sensor_key]['detected']:  # chessboard not detected by sensor in collection
                    continue

                if sensor['msg_type'] == 'Image':
                    # print('Computing error ... ')

                    # Compute chessboard points in local sensor reference frame
                    trans = dataset_chessboard[collection_key]['trans']
                    quat = dataset_chessboard[collection_key]['quat']
                    root_T_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                    # print('root_T_chessboard=\n' + str(root_T_chessboard))

                    sensor_T_root = np.linalg.inv(utilities.getAggregateTransform(sensor['chain'],
                                                                                  collection['transforms']))
                    # print('sensor_T_root=\n' + str(sensor_T_root))

                    # pts_root = np.dot(chessboard_T_root, chessboard_points)
                    pts_root = np.dot(root_T_chessboard, chessboard_points)
                    # print('pts_root')
                    # print(pts_root)

                    pts_sensor = np.dot(sensor_T_root, pts_root)
                    # print('pts_sensor')
                    # print(pts_sensor)

                    K = np.ndarray((3, 3), buffer=np.array(sensor['camera_info']['K']), dtype=np.float)
                    D = np.ndarray((5, 1), buffer=np.array(sensor['camera_info']['D']), dtype=np.float)
                    width = collection['data'][sensor_key]['width']
                    height = collection['data'][sensor_key]['height']

                    pixs, valid_pixs, dists = utilities.projectToCamera(K, D, width, height, pts_sensor[0:3, :])
                    # print('pixs')
                    # print(pixs)
                    # print(pixs[:,0])
                    # print(pixs.shape)

                    pixs_ground_truth = collection['labels'][sensor_key]['idxs']
                    # print('pixs_ground_truth')
                    # print(pixs_ground_truth)

                    array_gt = np.zeros(pixs.shape, dtype=np.float)
                    for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                        array_gt[0][idx] = pix_ground_truth['x']
                        array_gt[1][idx] = pix_ground_truth['y']

                    # print('array_gt')
                    # print(array_gt)
                    # print(array_gt[:,0])
                    # print(array_gt.shape)

                    # TODO revise the computation of matrix norms

                    # error = np.linalg.norm(pixs[:, 0:2] - array_gt[:, 0:2])
                    # error = np.linalg.norm(pixs[0:1, :] - array_gt[0:1, :])

                    # error = math.sqrt( (pixs[0,0] - array_gt[0,0])**2 + (pixs[0,1] - array_gt[0,1])**2 ) # this was wrong!
                    # error = math.sqrt( (pixs[0,0] - array_gt[0,0])**2 + (pixs[1,0] - array_gt[1,0])**2 )

                    error_sum = 0
                    for idx in range(0, args['chess_num_x'] * args['chess_num_y']):
                        error_sum += math.sqrt(
                            (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)

                    error = error_sum / (args['chess_num_x'] * args['chess_num_y'])
                    print('error for sensor ' + sensor_key + ' in collection ' + collection_key + ' is ' + str(error))
                    errors.append(error)

                    # store projected pixels into dataset_sensors dict for drawing in visualization function
                    idxs_projected = []
                    for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                        idxs_projected.append({'x': pixs[0][idx], 'y': pixs[1][idx]})

                    collection['labels'][sensor_key]['idxs_projected'] = idxs_projected

                    global first_time
                    if first_time:
                        collection['labels'][sensor_key]['idxs_initial'] = deepcopy(idxs_projected)
                        first_time = False

                elif sensor['msg_type'] == 'LaserScan':
                    # TODO compute the error for lasers
                    errors.append(0)
                else:
                    raise ValueError("Unknown sensor msg_type")



        # Return the errors
        return errors


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    # TODO residuals: each error is computed after the sensor and the chessboard of a collection.
    #  Thus, each error will be affected by the parameters tx,ty,tz,r1,r2,r3 of the sensor and the chessboard

    for collection_key, collection in dataset_sensors['collections'].items():
        # print('for collection_key ' + str(collection_key))
        for sensor_key, sensor in dataset_sensors['sensors'].items():

            # print('for sensor_key ' + str(sensor_key))
            if not collection['labels'][sensor_key]['detected']:  # if chessboard not detected by sensor in collection
                continue

            # compute cost function
            params = opt.getParamsContainingPattern('S_' + sensor_key)
            params.extend(opt.getParamsContainingPattern('C_' + collection_key + '_'))
            opt.pushResidual(name=collection_key + '_' + sensor_key, params=params)

    print('residuals = ' + str(opt.residuals))

    # ---------------------------------------
    # --- Compute the SPARSE MATRIX
    # ---------------------------------------
    opt.computeSparseMatrix()

    # ---------------------------------------
    # --- SETUP THE VISUALIZATION FUNCTION
    # ---------------------------------------
    # TODO we should plot the poses of the cameras and the chessboards in 3D,
    #  and perhaps the sensor data with the projections.

    if args['view_optimization']:

        # Create opencv windows. One per sensor image and collection
        counter = 0
        for collection_key, collection in dataset_sensors['collections'].items():
            # print('for collection_key ' + str(collection_key))
            for sensor_key, sensor in dataset_sensors['sensors'].items():

                # print('for sensor_key ' + str(sensor_key))
                if not collection['labels'][sensor_key]['detected']:  # chessboard not detected by sensor in collection
                    continue

                if sensor['msg_type'] == 'Image':
                    filename = os.path.dirname(args['json_file']) + '/' + collection['data'][sensor_key]['data_file']

                    image = cv2.imread(filename)
                    window_name = sensor_key + '-' + collection_key
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.moveWindow(window_name, 300 * counter, 50)
                    cv2.imshow(window_name, image)
                    counter += 1

        # Create a 3D plot in which the sensor poses and chessboards are drawn
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #
        #     ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        #     ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
        #     limit = 1.5
        #     ax.set_xlim3d(-limit, limit), ax.set_ylim3d(-limit, limit), ax.set_zlim3d(-limit, limit)
        #     ax.view_init(elev=122, azim=-87)
        #
        #     # Draw world axis
        #     world_T_world = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float)
        #     utilities.drawAxis3D(ax, world_T_world, "world", axis_scale=0.7, line_width=3)
        #
        #     # Draw cameras
        #     for camera in dataset_cameras.cameras:
        #         camera.handle_frame = utilities.drawAxis3D(ax, camera.rgb.matrix, "C" + camera.name, axis_scale=0.3,
        #                                                    line_width=2)
        #         # print("camera " + camera.name + " " + str(camera.handle_frame))
        #
        #     # Draw Arucos
        #     dataset_arucos.handles = {}
        #     for aruco_id, aruco in dataset_arucos.arucos.items():
        #         dataset_arucos.handles[aruco_id] = utilities.drawAxis3DOrigin(ax, aruco.matrix, 'A' + str(aruco_id),
        #                                                                       line_width=1.0,
        #                                                                       fontsize=8,
        #                                                                       handles=None)
        #         # print("aruco " + str(aruco_id) + "= " + str(dataset_arucos.handles[aruco_id]))
        #
        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(time_to_wait=0.01, verbose=True):
            exit(0)

    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text


    def visualizationFunction(data):
        # Get the data from the model
        dataset_sensors = data['dataset_sensors']
        dataset_chessboard = data['dataset_chessboard']

        for collection_key, collection in dataset_sensors['collections'].items():
            # print('for collection_key ' + str(collection_key))
            for sensor_key, sensor in dataset_sensors['sensors'].items():

                # print('for sensor_key ' + str(sensor_key))
                if not collection['labels'][sensor_key]['detected']:  # chessboard not detected by sensor in collection
                    continue

                if sensor['msg_type'] == 'Image':
                    filename = os.path.dirname(args['json_file']) + '/' + collection['data'][sensor_key]['data_file']

                    # TODO should not read image again from disk
                    image = cv2.imread(filename)
                    width = collection['data'][sensor_key]['width']
                    height = collection['data'][sensor_key]['height']
                    diagonal = math.sqrt(width ** 2 + height ** 2)

                    points_projected = collection['labels'][sensor_key]['idxs_projected']
                    for point_projected in points_projected:
                        x = int(round(point_projected['x']))
                        y = int(round(point_projected['y']))
                        cv2.line(image, (x, y), (x, y), (255, 0, 0), int(1E-2 * diagonal))

                    points_ground_truth = collection['labels'][sensor_key]['idxs']
                    for point_ground_truth in points_ground_truth:
                        x = int(round(point_ground_truth['x']))
                        y = int(round(point_ground_truth['y']))
                        utilities.drawSquare2D(image, x, y, int(8E-3 * diagonal), color=(0, 0, 255), thickness=2)

                    window_name = sensor_key + '-' + collection_key
                    cv2.imshow(window_name, image)

                elif sensor['msg_type'] == 'LaserScan':
                    pass
                else:
                    raise ValueError("Unknown sensor msg_type")

        # for i, _camera in enumerate(data_cameras.cameras):
        #     image = deepcopy(_camera.rgb.image)
        #     # print("Cam " + str(camera.name))
        #     for _aruco_id, _aruco_detection in _camera.rgb.aruco_detections.items():
        #         # print("Aruco " + str(aruco_id))
        #         # print("Pixel center coords (ground truth) = " + str(aruco_detection.center))  # ground truth
        #
        #         utilities.drawSquare2D(image, _aruco_detection.center[0], _aruco_detection.center[1], 10,
        #                                color=(0, 0, 255), thickness=2)
        #
        #         cv2.putText(image, "Id:" + str(_aruco_id), _aruco_detection.center, font, 1, (0, 255, 0), 2,
        #                     cv2.LINE_AA)
        #
        #         # cv2.line(image, aruco_detection.center, aruco_detection.center, (0, 0, 255), 10)
        #         # print("Pixel center projected = " + str(aruco_detection.projected))  # ground truth
        #
        #         if 0 < _aruco_detection.projected[0] < _camera.rgb.camera_info.width \
        #                 and 0 < _aruco_detection.projected[1] < _camera.rgb.camera_info.height:
        #             x = int(_aruco_detection.projected[0])
        #             y = int(_aruco_detection.projected[1])
        #             # cv2.line(image, aruco_detection.projected, aruco_detection.projected, (255, 0, 0), 10)
        #             cv2.line(image, (x, y), (x, y), (255, 0, 0), 10)
        #
        #         # TODO: debug drawing first detection code
        #         if 0 < _aruco_detection.first_projection[0] < _camera.rgb.camera_info.width \
        #                 and 0 < _aruco_detection.first_projection[1] < _camera.rgb.camera_info.height:
        #             x = int(_aruco_detection.first_projection[0])
        #             y = int(_aruco_detection.first_projection[1])
        #             # cv2.line(image, aruco_detection.first_projection, aruco_detection.first_projection, (0, 255, 0), 10)
        #             cv2.line(image, (x, y), (x, y), (0, 255, 0), 10)
        #
        #     cv2.imshow('Cam ' + _camera.name, image)
        #
        # # Draw camera's axes
        # for _camera in data_cameras.cameras:
        #     utilities.drawAxis3D(ax=ax, transform=_camera.rgb.matrix, text="C" + _camera.name, axis_scale=0.3,
        #                          line_width=2,
        #                          handles=_camera.handle_frame)
        #
        # # Draw Arucos
        # for _aruco_id, _aruco in data_arucos.arucos.items():
        #     utilities.drawAxis3DOrigin(ax, _aruco.matrix, 'A' + str(_aruco_id), line_width=1.0,
        #                                handles=data_arucos.handles[_aruco_id])

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)


    opt.setVisualizationFunction(visualizationFunction, args['view_optimization'], niterations=1)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # Already created when pushing the parameters

    # opt.x = opt.addNoiseToX(noise=0.1)
    # opt.fromXToData()
    # opt.callObjectiveFunction()
    # exit(0)

    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    # print("\n\nStarting optimization")

    # This optimizes well
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5, 'diff_step': 1e-3})

    # opt.startOptimization(
    #    optimization_options={'x_scale': 'jac', 'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5, 'diff_step': 1e-4,
    #                          'max_nfev': 1})

    # This optimized forever but was already at 1.5 pixels avg errror and going when I interrupted it
    # opt.startOptimization(optimization_options={'x_scale': 'jac', 'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'diff_step': 1e-4})

    print('\n-----------------')
    opt.printParameters(opt.x0, text='Initial parameters')
    print('\n')
    opt.printParameters(opt.xf, text='Final parameters')

    # ---------------------------------------
    # --- Save Results
    # ---------------------------------------
    # Todo should be saved back to a json or directly to xacro?
