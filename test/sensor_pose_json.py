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
    # del dataset_sensors['sensors']['left_laser']
    # del dataset_sensors['sensors']['right_laser']
    # del dataset_sensors['collections']['0']
    # del dataset_sensors['collections']['1']
    del dataset_sensors['collections']['2']
    del dataset_sensors['collections']['3']
    del dataset_sensors['collections']['4']
    del dataset_sensors['collections']['5']

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
    chessboard_points = np.vstack(
        (chessboard_points, np.ones((1, args['chess_num_x'] * args['chess_num_y']), dtype=np.float)))

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
                objp[:, :2] = args['chess_size'] * np.mgrid[0:args['chess_num_x'], 0:args['chess_num_y']].T.reshape(-1,
                                                                                                                    2)

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
    # --- SETUP OPTIMIZER
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


    def getterCameraIntrinsics(data, sensor_name):
        fx = data['sensors'][sensor_name]['camera_info']['K'][0]
        fy = data['sensors'][sensor_name]['camera_info']['K'][4]
        cx = data['sensors'][sensor_name]['camera_info']['K'][2]
        cy = data['sensors'][sensor_name]['camera_info']['K'][5]
        D = data['sensors'][sensor_name]['camera_info']['D']
        intrinsics = [fx, fy, cx, cy]
        intrinsics.extend(D)
        return intrinsics


    def setterCameraIntrinsics(data, value, sensor_name):
        assert len(value) == 9, "value must be a list with length 9."
        data['sensors'][sensor_name]['camera_info']['K'][0] = value[0]
        data['sensors'][sensor_name]['camera_info']['K'][4] = value[1]
        data['sensors'][sensor_name]['camera_info']['K'][2] = value[2]
        data['sensors'][sensor_name]['camera_info']['K'][5] = value[3]
        data['sensors'][sensor_name]['camera_info']['D'] = value[4:]


    # Add parameters related to the sensors
    for sensor_key, sensor in dataset_sensors['sensors'].items():
        opt.pushParamV3(group_name='S_' + sensor_key + '_t', data_key='dataset_sensors',
                        getter=partial(getterSensorTranslation, sensor_name=sensor_key),
                        setter=partial(setterSensorTranslation, sensor_name=sensor_key),
                        suffix=['x', 'y', 'z'])

        opt.pushParamVector(group_name='S_' + sensor_key + '_r', data_key='dataset_sensors',
                            getter=partial(getterSensorRotation, sensor_name=sensor_key),
                            setter=partial(setterSensorRotation, sensor_name=sensor_key),
                            suffix=['1', '2', '3'])

        if sensor['msg_type'] == 'Image':  # if sensor is a camera add extrinsics
            opt.pushParamVector(group_name='S_' + sensor_key + '_I_', data_key='dataset_sensors',
                                getter=partial(getterCameraIntrinsics, sensor_name=sensor_key),
                                setter=partial(setterCameraIntrinsics, sensor_name=sensor_key),
                                suffix=['fx', 'fy', 'cx', 'cy', 'd0', 'd1', 'd2', 'd3', 'd4'])


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
                        suffix=['x', 'y', 'z'])

        opt.pushParamV3(group_name='C_' + collection_key + '_r', data_key='dataset_chessboard',
                        getter=partial(getterChessBoardRotation, collection=collection_key),
                        setter=partial(setterChessBoardRotation, collection=collection_key),
                        suffix=['1', '2', '3'])

    opt.printParameters()


    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
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
        dataset_sensors = data['dataset_sensors']
        dataset_chessboard = data['dataset_chessboard']

        errors = []

        for collection_key, collection in dataset_sensors['collections'].items():
            for sensor_key, sensor in dataset_sensors['sensors'].items():
                if not collection['labels'][sensor_key]['detected']:  # chessboard not detected by sensor in collection
                    continue

                if sensor['msg_type'] == 'Image':

                    # Compute chessboard points in local sensor reference frame
                    trans = dataset_chessboard[collection_key]['trans']
                    quat = dataset_chessboard[collection_key]['quat']
                    root_T_chessboard = utilities.translationQuaternionToTransform(trans, quat)

                    sensor_T_root = np.linalg.inv(utilities.getAggregateTransform(sensor['chain'],
                                                                                  collection['transforms']))

                    pts_root = np.dot(root_T_chessboard, chessboard_points)
                    pts_sensor = np.dot(sensor_T_root, pts_root)

                    K = np.ndarray((3, 3), buffer=np.array(sensor['camera_info']['K']), dtype=np.float)
                    D = np.ndarray((5, 1), buffer=np.array(sensor['camera_info']['D']), dtype=np.float)
                    width = collection['data'][sensor_key]['width']
                    height = collection['data'][sensor_key]['height']

                    pixs, valid_pixs, dists = utilities.projectToCamera(K, D, width, height, pts_sensor[0:3, :])
                    pixs_ground_truth = collection['labels'][sensor_key]['idxs']

                    array_gt = np.zeros(pixs.shape, dtype=np.float)
                    for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                        array_gt[0][idx] = pix_ground_truth['x']
                        array_gt[1][idx] = pix_ground_truth['y']

                    error_sum = 0
                    for idx in range(0, args['chess_num_x'] * args['chess_num_y']):
                        error_sum += math.sqrt(
                            (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)

                    error = error_sum / (args['chess_num_x'] * args['chess_num_y'])
                    errors.append(error)

                    # store projected pixels into dataset_sensors dict for drawing in visualization function
                    idxs_projected = []
                    for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                        idxs_projected.append({'x': pixs[0][idx], 'y': pixs[1][idx]})

                    collection['labels'][sensor_key]['idxs_projected'] = idxs_projected

                    if not 'idxs_initial' in collection['labels'][sensor_key]:  # store the first projections
                        collection['labels'][sensor_key]['idxs_initial'] = deepcopy(idxs_projected)

                elif sensor['msg_type'] == 'LaserScan':

                    # Get laser points that belong to the chessboard
                    idxs = collection['labels'][sensor_key]['idxs']
                    rhos = [collection['data'][sensor_key]['ranges'][idx] for idx in idxs]
                    thetas = [collection['data'][sensor_key]['angle_min'] +
                              collection['data'][sensor_key]['angle_increment'] * idx for idx in idxs]

                    # Convert from polar to cartesian coordinates and create np array with xyz coords
                    pts_laser = np.zeros((3, len(rhos)), np.float32)
                    for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                        pts_laser[0, idx] = rho * math.cos(theta)
                        pts_laser[1, idx] = rho * math.sin(theta)

                    pts_laser = np.vstack((pts_laser, np.ones((1, pts_laser.shape[1]), dtype=np.float)))

                    # Get transforms
                    root_T_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                    pts_root = np.dot(root_T_sensor, pts_laser)

                    trans = dataset_chessboard[collection_key]['trans']
                    quat = dataset_chessboard[collection_key]['quat']
                    chessboard_T_root = np.linalg.inv(utilities.translationQuaternionToTransform(trans, quat))

                    pts_chessboard = np.dot(chessboard_T_root, pts_root)

                    # TODO error in meters? Seems small when compared with pixels ...
                    error = np.average(np.absolute(pts_chessboard[2, :])) * 100

                    errors.append(error)

                    # Store for visualization
                    collection['labels'][sensor_key]['pts_root'] = pts_root
                    if not 'pts_root_initial' in collection['labels'][sensor_key]:  # store the first projections
                        collection['labels'][sensor_key]['pts_root_initial'] = deepcopy(pts_root)

                else:
                    raise ValueError("Unknown sensor msg_type")

                # print('error for sensor ' + sensor_key + ' in collection ' + collection_key + ' is ' + str(error))
        # Return the errors
        return errors


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    # Each error is computed after the sensor and the chessboard of a collection. Thus, each error will be affected
    # by the parameters tx,ty,tz,r1,r2,r3 of the sensor and the chessboard

    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][sensor_key]['detected']:  # if chessboard not detected by sensor in collection
                continue

            params = opt.getParamsContainingPattern('S_' + sensor_key)  # sensor related params
            params.extend(opt.getParamsContainingPattern('C_' + collection_key + '_'))  # chessboard related params
            opt.pushResidual(name=collection_key + '_' + sensor_key, params=params)

    print('residuals = ' + str(opt.residuals))

    # ---------------------------------------
    # --- Compute the SPARSE MATRIX
    # ---------------------------------------
    opt.computeSparseMatrix()

    # ---------------------------------------
    # --- SETUP THE VISUALIZATION FUNCTION
    # ---------------------------------------
    if args['view_optimization']:
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text

        # Create colormaps to be used. Sort the keys to have the same color distribution over the collections
        color_map = cm.plasma(np.linspace(0, 1, args['chess_num_x'] * args['chess_num_y']))

        color_map_collections = cm.Set3(np.linspace(0, 1, len(dataset_sensors['collections'].keys())))
        for idx, collection_key in enumerate(sorted(dataset_sensors['collections'].keys())):
            dataset_sensors['collections'][collection_key]['color'] = color_map_collections[idx, :]
            dataset_chessboard[collection_key]['color'] = color_map_collections[idx, :]

        color_map_sensors = cm.gist_rainbow(np.linspace(0, 1, len(dataset_sensors['sensors'].keys())))
        for idx, sensor_key in enumerate(sorted(dataset_sensors['sensors'].keys())):
            dataset_sensors['sensors'][sensor_key]['color'] = color_map_sensors[idx, :]

        # Create opencv windows. One per sensor image and collection
        counter = 0
        for collection_key, collection in dataset_sensors['collections'].items():
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
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
        # limit = 1.5
        ax.set_xlim3d(-1.5, 1.5), ax.set_ylim3d(-4, 1.5), ax.set_zlim3d(-.5, 1.5)
        ax.view_init(elev=27, azim=46)

        # Draw world axis
        world_T_world = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float)
        utilities.drawAxis3D(ax, world_T_world, "root", axis_scale=0.5, line_width=1.5)

        # Draw sensor poses (use sensor pose from collection '0' since they are all the same)
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            root_T_sensor = utilities.getAggregateTransform(sensor['chain'],
                                                            dataset_sensors['collections']['0']['transforms'])
            sensor['handle'] = utilities.drawAxis3D(ax, root_T_sensor, sensor_key, text_color=sensor['color'],
                                                    axis_scale=0.3, line_width=1.5)

        # Draw chessboard poses
        for collection_key, collection in dataset_chessboard.items():
            root_T_chessboard = utilities.translationQuaternionToTransform(collection['trans'], collection['quat'])

            collection['handle'] = utilities.drawChessBoard(ax, root_T_chessboard, chessboard_points,
                                                            'C' + collection_key, chess_num_x=args['chess_num_x'],
                                                            chess_num_y=args['chess_num_y'], color=collection['color'],
                                                            axis_scale=0.5, line_width=2)

        # Draw laser data
        for collection_key, collection in dataset_sensors['collections'].items():
            for sensor_key, sensor in dataset_sensors['sensors'].items():
                if not collection['labels'][sensor_key]['detected']:  # chessboard not detected by sensor in collection
                    continue
                if not sensor['msg_type'] == 'LaserScan':
                    continue

                # Get laser points that belong to the chessboard
                idxs = collection['labels'][sensor_key]['idxs']
                rhos = [collection['data'][sensor_key]['ranges'][idx] for idx in idxs]
                thetas = [collection['data'][sensor_key]['angle_min'] +
                          collection['data'][sensor_key]['angle_increment'] * idx for idx in idxs]

                # Convert from polar to cartesian coordinates and create np array with xyz coords
                pts_laser = np.zeros((3, len(rhos)), np.float32)
                for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                    pts_laser[0, idx] = rho * math.cos(theta)
                    pts_laser[1, idx] = rho * math.sin(theta)

                # homogenize points
                pts_laser = np.vstack((pts_laser, np.ones((1, pts_laser.shape[1]), dtype=np.float)))

                # Transform points to root
                root_T_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                pts_root = np.dot(root_T_sensor, pts_laser)

                # draw points
                sensor['pts_handle'] = utilities.drawPoints3D(ax, None, pts_root, color=collection['color'],
                                                              marker_size=1.5, line_width=2.2, marker='-',
                                                              mfc=collection['color'], text=None,
                                                              text_color=sensor['color'], sensor_color=sensor['color'],
                                                              handles=None)

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(time_to_wait=0.01, verbose=True):
            exit(0)

    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    def visualizationFunction(data):
        # Get the data from the model
        dataset_sensors = data['dataset_sensors']
        dataset_chessboard = data['dataset_chessboard']

        for collection_key, collection in dataset_sensors['collections'].items():
            for sensor_key, sensor in dataset_sensors['sensors'].items():

                if not collection['labels'][sensor_key]['detected']:  # chessboard not detected by sensor in collection
                    continue

                if sensor['msg_type'] == 'Image':
                    filename = os.path.dirname(args['json_file']) + '/' + collection['data'][sensor_key]['data_file']

                    # TODO should not read image again from disk
                    image = cv2.imread(filename)
                    width = collection['data'][sensor_key]['width']
                    height = collection['data'][sensor_key]['height']
                    diagonal = math.sqrt(width ** 2 + height ** 2)

                    # Draw projected points (as dots)
                    for idx, point in enumerate(collection['labels'][sensor_key]['idxs_projected']):
                        x = int(round(point['x']))
                        y = int(round(point['y']))
                        color = (color_map[idx, 2] * 255, color_map[idx, 1] * 255, color_map[idx, 0] * 255)
                        cv2.line(image, (x, y), (x, y), color, int(6E-3 * diagonal))

                    # Draw ground truth points (as squares)
                    for idx, point in enumerate(collection['labels'][sensor_key]['idxs']):
                        x = int(round(point['x']))
                        y = int(round(point['y']))
                        color = (color_map[idx, 2] * 255, color_map[idx, 1] * 255, color_map[idx, 0] * 255)
                        utilities.drawSquare2D(image, x, y, int(8E-3 * diagonal), color=color, thickness=2)

                    # Draw initial projected points (as crosses)
                    for idx, point in enumerate(collection['labels'][sensor_key]['idxs_initial']):
                        x = int(round(point['x']))
                        y = int(round(point['y']))
                        color = (color_map[idx, 2] * 255, color_map[idx, 1] * 255, color_map[idx, 0] * 255)
                        utilities.drawCross2D(image, x, y, int(8E-3 * diagonal), color=color, thickness=1)

                    window_name = sensor_key + '-' + collection_key
                    cv2.imshow(window_name, image)

                elif sensor['msg_type'] == 'LaserScan':
                    pts_root = collection['labels'][sensor_key]['pts_root']
                    utilities.drawPoints3D(ax, None, pts_root, line_width=1.0, handles=sensor['pts_handle'])
                else:
                    raise ValueError("Unknown sensor msg_type")


        # Draw sensor poses (use sensor pose from collection '0' since they are all the same)
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            root_T_sensor = utilities.getAggregateTransform(sensor['chain'],
                                                            dataset_sensors['collections']['0']['transforms'])
            utilities.drawAxis3D(ax, root_T_sensor, sensor_key, axis_scale=0.3, line_width=2,
                                 handles=sensor['handle'])

        # Draw chessboard poses
        for idx, (collection_key, collection) in enumerate(dataset_chessboard.items()):
            root_T_chessboard = utilities.translationQuaternionToTransform(collection['trans'], collection['quat'])
            color_collection = color_map_collections[idx, :]
            utilities.drawChessBoard(ax, root_T_chessboard, chessboard_points, 'C' + collection_key,
                                     chess_num_x=args['chess_num_x'], chess_num_y=args['chess_num_y'],
                                     color=color_collection, axis_scale=0.3, line_width=2,
                                     handles=collection['handle'])

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
        optimization_options={'x_scale': 'jac', 'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5, 'diff_step': 1e-4})

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
