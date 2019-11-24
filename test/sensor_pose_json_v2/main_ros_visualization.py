#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import json
import pprint

import rospy
import tf
import visualization_msgs
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, sensor_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import OptimizationUtils.OptimizationUtils as OptimizationUtils
import KeyPressManager.KeyPressManager as KeyPressManager
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from functools import partial
from matplotlib import cm
from open3d import *

from getter_and_setters import *
from objective_function import *
from test.sensor_pose_json_v2.visualization import *


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def walk(node):
    for key, item in node.items():
        if isinstance(item, dict):
            walk(item)
        else:
            if isinstance(item, np.ndarray) and key == 'data':  # to avoid saning images in the json
                del node[key]

            elif isinstance(item, np.ndarray):
                node[key] = item.tolist()
                print('Converted to list')
            pass


# Save to json file

def createJSONFile(output_file, input):
    D = deepcopy(input)

    walk(D)

    print("Saving the json output file to " + str(output_file) + ", please wait, it could take a while ...")
    f = open(output_file, 'w')
    json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)  # to get only four decimal places on the json file
    print >> f, json.dumps(D, indent=2, sort_keys=True)
    f.close()
    print("Completed.")


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
def main():
    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-json", "--json_file", help="Json file containing input dataset.", type=str, required=True)
    ap.add_argument("-csize", "--chess_size", help="Size in meters of the side of the chessboard's squares.",
                    type=float, required=True)
    ap.add_argument("-cradius", "--chess_radius",
                    help="Radius in meters of the maximum side of the chessboard calibration pattern.",
                    type=float, required=True)
    ap.add_argument("-cnumx", "--chess_num_x", help="Chessboard's number of corners in horizontal dimension.",
                    type=int, required=True)
    ap.add_argument("-cnumy", "--chess_num_y", help="Chessboard's number of corners in vertical dimension.",
                    type=int, required=True)
    ap.add_argument("-si", "--show_images", help="shows images for each camera", action='store_true', default=False)

    # Check https://stackoverflow.com/questions/52431265/how-to-use-a-lambda-as-parameter-in-python-argparse
    def create_lambda_with_globals(s):
        return eval(s, globals())

    ap.add_argument("-ssf", "--sensor_selection_function", default=None, type=create_lambda_with_globals,
                    help='A string to be evaluated into a lambda function that receives a sensor name as input and '
                         'returns True or False to indicate if the sensor should be loaded (and used in the '
                         'optimization). The Syntax is lambda name: f(x), where f(x) is the function in python '
                         'language. Example: lambda name: name in ["left_laser", "frontal_camera"] , to load only '
                         'sensors left_laser and frontal_camera')
    ap.add_argument("-csf", "--collection_selection_function", default=None, type=create_lambda_with_globals,
                    help='A string to be evaluated into a lambda function that receives a collection name as input and '
                         'returns True or False to indicate if the collection should be loaded (and used in the '
                         'optimization). The Syntax is lambda name: f(x), where f(x) is the function in python '
                         'language. Example: lambda name: int(name) > 5 , to load only collections 6, 7, and onward.')

    args = vars(ap.parse_args())
    print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file
    # ---------------------------------------
    """ Loads a json file containing the detections"""
    f = open(args['json_file'], 'r')
    dataset_sensors = json.load(f)

    # Load images from files into memory. Images in the json file are stored in separate png files and in their place
    # a field "data_file" is saved with the path to the file. We must load the images from the disk.
    for _collection_key, collection in dataset_sensors['collections'].items():
        for _sensor_key, sensor in dataset_sensors['sensors'].items():
            if not sensor['msg_type'] == 'Image':  # nothing to do here.
                continue

            filename = os.path.dirname(args['json_file']) + '/' + collection['data'][_sensor_key]['data_file']
            collection['data'][_sensor_key]['data'] = cv2.imread(filename)

    if not args['collection_selection_function'] is None:
        deleted = []
        for _collection_key in dataset_sensors['collections'].keys():
            if not args['collection_selection_function'](_collection_key):  # use the lambda expression csf
                deleted.append(_collection_key)
                del dataset_sensors['collections'][_collection_key]
        print("Deleted collections: " + str(deleted))

    # DELETING COLLECTIONS WHERE THE CHESSBOARD WAS NOT FOUND BY BOTH CAMERAS:

    for _collection_key, collection in dataset_sensors['collections'].items():
        for _sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][_sensor_key]['detected']:
                del dataset_sensors['collections'][_collection_key]
                break
    print("\nCollections studied:\n")
    for _collection_key, collection in dataset_sensors['collections'].items():
        print(_collection_key)

    # ---------------------------------------
    # --- CREATE CHESSBOARD DATASET
    # ---------------------------------------
    # objp = np.zeros((args['chess_num_x'] * args['chess_num_y'], 3), np.float32)
    # objp[:, :2] = args['chess_size'] * np.mgrid[0:0.1:args['chess_num_x'], 0:0.1:args['chess_num_y']].T.reshape(-1, 2)
    # chessboard_evaluation_points = np.transpose(objp)
    # chessboard_evaluation_points = np.vstack(
    #     (chessboard_evaluation_points, np.ones((1, args['chess_num_x'] * args['chess_num_y']), dtype=np.float)))
    #
    #
    # print(chessboard_evaluation_points.shape)

    dataset_sensors['chessboards'] = {'chess_num_x': args['chess_num_x'], 'chess_num_y': args['chess_num_y'],
                                      'number_corners': int(args['chess_num_x'] * args['chess_num_y']),
                                      'square_size': args['chess_size'], 'collections': {}}

    dataset_chessboards = dataset_sensors['chessboards']

    factor = round(1.)
    num_pts = int((args['chess_num_x'] * factor) * (args['chess_num_y'] * factor))
    num_l_pts = int((args['chess_num_x'] * factor) * 2) + int((args['chess_num_y'] * factor) * 2) + 4
    chessboard_evaluation_points = np.zeros((4, num_pts), np.float32)
    chessboard_limit_points = np.zeros((4, num_l_pts), np.float32)
    step_x = (args['chess_num_x']) * args['chess_size'] / (args['chess_num_x'] * factor)
    step_y = (args['chess_num_y']) * args['chess_size'] / (args['chess_num_y'] * factor)

    counter = 0
    l_counter = 0

    for idx_y in range(0, int(args['chess_num_y'] * factor)):
        y = idx_y * step_y
        for idx_x in range(0, int(args['chess_num_x'] * factor)):
            x = idx_x * step_x
            chessboard_evaluation_points[0, counter] = x
            chessboard_evaluation_points[1, counter] = y
            chessboard_evaluation_points[2, counter] = 0
            chessboard_evaluation_points[3, counter] = 1
            counter += 1
            if idx_y == 0:
                chessboard_limit_points[0, l_counter] = x - step_x
                chessboard_limit_points[1, l_counter] = y - step_y
                chessboard_limit_points[2, l_counter] = 0
                chessboard_limit_points[3, l_counter] = 1
                l_counter += 1

                if idx_x == (int(args['chess_num_x'] * factor) - 1):
                    chessboard_limit_points[0, l_counter] = x
                    chessboard_limit_points[1, l_counter] = y - step_y
                    chessboard_limit_points[2, l_counter] = 0
                    chessboard_limit_points[3, l_counter] = 1
                    l_counter += 1

            if idx_x == (int(args['chess_num_x'] * factor) - 1):
                chessboard_limit_points[0, l_counter] = x + step_x
                chessboard_limit_points[1, l_counter] = y - step_y
                chessboard_limit_points[2, l_counter] = 0
                chessboard_limit_points[3, l_counter] = 1
                l_counter += 1

                if idx_y == (int(args['chess_num_y'] * factor) - 1):
                    chessboard_limit_points[0, l_counter] = x + step_x
                    chessboard_limit_points[1, l_counter] = y
                    chessboard_limit_points[2, l_counter] = 0
                    chessboard_limit_points[3, l_counter] = 1
                    l_counter += 1

    for idx_y in range(0, int(args['chess_num_y'] * factor)):
        idx_y = abs(idx_y - (int(args['chess_num_y'] * factor) - 1))
        y = idx_y * step_y

        for idx_x in range(0, int(args['chess_num_x'] * factor)):
            idx_x = abs(idx_x - (int(args['chess_num_x'] * factor) - 1))
            x = idx_x * step_x

            if idx_y == (int(args['chess_num_y'] * factor) - 1):
                chessboard_limit_points[0, l_counter] = x + step_x
                chessboard_limit_points[1, l_counter] = y + step_y
                chessboard_limit_points[2, l_counter] = 0
                chessboard_limit_points[3, l_counter] = 1
                l_counter += 1

                if idx_x == 0:
                    chessboard_limit_points[0, l_counter] = x
                    chessboard_limit_points[1, l_counter] = y + step_y
                    chessboard_limit_points[2, l_counter] = 0
                    chessboard_limit_points[3, l_counter] = 1
                    l_counter += 1

            if idx_x == 0:
                chessboard_limit_points[0, l_counter] = x - step_x
                chessboard_limit_points[1, l_counter] = y + step_y
                chessboard_limit_points[2, l_counter] = 0
                chessboard_limit_points[3, l_counter] = 1
                l_counter += 1

                if idx_y == 0:
                    chessboard_limit_points[0, l_counter] = x - step_x
                    chessboard_limit_points[1, l_counter] = y
                    chessboard_limit_points[2, l_counter] = 0
                    chessboard_limit_points[3, l_counter] = 1

    dataset_chessboards['evaluation_points'] = chessboard_evaluation_points
    dataset_chessboards['limit_points'] = chessboard_limit_points

    objp = np.zeros((args['chess_num_x'] * args['chess_num_y'], 3), np.float32)
    objp[:, :2] = args['chess_size'] * np.mgrid[0:args['chess_num_x'], 0:args['chess_num_y']].T.reshape(-1, 2)
    chessboard_points = np.transpose(objp)
    chessboard_points = np.vstack(
        (chessboard_points, np.ones((1, args['chess_num_x'] * args['chess_num_y']), dtype=np.float)))
    # print("chessboard_points")
    # print(chessboard_points)
    # print("objp")
    # print(objp)

    pts_l_chess = np.zeros((3, l_counter), np.float32)
    for i in range(0, l_counter):
        pts_l_chess[0, i] = chessboard_limit_points[0, i]
        pts_l_chess[1, i] = chessboard_limit_points[1, i]

    # homogenize points
    pts_l_chess = np.vstack((pts_l_chess, np.ones((1, pts_l_chess.shape[1]), dtype=np.float)))

    dataset_chessboard_points = {'points': chessboard_points, 'l_points': pts_l_chess}

    for _collection_key, collection in dataset_sensors['collections'].items():
        flg_detected_chessboard = False
        for _sensor_key, sensor in dataset_sensors['sensors'].items():

            if not collection['labels'][_sensor_key]['detected']:  # if chessboard not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'Image':

                image_rgb = collection['data'][_sensor_key]['data']

                mtx = np.ndarray((3, 3), dtype=np.float,
                                 buffer=np.array(sensor['camera_info']['K']))

                dist = np.ndarray((5, 1), dtype=np.float,
                                  buffer=np.array(sensor['camera_info']['D']))

                def draw(img, _corners, imgpts):
                    corner = tuple(_corners[0].ravel())
                    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
                    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
                    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)
                    return img

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                objp = np.zeros((args['chess_num_x'] * args['chess_num_y'], 3), np.float32)
                objp[:, :2] = args['chess_size'] * np.mgrid[0:args['chess_num_x'], 0:args['chess_num_y']].T.reshape(-1,
                                                                                                                    2)

                # objp = [[],[],[]]

                axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)

                gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
                # corners = np.zeros((2,48), dtype = np.int)
                ret, corners = cv2.findChessboardCorners(gray, (args['chess_num_x'], args['chess_num_y']))
                # TODO use the corners already in the json
                if ret == True:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    print("corners2.shape()")
                    print(corners2.shape)
                    # Find the rotation and translation vectors.
                    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
                    # print("First guess is:\n" + str(rvecs) + "\n" + str(tvecs))

                    # project 3D points to image plane
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                    image_rgb = draw(image_rgb, corners2, imgpts)
                    # cv2.imshow('img', image_rgb)

                    root_T_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                    # print('root_T_sensor=\n' + str(root_T_sensor) + '\n\n')

                    sensor_T_chessboard = utilities.traslationRodriguesToTransform(tvecs, rvecs)
                    # print('sensor_T_chessboard =\n ' + str(sensor_T_chessboard))

                    root_T_chessboard = np.dot(root_T_sensor, sensor_T_chessboard)
                    # print('root_T_chessboard =\n ' + str(root_T_chessboard))

                    d = {}
                    d['trans'] = list(root_T_chessboard[0:3, 3])

                    T = deepcopy(root_T_chessboard)
                    T[0:3, 3] = 0  # remove translation component from 4x4 matrix
                    d['quat'] = list(transformations.quaternion_from_matrix(T))

                    print('Creating first guess for collection ' + _collection_key + ' using sensor ' + _sensor_key)
                    dataset_chessboards['collections'][_collection_key] = d

                    flg_detected_chessboard = True
                    break  # don't search for this collection's chessboard on anymore sensors

                    # cv2.waitKey(10)

        if not flg_detected_chessboard:  # Abort when the chessboard is not detected by any camera on this collection
            raise ValueError('Collection ' + _collection_key + ' could not find chessboard.')

    # exit(0)
    # ---------------------------------------
    # --- FILTER SOME OF THE ELEMENTS LOADED, TO USE ONLY A SUBSET IN THE CALIBRATION
    # ---------------------------------------
    if not args['sensor_selection_function'] is None:
        deleted = []
        for _sensor_key in dataset_sensors['sensors'].keys():
            if not args['sensor_selection_function'](_sensor_key):  # use the lambda expression ssf
                deleted.append(_sensor_key)
                del dataset_sensors['sensors'][_sensor_key]
        print("Deleted sensors: " + str(deleted))

    print('Loaded dataset containing ' + str(len(dataset_sensors['sensors'].keys())) + ' sensors and ' + str(
        len(dataset_sensors['collections'].keys())) + ' collections.')

    # ---------------------------------------
    # --- SETUP OPTIMIZER
    # ---------------------------------------

    opt = OptimizationUtils.Optimizer()
    opt.addModelData('dataset_sensors', dataset_sensors)
    # opt.addModelData('dataset_chessboards', dataset_chessboards)
    opt.addModelData('dataset_chessboard_points', dataset_chessboard_points)

    # For the getters we only need to get one collection. Lets take the first key on the dictionary and always get that
    # transformation.
    selected_collection_key = dataset_sensors['collections'].keys()[0]

    # ------------  Sensors -----------------
    # Each sensor will have a position (tx,ty,tz) and a rotation (r1,r2,r3)

    # Add parameters related to the sensors
    translation_delta = 0.3
    for _sensor_key, sensor in dataset_sensors['sensors'].items():
        initial_values = getterSensorTranslation(dataset_sensors, sensor_key=_sensor_key,
                                                 collection_key=selected_collection_key)

        bound_max = [x + translation_delta for x in initial_values]
        bound_min = [x - translation_delta for x in initial_values]
        opt.pushParamVector(group_name='S_' + _sensor_key + '_t', data_key='dataset_sensors',
                            getter=partial(getterSensorTranslation, sensor_key=_sensor_key,
                                           collection_key=selected_collection_key),
                            setter=partial(setterSensorTranslation, sensor_key=_sensor_key),
                            suffix=['x', 'y', 'z'])
        # bound_max=bound_max, bound_min=bound_min)

        opt.pushParamVector(group_name='S_' + _sensor_key + '_r', data_key='dataset_sensors',
                            getter=partial(getterSensorRotation, sensor_key=_sensor_key,
                                           collection_key=selected_collection_key),
                            setter=partial(setterSensorRotation, sensor_key=_sensor_key),
                            suffix=['1', '2', '3'])

        if sensor['msg_type'] == 'Image':  # if sensor is a camera add intrinsics
            opt.pushParamVector(group_name='S_' + _sensor_key + '_I_', data_key='dataset_sensors',
                                getter=partial(getterCameraIntrinsics, sensor_key=_sensor_key),
                                setter=partial(setterCameraIntrinsics, sensor_key=_sensor_key),
                                suffix=['fx', 'fy', 'cx', 'cy', 'd0', 'd1', 'd2', 'd3', 'd4'])

    # ------------  Chessboard -----------------
    # Each Chessboard will have the position (tx,ty,tz) and rotation (r1,r2,r3)

    # Add translation and rotation parameters related to the Chessboards
    for _collection_key in dataset_chessboards['collections']:
        # initial_values = getterChessBoardTranslation(dataset_chessboards, collection_key)
        # bound_max = [x + translation_delta for x in initial_values]
        # bound_min = [x - translation_delta for x in initial_values]
        opt.pushParamVector(group_name='C_' + _collection_key + '_t', data_key='dataset_sensors',
                            getter=partial(getterChessBoardTranslation, collection_key=_collection_key),
                            setter=partial(setterChessBoardTranslation, collection_key=_collection_key),
                            suffix=['x', 'y', 'z'])
        # ,bound_max=bound_max, bound_min=bound_min)

        opt.pushParamVector(group_name='C_' + _collection_key + '_r', data_key='dataset_sensors',
                            getter=partial(getterChessBoardRotation, collection_key=_collection_key),
                            setter=partial(setterChessBoardRotation, collection_key=_collection_key),
                            suffix=['1', '2', '3'])

    # opt.printParameters()

    # Create a 3D plot in which the sensor poses and chessboards are drawn
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    # limit = 1.5
    ax.set_xlim3d(-1.5, 1.5), ax.set_ylim3d(-4, 1.5), ax.set_zlim3d(-.5, 1.5)
    ax.view_init(elev=27, azim=46)

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    # Each error is computed after the sensor and the chessboard of a collection. Thus, each error will be affected
    # by the parameters tx,ty,tz,r1,r2,r3 of the sensor and the chessboard

    for _sensor_key, sensor in dataset_sensors['sensors'].items():
        for _collection_key, collection in dataset_sensors['collections'].items():
            if not collection['labels'][_sensor_key]['detected']:  # if chessboard not detected by sensor in collection
                continue

            params = opt.getParamsContainingPattern('S_' + _sensor_key)  # sensor related params
            params.extend(opt.getParamsContainingPattern('C_' + _collection_key + '_'))  # chessboard related params

            if sensor['msg_type'] == 'Image':  # if sensor is a camera use four residuals
                # for idx in range(0, dataset_chessboards['number_corners']):
                for idx in range(0, 4):
                    opt.pushResidual(name=_collection_key + '_' + _sensor_key + '_' + str(idx), params=params)

            elif sensor['msg_type'] == 'LaserScan':  # if sensor is a 2D lidar add two residuals
                for idx in range(0, 4):
                    opt.pushResidual(name=_collection_key + '_' + _sensor_key + '_' + str(idx), params=params)

    # print('residuals = ' + str(opt.residuals))
    opt.printResiduals()

    # ---------------------------------------
    # --- Compute the SPARSE MATRIX
    # ---------------------------------------
    opt.computeSparseMatrix()
    # opt.printSparseMatrix()
    # exit(0)

    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    if args['view_optimization']:
        dataset_graphics = setupVisualization(dataset_sensors, args)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(dataset_graphics)
        opt.addModelData('dataset_graphics', dataset_graphics)

    opt.setVisualizationFunction(visualizationFunction, args['view_optimization'], niterations=1, figures=[fig])

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

    opt.startOptimization(
        optimization_options={'ftol': 1e-4, 'xtol': 1e-8, 'gtol': 1e-5, 'diff_step': 1e-4, 'x_scale': 'jac'})

    print('\n-----------------')
    opt.printParameters(opt.x0, text='Initial parameters')
    print('\n')
    opt.printParameters(opt.xf, text='Final parameters')

    # ---------------------------------------
    # --- Save Results
    # ---------------------------------------
    # Write json file with updated dataset_sensors
    createJSONFile('test/sensor_pose_json_v2/results/dataset_sensors_results.json', dataset_sensors)


if __name__ == "__main__":
    main()
