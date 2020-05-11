#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import numpy as np
from copy import deepcopy

import cv2
from tf import transformations

from OptimizationUtils import utilities


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

def createChessBoardData(args, dataset_sensors):
    """
    Creates the necessary data related to the chessboard calibration pattern
    :return: a dataset_chessboard dictionaryint((((args['chess_num_y'] * factor) - 1) * n) + 1) * (args['chess_num_x'] * factor)
    """

    # objp = np.zeros((args['chess_num_x'] * args['chess_num_y'], 3), np.float32)
    # objp[:, :2] = args['chess_size'] * np.mgrid[0:0.1:args['chess_num_x'], 0:0.1:args['chess_num_y']].T.reshape(-1, 2)
    # chessboard_evaluation_points = np.transpose(objp)
    # chessboard_evaluation_points = np.vstack(
    #     (chessboard_evaluation_points, np.ones((1, args['chess_num_x'] * args['chess_num_y']), dtype=np.float)))
    #
    #
    # print(chessboard_evaluation_points.shape)

    num_x = dataset_sensors['calibration_config']['calibration_pattern']['dimension']['x']
    num_y = dataset_sensors['calibration_config']['calibration_pattern']['dimension']['y']
    square = dataset_sensors['calibration_config']['calibration_pattern']['size']
    dataset_chessboards = {'chess_num_x': num_x, 'chess_num_y': num_y, 'number_corners': int(num_x) * int(num_y),
                           'square_size': square,
                           'collections': {}}

    step = 0.01

    # Limit Points ----------------
    # Miguel's way: Create limit_points (all points defined in 2D, no z, not homogeneous) in the chessboard ref frame.
    pts = [[], []]

    # Left vertical line
    x = -square
    y0 = -square
    y1 = num_y * square
    for y in list(np.linspace(y0, y1, num=int(abs(y1 - y0) / step), dtype=np.float)):
        pts[0].append(x), pts[1].append(y)

    # Right vertical line
    x = num_x * square
    y0 = -square
    y1 = num_y * square
    for y in list(np.linspace(y0, y1, num=int(abs(y1 - y0) / step), dtype=np.float)):
        pts[0].append(x), pts[1].append(y)

    # Top horizontal line
    # x0 = -square
    # x1 = num_x * square
    # y = -square
    # for x in list(np.linspace(x0, x1, num=int(abs(x1 - x0) / step), dtype=np.float)):
    #     pts[0].append(x), pts[1].append(y)

    # Bottom horizontal line
    # x0 = -square
    # x1 = num_x * square
    # y = num_y * square
    # for x in list(np.linspace(x0, x1, num=int(abs(x1 - x0) / step), dtype=np.float)):
    #     pts[0].append(x), pts[1].append(y)

    pts = np.array(pts, np.float)  # convert list to numpy array
    pts = np.vstack((pts, np.zeros((1, pts.shape[1]))))  # add z = 0 coordinates to all points
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # homogenize all points
    dataset_chessboards['limit_points'] = pts

    # Limit Points ----------------
    # Miguel's way: Create inner_points (all points defined in 2D, no z, not homogeneous) in the chessboard ref frame.
    pts = [[], []]

    # Vertical lines
    for x in [x * square for x in range(0, num_x)]:
        y0 = 0
        y1 = (num_y-1) * square
        for y in list(np.linspace(y0, y1, num=int(abs(y1 - y0) / step), dtype=np.float)):
            pts[0].append(x), pts[1].append(y)

    # Horizontal lines
    # for y in [y * square for y in range(0, num_y)]:
    #     x0 = 0
    #     x1 = (num_x-1) * square
    #     for x in list(np.linspace(x0, x1, num=int(abs(x1 - x0) / step), dtype=np.float)):
    #         pts[0].append(x), pts[1].append(y)

    pts = np.array(pts, np.float)  # convert list to numpy array
    pts = np.vstack((pts, np.zeros((1, pts.shape[1]))))  # add z = 0 coordinates to all points
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # homogenize all points
    dataset_chessboards['inner_points'] = pts

    # TODO limit points number should be a parsed argument
    n = 10
    factor = round(1.)
    num_pts = int((num_x * factor) * (num_y * factor))
    num_l_pts = int((num_x * factor) * 2 * n) + int((num_y * factor) * 2 * n) + (4 * n)
    num_i_pts = int(((num_x * factor) - 1) * (n - 1)) * (num_y * factor) + int(
        ((num_y * factor) - 1) * (n - 1)) * (num_x * factor) + num_pts
    chessboard_evaluation_points = np.zeros((4, num_pts), np.float32)
    chessboard_limit_points = np.zeros((4, int(num_l_pts)), np.float32)
    chessboard_inner_points = np.zeros((4, int(num_i_pts)), np.float32)
    step_x = num_x * square / (num_x * factor)
    step_y = num_y * square / (num_y * factor)

    counter = 0
    l_counter = 0
    i_counter = 0
    # TODO @afonsocastro should put this more synthesized
    for idx_y in range(0, int(num_y * factor)):
        y = idx_y * step_y
        for idx_x in range(0, int(num_x * factor)):
            x = idx_x * step_x
            chessboard_evaluation_points[0, counter] = x
            chessboard_evaluation_points[1, counter] = y
            chessboard_evaluation_points[2, counter] = 0
            chessboard_evaluation_points[3, counter] = 1
            counter += 1
            # if idx_x != (int(num_x * factor) - 1):
            #     for i in range(0, n):
            #         chessboard_inner_points[0, i_counter] = x + (i * (step_x / n))
            #         chessboard_inner_points[1, i_counter] = y
            #         chessboard_inner_points[2, i_counter] = 0
            #         chessboard_inner_points[3, i_counter] = 1
            #         i_counter += 1
            # else:
            #     chessboard_inner_points[0, i_counter] = x
            #     chessboard_inner_points[1, i_counter] = y
            #     chessboard_inner_points[2, i_counter] = 0
            #     chessboard_inner_points[3, i_counter] = 1
            #     i_counter += 1

            if idx_y != (int(num_y * factor) - 1):
                for i in range(1, n):
                    chessboard_inner_points[0, i_counter] = x
                    chessboard_inner_points[1, i_counter] = y + (i * (step_y / n))
                    chessboard_inner_points[2, i_counter] = 0
                    chessboard_inner_points[3, i_counter] = 1
                    i_counter += 1

            # if idx_y == 0:
            #     for i in range(0, n):
            #         chessboard_limit_points[0, l_counter] = x - ((n - i) * (step_x / n))
            #         chessboard_limit_points[1, l_counter] = y - step_y
            #         chessboard_limit_points[2, l_counter] = 0
            #         chessboard_limit_points[3, l_counter] = 1
            #         l_counter += 1
            #
            #     if idx_x == (int(num_x * factor) - 1):
            #         for i in range(n, 0, -1):
            #             chessboard_limit_points[0, l_counter] = x + ((n - i) * (step_x / n))
            #             chessboard_limit_points[1, l_counter] = y - step_y
            #             chessboard_limit_points[2, l_counter] = 0
            #             chessboard_limit_points[3, l_counter] = 1
            #             l_counter += 1

            if idx_x == (int(num_x * factor) - 1):
                for i in range(0, n):
                    chessboard_limit_points[0, l_counter] = x + step_x
                    chessboard_limit_points[1, l_counter] = y - ((n - i) * (step_y / n))
                    chessboard_limit_points[2, l_counter] = 0
                    chessboard_limit_points[3, l_counter] = 1
                    l_counter += 1

                if idx_y == (int(num_y * factor) - 1):
                    for i in range(n, 0, -1):
                        chessboard_limit_points[0, l_counter] = x + step_x
                        chessboard_limit_points[1, l_counter] = y + ((n - i) * (step_y / n))
                        chessboard_limit_points[2, l_counter] = 0
                        chessboard_limit_points[3, l_counter] = 1
                        l_counter += 1

    for idx_y in range(0, int(num_y * factor)):
        idx_y = abs(idx_y - (int(num_y * factor) - 1))
        y = idx_y * step_y

        for idx_x in range(0, int(num_x * factor)):
            idx_x = abs(idx_x - (int(num_x * factor) - 1))
            x = idx_x * step_x

            # if idx_y == (int(num_y * factor) - 1):
            #     for i in range(0, n):
            #         chessboard_limit_points[0, l_counter] = x + ((n - i) * (step_x / n))
            #         chessboard_limit_points[1, l_counter] = y + step_y
            #         chessboard_limit_points[2, l_counter] = 0
            #         chessboard_limit_points[3, l_counter] = 1
            #         l_counter += 1
            #
            #     if idx_x == 0:
            #         for i in range(n, 0, -1):
            #             chessboard_limit_points[0, l_counter] = x - ((n - i) * (step_x / n))
            #             chessboard_limit_points[1, l_counter] = y + step_y
            #             chessboard_limit_points[2, l_counter] = 0
            #             chessboard_limit_points[3, l_counter] = 1
            #             l_counter += 1

            if idx_x == 0:
                for i in range(0, n):
                    chessboard_limit_points[0, l_counter] = x - step_x
                    chessboard_limit_points[1, l_counter] = y + ((n - i) * (step_y / n))
                    chessboard_limit_points[2, l_counter] = 0
                    chessboard_limit_points[3, l_counter] = 1
                    l_counter += 1

                if idx_y == 0:
                    for i in range(n, 0, -1):
                        chessboard_limit_points[0, l_counter] = x - step_x
                        chessboard_limit_points[1, l_counter] = y - ((n - i) * (step_y / n))
                        chessboard_limit_points[2, l_counter] = 0
                        chessboard_limit_points[3, l_counter] = 1
                        l_counter += 1

    dataset_chessboards['evaluation_points'] = chessboard_evaluation_points
    # dataset_chessboards['limit_points'] = chessboard_limit_points
    # dataset_chessboards['inner_points'] = chessboard_inner_points

    # print('chessboard_limit_points.shape=' + str(chessboard_limit_points.shape))
    # exit(0)

    objp = np.zeros((num_x * num_y, 3), np.float32)
    objp[:, :2] = square * np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)
    chessboard_points = np.transpose(objp)
    chessboard_points = np.vstack(
        (chessboard_points, np.ones((1, num_x * num_y), dtype=np.float)))

    pts_l_chess = np.zeros((3, l_counter), np.float32)
    for i in range(0, l_counter):
        pts_l_chess[0, i] = chessboard_limit_points[0, i]
        pts_l_chess[1, i] = chessboard_limit_points[1, i]

    pts_i_chess = np.zeros((3, i_counter), np.float32)
    for i in range(0, i_counter):
        pts_i_chess[0, i] = chessboard_inner_points[0, i]
        pts_i_chess[1, i] = chessboard_inner_points[1, i]

    # homogenize points
    pts_l_chess = np.vstack((pts_l_chess, np.ones((1, pts_l_chess.shape[1]), dtype=np.float)))

    dataset_chessboard_points = {'points': chessboard_points, 'l_points': pts_l_chess, 'i_points': pts_i_chess}

    for collection_key, collection in dataset_sensors['collections'].items():
        flg_detected_chessboard = False
        for sensor_key, sensor in dataset_sensors['sensors'].items():

            if not collection['labels'][sensor_key]['detected']:  # if chessboard not detected by sensor in collection
                print('Collection ' + str(collection_key) + ': Chessboard not detected by sensor ' + str(sensor_key))
                continue

            # change accordingly to the first camera to give chessboard first poses
            if (sensor['msg_type'] == 'Image'):

                K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(sensor['camera_info']['K']))
                D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(sensor['camera_info']['D']))

                # TODO should we not read these from the dictionary?
                objp = np.zeros((num_x * num_y, 3), np.float32)
                objp[:, :2] = square * np.mgrid[0:num_x, 0:num_y].T.reshape(-1,2)
                # Build a numpy array with the chessboard corners
                corners = np.zeros((len(collection['labels'][sensor_key]['idxs']), 1, 2), dtype=np.float)
                for idx, point in enumerate(collection['labels'][sensor_key]['idxs']):
                    corners[idx, 0, 0] = point['x']
                    corners[idx, 0, 1] = point['y']

                # Find pose of the camera w.r.t the chessboard
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners, K, D)

                # Compute the pose of he chessboard w.r.t the base_link
                root_T_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                sensor_T_chessboard = utilities.traslationRodriguesToTransform(tvecs, rvecs)
                root_T_chessboard = np.dot(root_T_sensor, sensor_T_chessboard)
                T = deepcopy(root_T_chessboard)
                T[0:3, 3] = 0  # remove translation component from 4x4 matrix

                print('Creating first guess for collection ' + collection_key + ' using sensor ' + sensor_key)
                dataset_chessboards['collections'][str(collection_key)] = {
                    'trans': list(root_T_chessboard[0:3, 3]),
                    'quat': list(transformations.quaternion_from_matrix(T))}

                flg_detected_chessboard = True
                break  # don't search for this collection's chessboard on anymore sensors

        if not flg_detected_chessboard:  # Abort when the chessboard is not detected by any camera on this collection
            raise ValueError('Collection ' + collection_key + ' could not find chessboard.')

    return dataset_chessboards, dataset_chessboard_points
