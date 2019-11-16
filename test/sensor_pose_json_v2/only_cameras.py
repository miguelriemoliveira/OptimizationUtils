#!/usr/bin/env python
"""
UNDER CONSTRUCTION YET"""

# -------------------------------------------------------------------------------
# --- IMPORTS
# -------------------------------------------------------------------------------
import json
import os

import OptimizationUtils.utilities as utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import inv
import argparse
import cv2
from objective_function import *
import math
from open3d import *

from getter_and_setters import *


if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-json_opt", "--json_file_opt",
                    help="Json file containing input dataset from optimization procedure.", type=str, required=True)
    # ap.add_argument("-cnum_x", "--chess_num_x", help="Number of x squares.", type=int, required=True)
    # ap.add_argument("-cnum_y", "--chess_num_y", help="Number of y squares.", type=int, required=True)
    # ap.add_argument("-csize", "--chess_size", help="Size squares.", type=float, required=True)

    args = vars(ap.parse_args())
    # print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file and read sensors that will be compared
    # ---------------------------------------
    """ Loads a json file containing the chessboards poses for each collection"""
    f = open(args['json_file_opt'], 'r')
    data_opt = json.load(f)

    n_sensors = 0
    for sensor_key in data_opt['sensors'].keys():
        n_sensors += 1

    n_cams = 0
    for sensor_key, sensor in data_opt['sensors'].items():
        if sensor['msg_type'] == 'Image':
            n_cams += 1

    n_collections = 0
    for collection_key in data_opt['collections'].items():
        n_collections += 1

    num_x = 9  # args['chess_num_x']
    num_y = 6  # args['chess_num_y']
    n_points = num_x * num_y
    chess_size = 0.101  # args['chess_size']

    # Load images from files into memory. Images in the json file are stored in separate png files and in their place
    # a field "data_file" is saved with the path to the file. We must load the images from the disk.
    # for collection_key, collection in data_opt['collections'].items():
    #     for sensor_key, sensor in data_opt['sensors'].items():
    #         if not sensor['msg_type'] == 'Image':  # nothing to do here.
    #             continue
    #
    #         filename = os.path.dirname(args['json_file_opt']) + '/' + collection['data'][sensor_key]['data_file']
    #         collection['data'][sensor_key]['data'] = cv2.imread(filename)

# Intrinsic matrixes and Distortion parameteres:
    for sensor_key, sensor in data_opt['sensors'].items():
        if sensor_key == 'top_left_camera':
            K_1 = np.zeros((3, 3), np.float32)
            D_1 = np.zeros((5, 1), np.float32)

            K_1[0, :] = sensor['camera_info']['K'][0:3]
            K_1[1, :] = sensor['camera_info']['K'][3:6]
            K_1[2, :] = sensor['camera_info']['K'][6:9]

            D_1[:, 0] = sensor['camera_info']['D'][0:5]

        if sensor_key == 'top_right_camera':
            K_2 = np.zeros((3, 3), np.float32)
            D_2 = np.zeros((5, 1), np.float32)

            K_2[0, :] = sensor['camera_info']['K'][0:3]
            K_2[1, :] = sensor['camera_info']['K'][3:6]
            K_2[2, :] = sensor['camera_info']['K'][6:9]

            D_2[:, 0] = sensor['camera_info']['D'][0:5]

    points_1 = np.zeros((2, 0), np.float32)
    points_2 = np.zeros((2, 0), np.float32)

    accepted_collections = 0
    # leg = []
    rejected_collections = []
    avg_error_2 = []
    avg_error_1 = []

    # ---------------------------------------------------------------------
    # # !!!!!!!!!!!!!!!!!!!!!!# TO REJECT: COLLECTIONS 1, 7 AND 8
    # ------------------------------------------------------------------------

    for collection_key, collection in data_opt['collections'].items():
        if not (collection['labels']['top_left_camera']['detected'] and collection['labels']['top_right_camera']['detected']):
            continue
        else:
            # image_rgb1 = collection['data']['top_left_camera']['data']
            #
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # objp = np.zeros((num_x * num_y, 3), np.float32)
            # objp[:, :2] = chess_size * np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)
            #
            # axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
            #
            # gray1 = cv2.cvtColor(image_rgb1, cv2.COLOR_BGR2GRAY)
            #
            # ret1, corners1 = cv2.findChessboardCorners(gray1, (num_x, num_y))
            #
            # # TODO use the corners already in the json
            # if ret1 == True:
            #     corners21 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            #     # Find the rotation and translation vectors.
            #     ret1, rvecs1, tvecs1 = cv2.solvePnP(objp, corners21, K_1, D_1)
            #
            #     # project 3D points to image plane
            #     imgpts1, jac1 = cv2.projectPoints(axis, rvecs1, tvecs1, K_1, D_1)
            #
            #
            #     sensor1_T_chessboard = utilities.traslationRodriguesToTransform(tvecs1, rvecs1)
            #
            #     root_T_chessboard = np.dot(root_T_sensor1, sensor1_T_chessboard)
            #

            root_T_sensor1 = utilities.getAggregateTransform(data_opt['sensors']['top_left_camera']['chain'], collection['transforms'])
            root_T_sensor2 = utilities.getAggregateTransform(data_opt['sensors']['top_right_camera']['chain'], collection['transforms'])

            root_T_chessboard = utilities.translationQuaternionToTransform(
                 data_opt['chessboards']['collections'][collection_key]['trans'],
                 data_opt['chessboards']['collections'][collection_key]['quat'])

            sensor2_T_chessboard = np.dot(inv(root_T_sensor2), root_T_chessboard)
            sensor1_T_chessboard = np.dot(inv(root_T_sensor1), root_T_chessboard)

            # -------------------------------------------------------------------
            # ------ Object Points
            # -------------------------------------------------------------------
            factor = round(1.)
            object_points = np.zeros((3, n_points), np.float32)
            step_x = num_x * chess_size / (num_x * factor)
            step_y = num_y * chess_size / (num_y * factor)

            counter = 0

            for idx_y in range(0, int(num_y * factor)):
                y = idx_y * step_y
                for idx_x in range(0, int(num_x * factor)):
                    x = idx_x * step_x
                    object_points[0, counter] = x
                    object_points[1, counter] = y
                    object_points[2, counter] = 1
                    counter += 1

            s2_T_chess_h = sensor2_T_chessboard
            s1_T_chess_h = sensor1_T_chessboard
            s1_T_chess = np.zeros((3, 3), np.float32)
            s2_T_chess = np.zeros((3, 3), np.float32)

            for c in range(0, 2):
                for l in range(0, 3):
                    s1_T_chess[l, c] = s1_T_chess_h[l, c]
                    s2_T_chess[l, c] = s2_T_chess_h[l, c]

            s1_T_chess[:, 2] = s1_T_chess_h[0:3, 3]
            s2_T_chess[:, 2] = s2_T_chess_h[0:3, 3]

            # -------------------------------------------------------------------
            # ------ BUILDING RELATION MATRIXES
            # -------------------------------------------------------------------

            A2 = np.dot(K_2, s2_T_chess)
            A1 = np.dot(K_1, s1_T_chess)

            # -------------------------------------------------------------------
            # ------ Points to compute the difference
            # -------------------------------------------------------------------

            idx_s1_gt = np.ones((3, n_points), np.float32)
            idx_s2_gt = np.ones((3, n_points), np.float32)

            for idx, point in enumerate(collection['labels']['top_right_camera']['idxs']):
                idx_s2_gt[0, idx] = point['x']
                idx_s2_gt[1, idx] = point['y']

            for idx, point in enumerate(collection['labels']['top_left_camera']['idxs']):
                idx_s1_gt[0, idx] = point['x']
                idx_s1_gt[1, idx] = point['y']

            # -------------------------------------------------------------------
            # ------ COMPARISON BETWEEN THE ERROR OF ALL CALIBRATION PROCEDURES
            # -------------------------------------------------------------------

            s_idx_s2_proj = np.dot(A2, object_points)
            s_idx_s1_proj = np.dot(A1, object_points)
            soma_1 = 0
            soma_2 = 0
            for i in range(0, n_points):
                soma_1 += s_idx_s1_proj[2, i]
                soma_2 += s_idx_s2_proj[2, i]
            media_1 = soma_1 / n_points
            media_2 = soma_2 / n_points
            s_1 = 1 / media_1
            s_2 = 1 / media_2
            idx_s2_proj = s_2 * s_idx_s2_proj  # (*s_opt)
            idx_s1_proj = s_1 * s_idx_s1_proj  # (*s_opt)

            # print("\n re-projected idx (without s): (OPT)")
            # print(s_idx_s2_proj_opt[:, 0:3])

            # -------------------------------------------------------------------
            # ------ ERROR!!!

            points_1_ = idx_s1_proj[0:2, :] - idx_s1_gt[0:2, :]
            points_2_ = idx_s2_proj[0:2, :] - idx_s2_gt[0:2, :]

            # -------------------------------------------------------------------

            x_max_1 = np.amax(np.abs(points_1_[0, :]))
            y_max_1 = np.amax(np.abs(points_1_[1, :]))
            x_max_2 = np.amax(np.abs(points_2_[0, :]))
            y_max_2 = np.amax(np.abs(points_2_[1, :]))
            x_min_1 = np.amin(np.abs(points_1_[0, :]))
            y_min_1 = np.amin(np.abs(points_1_[1, :]))
            x_min_2 = np.amin(np.abs(points_2_[0, :]))
            y_min_2 = np.amin(np.abs(points_2_[1, :]))

            print '\nCOLLECTION:'
            print(collection_key)
            print ("\nx_max_1: " + str(x_max_1))
            print ("\ny_max_1: " + str(y_max_1))
            print ("\nx_max_2: " + str(x_max_2))
            print ("\ny_max_2: " + str(y_max_2))
            print ("\nx_min_1: " + str(x_min_1))
            print ("\ny_min_1: " + str(y_min_1))
            print ("\nx_min_2: " + str(x_min_2))
            print ("\ny_min_2: " + str(y_min_2))
            #
            # tol = 40
            # if x_max_1 > tol or x_max_2 > tol or y_max_1 > tol or y_max_2 > tol:
            #     rejected_collections.append(str(collection_key))
            #     continue
            # if collection_key == '1' or collection_key == '7' or collection_key == '8':
            #     rejected_collections.append(str(collection_key))
            #     continue
            # else:

            e_tot_2 = 0
            e_tot_1 = 0
            for idx in range(0, n_points):
                e2 = math.sqrt((idx_s2_proj[0, idx] - idx_s2_gt[0, idx]) ** 2 + (idx_s2_proj[1, idx] - idx_s2_gt[1, idx]) ** 2)
                e1 = math.sqrt((idx_s1_proj[0, idx] - idx_s1_gt[0, idx]) ** 2 + (idx_s1_proj[1, idx] - idx_s1_gt[1, idx]) ** 2)
                e_tot_1 += e1
                e_tot_2 += e2

            avg_error_1 = np.append(avg_error_1, e_tot_1 / n_points)
            avg_error_2 = np.append(avg_error_2, e_tot_2 / n_points)

            accepted_collections += 1
            # leg.append(str(collection_key))

            for n in range(0, n_points):
                points_1 = np.append(points_1, points_1_[:, n:n + 1], 1)
                points_2 = np.append(points_2, points_2_[:, n:n + 1], 1)

            # print("\nOBJ POINTS:\n")
            # print(object_points)

    total_points = n_points * accepted_collections
    print '\nTotal studied collections (for each camera): '
    print(accepted_collections)
    print '\nTotal studied points (for each camera): '
    print(total_points)
    print '\nRejected Collections: '
    print(rejected_collections)

    avg_error_x_1 = np.sum(np.abs(points_1[0, :]))/total_points
    avg_error_y_1 = np.sum(np.abs(points_1[1, :])) / total_points
    avg_error_x_2 = np.sum(np.abs(points_2[0, :]))/total_points
    avg_error_y_2 = np.sum(np.abs(points_2[1, :])) / total_points
    average_error_1 = np.sum(avg_error_1) / accepted_collections
    average_error_2 = np.sum(avg_error_2) / accepted_collections
    standard_deviation_1 = np.std(points_1)
    standard_deviation_ax2_1 = np.std(points_1, axis=1)
    standard_deviation_2 = np.std(points_2)
    standard_deviation_ax2_2 = np.std(points_2, axis=1)

    print("\nAVERAGE ERROR (top left camera):")
    print("x = " + str(avg_error_x_1) + " pix ;   y = " + str(avg_error_y_1) + " pix")
    print("\nAVERAGE ERROR (top right camera):")
    print("x = " + str(avg_error_x_2) + " pix ;   y = " + str(avg_error_y_2) + " pix")
    print("\nAVERAGE ERROR (TOTAL, top left camera):")
    print("avg_error = " + str(average_error_1))
    print("\nAVERAGE ERROR (TOTAL, top right camera):")
    print("avg_error = " + str(average_error_2))
    print("\nSTANDARD DEVIATION (normal, top left camera):")
    print("std = " + str(standard_deviation_1))
    print("\nSTANDARD DEVIATION (axis=1, top left camera):")
    print("std = " + str(standard_deviation_ax2_1))
    print("\nSTANDARD DEVIATION (normal, top right camera):")
    print("std = " + str(standard_deviation_2))
    print("\nSTANDARD DEVIATION (axis=1, top right camera):")
    print("std = " + str(standard_deviation_ax2_2))

    # -------------------------------------------------------------------
    # ------ SEE THE DIFFERENCE IN A SCATTER PLOT
    # -------------------------------------------------------------------
    # colors = cm.tab20b(np.linspace(0, 1, (points_1.shape[1]/n_points)))

    fig, ax = plt.subplots()
    plt.xlabel('x error (pixels)')
    plt.ylabel('y error (pixels)')

    plt.grid(True, color='k', linestyle='--', linewidth=0.1)
    string = "Difference between the image pts and the reprojected pts"
    plt.title(string)
    x_max = np.amax(np.absolute([points_1[0, :], points_2[0, :]]))
    y_max = np.amax(np.absolute([points_1[1, :], points_2[1, :]]))
    delta = 0.5
    ax.set_xlim(-x_max - delta, x_max + delta)
    ax.set_ylim(-y_max - delta, y_max + delta)
    # print '\nCOLORS:\n'
    # print(colors)
    scatter_points = []

    for c in range(0, accepted_collections):
        l1 = plt.scatter(points_1[0, (c * n_points):((c + 1) * n_points)],
                         points_1[1, (c * n_points):((c + 1) * n_points)], marker='o', color=[1, 0.5, 0])
        l2 = plt.scatter(points_2[0, (c * n_points):((c + 1) * n_points)],
                         points_2[1, (c * n_points):((c + 1) * n_points)], marker='v', color=[0.5, 0.5, 1])

        scatter_points.append([l1, l2])

    # legend1 = \

    plt.legend(scatter_points[0], ["top_left_camera", "top_right_camera"], loc="upper left", shadow=True)

    # plt.legend([l[0] for l in scatter_points], leg, loc=4, title="Collections", shadow=True)
    # plt.gca().add_artist(legend1)

    plt.show()


