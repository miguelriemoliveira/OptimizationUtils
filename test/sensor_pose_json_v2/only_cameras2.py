#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS
# -------------------------------------------------------------------------------
import json

import atom_core.atom

import OptimizationUtils.utilities as utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import inv
import argparse
import cv2
import os
import math


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


def detect_pose(obj_pts, corners, camera_matrix, distortion_vector):
    ret, rvecs, tvecs = cv2.solvePnP(obj_pts, corners, camera_matrix, distortion_vector)
    if ret:
        return cv2.Rodrigues(rvecs)[0], tvecs, corners


def matlab_stereo(data_model, sensor1, sensor2, collection_key_):

    # Intrinsics matrix:

    K_1_model = np.zeros((3, 3), np.float32)
    K_2_model = np.zeros((3, 3), np.float32)

    K_1_model[0, :] = data_model['K'][sensor1][0:3]
    K_1_model[1, :] = data_model['K'][sensor1][3:6]
    K_1_model[2, :] = data_model['K'][sensor1][6:9]

    K_2_model[0, :] = data_model['K'][sensor2][0:3]
    K_2_model[1, :] = data_model['K'][sensor2][3:6]
    K_2_model[2, :] = data_model['K'][sensor2][6:9]

    print("K_" + str(sensor1) + "_model:\n")
    print(K_1_model)
    print("\nK_" + str(sensor2) + "_model:\n")
    print(K_2_model)

    # Transforms from each sensor to chessboard:

    string_sensor_1 = str(sensor1 + "_optical")
    string_sensor_2 = str(sensor2 + "_optical")

    s1_T_chessboard_model_rot = utilities.rodriguesToMatrix(
        data_model['collections'][collection_key_]['transforms'][string_sensor_1]['rodr'])
    s1_T_chessboard_model_trans = data_model['collections'][collection_key_]['transforms'][string_sensor_1]['trans']
    s2_T_chessboard_model_rot = utilities.rodriguesToMatrix(
        data_model['collections'][collection_key_]['transforms'][string_sensor_2]['rodr'])
    s2_T_chessboard_model_trans = data_model['collections'][collection_key_]['transforms'][string_sensor_2]['trans']

    s1_T_chessboard_model = np.zeros((3, 3), np.float32)
    s2_T_chessboard_model = np.zeros((3, 3), np.float32)

    for i in range(0, 2):
        s1_T_chessboard_model[:, i] = s1_T_chessboard_model_rot[:, i]
        s2_T_chessboard_model[:, i] = s2_T_chessboard_model_rot[:, i]
    for ii in range(0, 3):
        s1_T_chessboard_model[ii, 2] = s1_T_chessboard_model_trans[ii]
        s2_T_chessboard_model[ii, 2] = s2_T_chessboard_model_trans[ii]

    # print("\n\nTESTE(tf s2 to chess matlab:\n")
    # print(s1_T_chessboard_model)

    A_model = np.dot(K_2_model, s2_T_chessboard_model)
    B_model = np.dot(A_model, inv(s1_T_chessboard_model))
    homography_matrix_model = np.dot(B_model, inv(K_1_model))

    return homography_matrix_model


if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-json_opt", "--json_file_opt",
                    help="Json file containing input dataset from optimization procedure.", type=str, required=True)
    # ap.add_argument("-fs", "--first_sensor", help="First Sensor: his evaluation points will be projected to the second "
    #                                               "sensor data.", type=str, required=True)
    # ap.add_argument("-ss", "--second_sensor", help="Second Sensor: his evaluation points will be compared with the "
    #                                                "projected ones from the first sensor.", type=str, required=True)

    args = vars(ap.parse_args())
    print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file and read sensors that will be compared
    # ---------------------------------------
    """ Loads a json file containing the chessboards poses for each collection"""
    f = open(args['json_file_opt'], 'r')
    data_opt = json.load(f)

    # sensor_1 = args['first_sensor']
    # sensor_2 = args['second_sensor']
    sensor_1 = str('top_left_camera')
    sensor_2 = str('top_right_camera')
    # collection = str(args['collection_choosed'])
    input_sensors = {'first_sensor': sensor_1, 'second_sensor': sensor_2}
    input_datas = {'data_opt': data_opt}

    # Load images from files into memory. Images in the json file are stored in separate png files and in their place
    # a field "data_file" is saved with the path to the file. We must load the images from the disk.
    for collection_key, collection in data_opt['collections'].items():
        for sensor_key, sensor in data_opt['sensors'].items():
            if not sensor['msg_type'] == 'Image':  # nothing to do here.
                continue

            filename = os.path.dirname(args['json_file_opt']) + '/' + collection['data'][sensor_key]['data_file']
            collection['data'][sensor_key]['data'] = cv2.imread(filename)

    n_sensors = 0
    for sensor_key in data_opt['sensors'].keys():
        n_sensors += 1

    for i_sensor_key, i_sensor in input_sensors.items():
        a = 0
        for sensor_key, sensor in data_opt['sensors'].items():
            a += 1
            if i_sensor == sensor['_name']:
                break
            elif a == n_sensors:
                print("ERROR: " + i_sensor + " doesn't exist on the input sensors list from the optimization json file.")
                exit(0)

    n_collections = 0
    for collection_key in data_opt['collections'].items():
        n_collections += 1

    n_points = 54  # data_opt['chessboards']['number_corners']
    chess_size = 0.101  # data_opt['chessboards']['square_size']
    # ---------------------------------------
    # --- FILTER only te two cameras of interest  (this is not strictly necessary)
    # ---------------------------------------
    for data_key, data in input_datas.items():
        for sensor_key, sensor in data['sensors'].items():
            if sensor_1 == sensor['_name']:
                continue
            elif sensor_2 == sensor['_name']:
                continue
            else:
                del data['sensors'][sensor_key]

    n_cams = 0

    for sensor_key, sensor in data_opt['sensors'].items():
        if sensor['msg_type'] == "Image":
            n_cams += 1

    # Intrinsic matrixes and Distortion parameteres:
    for data_key, data in input_datas.items():
        K_1 = np.zeros((3, 3), np.float32)
        K_2 = np.zeros((3, 3), np.float32)
        D_1 = np.zeros((5, 1), np.float32)
        D_2 = np.zeros((5, 1), np.float32)

        K_1[0, :] = data['sensors'][sensor_1]['camera_info']['K'][0:3]
        K_1[1, :] = data['sensors'][sensor_1]['camera_info']['K'][3:6]
        K_1[2, :] = data['sensors'][sensor_1]['camera_info']['K'][6:9]

        D_1[:, 0] = data['sensors'][sensor_1]['camera_info']['D'][0:5]

        K_2[0, :] = data['sensors'][sensor_2]['camera_info']['K'][0:3]
        K_2[1, :] = data['sensors'][sensor_2]['camera_info']['K'][3:6]
        K_2[2, :] = data['sensors'][sensor_2]['camera_info']['K'][6:9]

        D_2[:, 0] = data['sensors'][sensor_2]['camera_info']['D'][0:5]

        if data_key == 'data_opt':
            K_1_opt = K_1
            K_2_opt = K_2
            D_1_opt = D_1
            D_2_opt = D_2

    num_x = 9  # data_opt['chessboards']['chess_num_x']
    num_y = 6  # data_opt['chessboards']['chess_num_y']

    tf_sensors_1t2 = str(sensor_1 + '-' + sensor_2)
    tf_sensors_2t1 = str(sensor_2 + '-' + sensor_1)

    points_opt1 = np.zeros((2, 0), np.float32)
    points_opt2 = np.zeros((2, 0), np.float32)

    accepted_collections = 0
    rejected_collections = []
    leg = []
    avg_error = []

    for collection_key, collection in data_opt['collections'].items():
        if not (collection['labels'][sensor_2]['detected'] and collection['labels'][sensor_1]['detected']):
            continue
        else:
            # -------------------------------------------------------------------
            # ------ Detecting chessboard
            # -------------------------------------------------------------------

            image_rgb = collection['data']['top_right_camera']['data']

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((num_x * num_y, 3), np.float32)
            objp[:, :2] = chess_size * np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

            axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)

            gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y))
            # TODO use the corners already in the json
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, K_1_opt, D_1_opt)
                # print("First guess is:\n" + str(rvecs) + "\n" + str(tvecs))

                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K_1_opt, D_1_opt)

                root_T_sensor = atom_core.atom.getAggregateTransform(data_opt['sensors']['top_right_camera']['chain'],
                                                                     collection['transforms'])

                sensor_T_chessboard = utilities.traslationRodriguesToTransform(tvecs, rvecs)

                root_T_chessboard = np.dot(root_T_sensor, sensor_T_chessboard)

                # d_1 = {}
                # d_1['trans'] = list(root_T_chessboard[0:3, 3])
                #
                # T = deepcopy(root_T_chessboard)
                # T[0:3, 3] = 0  # remove translation component from 4x4 matrix
                # d_1['quat'] = list(transformations.quaternion_from_matrix(T))

            # -------------------------------------------------------------------
            # ------ Image Points
            # -------------------------------------------------------------------
            img_points_1 = np.ones((n_points, 2), np.float32)
            img_points_2 = np.ones((n_points, 2), np.float32)

            for idx, point in enumerate(data_opt['collections'][collection_key]['labels'][sensor_2]['idxs']):
                img_points_2[idx, 0] = point['x']
                img_points_2[idx, 1] = point['y']

            for idx, point in enumerate(data_opt['collections'][collection_key]['labels'][sensor_1]['idxs']):
                img_points_1[idx, 0] = point['x']
                img_points_1[idx, 1] = point['y']

            # -------------------------------------------------------------------
            # ------ Object Points
            # -------------------------------------------------------------------
            factor = round(1.)
            object_points = np.zeros((n_points, 3), np.float32)
            step_x = num_x * chess_size / (num_x * factor)
            step_y = num_y * chess_size / (num_y * factor)

            counter = 0

            for idx_y in range(0, int(num_y * factor)):
                y = idx_y * step_y
                for idx_x in range(0, int(num_x * factor)):
                    x = idx_x * step_x
                    object_points[counter, 0] = x
                    object_points[counter, 1] = y
                    object_points[counter, 2] = 0
                    counter += 1

            # # Finding transform from sensor 1 to chessboard:
            # ret1, rvecs1, tvecs1 = cv2.solvePnP(object_points, img_points_1, K_1_opt, D_1_opt)
            # if not ret1:
            #     print ("ERROR: Chessboard wasn't found on collection" + str(collection_key))
            #     exit(0)
            # s1_T_chess_h = np.zeros((4, 4), np.float32)
            # s1_T_chess_h[3, 3] = 1
            # s1_T_chess_h[0:3, 3] = tvecs1[:, 0]
            # s1_T_chess_h[0:3, 0:3] = utilities.rodriguesToMatrix(rvecs1)
            #
            # # Finding transform from sensor 2 to chessboard:
            # ret2, rvecs2, tvecs2 = cv2.solvePnP(object_points, img_points_2, K_1_opt, D_1_opt)
            # if not ret2:
            #     print ("ERROR: Chessboard wasn't found on collection" + str(collection_key))
            #     exit(0)
            # s2_T_chess_h = np.zeros((4, 4), np.float32)
            # s2_T_chess_h[3, 3] = 1
            # s2_T_chess_h[0:3, 3] = tvecs2[:, 0]
            # s2_T_chess_h[0:3, 0:3] = utilities.rodriguesToMatrix(rvecs2)

            for data_key, data in input_datas.items():
                if data_key == 'data_opt':  # ---------------------------OPTIMIZATION----------------------------------

                    root_T_s2 = atom_core.atom.getAggregateTransform(
                        data_opt['sensors'][sensor_2]['chain'], data_opt['collections'][collection_key]['transforms'])

                    root_T_s1 = atom_core.atom.getAggregateTransform(
                        data_opt['sensors'][sensor_1]['chain'], data_opt['collections'][collection_key]['transforms'])

                    # root_T_chessboard = utilities.translationQuaternionToTransform(
                    #      data_opt['chessboards']['collections'][collection_key]['trans'],
                    #      data_opt['chessboards']['collections'][collection_key]['quat'])

                    s2_T_chess_h_opt = np.dot(inv(root_T_s2), root_T_chessboard)
                    s1_T_chess_h_opt = np.dot(inv(root_T_s1), root_T_chessboard)
                    # s1_T_s2_h_opt = np.dot(s1_T_chess_h, inv(s2_T_chess_h_opt))
                    s1_T_chess_opt = np.zeros((3, 3), np.float32)
                    s2_T_chess_opt = np.zeros((3, 3), np.float32)

                    for c in range(0, 2):
                        for l in range(0, 3):
                            s1_T_chess_opt[l, c] = s1_T_chess_h_opt[l, c]
                            s2_T_chess_opt[l, c] = s2_T_chess_h_opt[l, c]

                    s1_T_chess_opt[:, 2] = s1_T_chess_h_opt[0:3, 3]
                    s2_T_chess_opt[:, 2] = s2_T_chess_h_opt[0:3, 3]

            # -------------------------------------------------------------------
            # ------ PRINTING TFS MATRIXES
            # -------------------------------------------------------------------
            print("\n Transform s1 T chess: (OPT)")
            print(s1_T_chess_opt)

            print("\n Transform s2 T chess: (OPT)")
            print(s2_T_chess_opt)

            # print("\n s1 T s2 h: (OPT)")
            # print(s1_T_s2_h_opt)

            # -------------------------------------------------------------------
            # ------ BUILDING HOMOGRAPHY MATRIXES
            # -------------------------------------------------------------------

            A1 = np.dot(K_2_opt, s2_T_chess_opt)
            B1 = np.dot(A1, inv(s1_T_chess_opt))
            C1 = np.dot(B1, inv(K_1_opt))
            homography_matrix_opt_1 = C1

            A2 = np.dot(K_1_opt, s1_T_chess_opt)
            B2 = np.dot(A2, inv(s2_T_chess_opt))
            C2 = np.dot(B2, inv(K_2_opt))
            homography_matrix_opt_2 = C2

            # print("\n K_1: (OPT)")
            # print(K_1_opt)
            #
            # print("\n K_2: (OPT)")
            # print(K_2_opt)

            # print("\n Homography matrix: (OPT)")
            # print(homography_matrix_opt)

            # -------------------------------------------------------------------
            # ------ Points to compute the difference
            # -------------------------------------------------------------------

            idx_s1_gt = np.ones((3, n_points), np.float32)
            idx_s2_gt = np.ones((3, n_points), np.float32)

            for idx, point in enumerate(data_opt['collections'][collection_key]['labels'][sensor_2]['idxs']):
                idx_s2_gt[0, idx] = point['x']
                idx_s2_gt[1, idx] = point['y']

            for idx, point in enumerate(data_opt['collections'][collection_key]['labels'][sensor_1]['idxs']):
                idx_s1_gt[0, idx] = point['x']
                idx_s1_gt[1, idx] = point['y']

            # -------------------------------------------------------------------
            # ------ COMPARISON BETWEEN THE ERROR OF ALL CALIBRATION PROCEDURES
            # -------------------------------------------------------------------

            # OPTIMIZATION:
            s_idx_s2_proj_opt = np.dot(homography_matrix_opt_1, idx_s1_gt)
            soma_opt1 = 0
            for i in range(0, n_points):
                soma_opt1 += s_idx_s2_proj_opt[2, i]
            media_opt1 = soma_opt1 / n_points
            s_opt1 = 1 / media_opt1
            idx_s2_proj_opt = s_opt1 * s_idx_s2_proj_opt  # (*s_opt)

            # s_idx_s1_proj_opt = np.dot(homography_matrix_opt_2, idx_s2_gt)
            # soma_opt2 = 0
            # for i in range(0, n_points):
            #     soma_opt2 += s_idx_s1_proj_opt[2, i]
            # media_opt2 = soma_opt2 / n_points
            # s_opt2 = 1 / media_opt2
            # idx_s1_proj_opt = s_opt2 * s_idx_s1_proj_opt  # (*s_opt)

            # print("\n re-projected idx (without s): (OPT)")
            # print(s_idx_s2_proj_opt[:, 0:3])

            # -------------------------------------------------------------------
            # ------ ERROR!!!

            points_opt_1 = idx_s2_proj_opt[0:2, :] - idx_s2_gt[0:2, :]
            # points_opt_2 = idx_s1_proj_opt[0:2, :] - idx_s1_gt[0:2, :]

            # -------------------------------------------------------------------

            x_max_opt1 = np.amax(np.abs(points_opt_1[0, :]))
            y_max_opt1 = np.amax(np.abs(points_opt_1[1, :]))
            # x_max_opt2 = np.amax(np.abs(points_opt_2[0, :]))
            # y_max_opt2 = np.amax(np.abs(points_opt_2[1, :]))

            print '\nCOLLECTION:'
            print(collection_key)
            print ("\nx_max_opt1: " + str(x_max_opt1))
            print ("\ny_max_opt1: " + str(y_max_opt1))
            # print ("\nx_max_opt2: " + str(x_max_opt2))
            # print ("\ny_max_opt2: " + str(y_max_opt2))

            if collection_key == '1' or collection_key == '2' or collection_key == '19':
                rejected_collections.append(str(collection_key))
                continue
            else:
                # if x_max_opt1 > 8:
                #     rejected_collections.append(str(collection_key))
                #     continue
                e_tot = 0
                for idx in range(0, n_points):
                    e1 = math.sqrt((idx_s2_proj_opt[0, idx] - idx_s2_gt[0, idx]) ** 2 + (
                                idx_s2_proj_opt[1, idx] - idx_s2_gt[1, idx]) ** 2)
                    e_tot += e1

                avg_error = np.append(avg_error, e_tot / n_points)

                for n in range(0, n_points):
                    points_opt1 = np.append(points_opt1, points_opt_1[:, n:n+1], 1)

                    # points_opt2 = np.append(points_opt2, points_opt_2[:, n:n+1], 1)
            print (points_opt_1.shape)
            accepted_collections += 1
            leg.append(str(collection_key))

    total_points = n_points * accepted_collections
    print '\nTotal studied points (for each procedure): '
    print(total_points)
    print '\nRejected Collections: '
    print(rejected_collections)

    avg_error_x_opt1 = np.sum(np.abs(points_opt1[0, :]))/total_points
    avg_error_y_opt1 = np.sum(np.abs(points_opt1[1, :])) / total_points
    average_error = np.sum(avg_error)/accepted_collections
    standard_deviation = np.std(points_opt1)
    standard_deviation_ax2 = np.std(points_opt1, axis=1)

    # avg_error_x_opt2 = np.sum(np.abs(points_opt2[0, :]))/total_points
    # avg_error_y_opt2 = np.sum(np.abs(points_opt2[1, :])) / total_points

    print("\nAVERAGE ERROR (opt1 = s1 project to s2):")
    print("x = " + str(avg_error_x_opt1) + " pix ;   y = " + str(avg_error_y_opt1) + " pix")
    print("\nAVERAGE ERROR (TOTAL):")
    print("avg_error = " + str(average_error))
    print("\nSTANDARD DEVIATION (normal):")
    print("std = " + str(standard_deviation))
    print("\nSTANDARD DEVIATION (axis=1):")
    print("std = " + str(standard_deviation_ax2))

    # print("\nAVERAGE ERROR (opt2 = s2 project to s1):")
    # print("x = " + str(avg_error_x_opt2) + " pix ;   y = " + str(avg_error_y_opt2) + " pix")

    # -------------------------------------------------------------------
    # ------ SEE THE DIFFERENCE IN A SCATTER PLOT
    # -------------------------------------------------------------------
    colors = cm.tab10(np.linspace(0, 1, (points_opt1.shape[1]/n_points)))

    fig, ax = plt.subplots()
    plt.xlabel('x error (pixels)')
    plt.ylabel('y error (pixels)')

    plt.grid(True, color='k', linestyle='--', linewidth=0.1)
    # string = "Difference between the image pts and the reprojected pts"
    # plt.title(string)
    # x_max = np.amax(np.absolute([points_opt1[0, :]]))
    # y_max = np.amax(np.absolute([points_opt1[1, :]]))
    # delta = 2
    ax.set_xlim(-9, 9)
    ax.set_ylim(-25, 25)
    # print '\nCOLORS:\n'
    # print(colors)
    scatter_points = []

    for c in range(0, accepted_collections):
        l1 = plt.scatter(points_opt1[0, (c * n_points):((c + 1) * n_points)],
                         points_opt1[1, (c * n_points):((c + 1) * n_points)], marker='o', color=colors[c])
        # l2 = plt.scatter(points_opt2[0, (c * n_points):((c + 1) * n_points)],
        #                  points_opt2[1, (c * n_points):((c + 1) * n_points)], marker='v', color=[0.5, 0.5, 1])

        scatter_points.append([l1])

    # legend1 = \
    # plt.legend(scatter_points[0], ["s1 to s2"], loc="upper left", shadow=True)

    plt.legend([l[0] for l in scatter_points], leg, loc=2, title="Collections", shadow=True)
    # plt.gca().add_artist(legend1)

    plt.show()
