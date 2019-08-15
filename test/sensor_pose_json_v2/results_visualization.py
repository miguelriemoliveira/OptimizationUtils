#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS
# -------------------------------------------------------------------------------
import json
import OptimizationUtils.utilities as utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import inv
import argparse
import cv2

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


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
    ap.add_argument("-json", "--json_file", help="Json file containing input dataset.", type=str, required=True)
    # ap.add_argument("-jm", "--json_model", help="Json file containing model dataset.", type=str, required=True)
    ap.add_argument("-fs", "--first_sensor", help="First Sensor: his evaluation points will be projected to the second "
                                                  "sensor data.", type=str, required=True)
    ap.add_argument("-ss", "--second_sensor", help="Second Sensor: his evaluation points will be compared with the "
                                                   "projected ones from the first sensor.", type=str, required=True)
    # ap.add_argument("-collection", "--collection_choosed", help="Must choose one collection to compare the points from"
    #                 " MATLAB stereo calibration with our calibration method.", type=int, required=True)

    args = vars(ap.parse_args())
    print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file and read sensors that will be compared
    # ---------------------------------------
    """ Loads a json file containing the chessboards poses for each collection"""
    f = open(args['json_file'], 'r')
    data = json.load(f)

    # """ Loads a json file containing the cameras infos obtained with a different software calibration (matlab)"""
    # fm = open(args['json_model'], 'r')         # This is the matlab results of stero calibration
    # data_model = json.load(fm)

    sensor_1 = args['first_sensor']
    sensor_2 = args['second_sensor']
    # collection = str(args['collection_choosed'])
    input_sensors = {'first_sensor': sensor_1, 'second_sensor': sensor_2}

    n_sensors = 0
    for sensor_key in data['sensors'].keys():
        n_sensors += 1

    for i_sensor_key, i_sensor in input_sensors.items():
        a = 0
        for sensor_key, sensor in data['sensors'].items():
            a += 1
            if i_sensor == sensor['_name']:
                break
            elif a == n_sensors:
                print("ERROR: " + i_sensor + " doesn't exist on the input sensors list from the json file.")
                exit(0)

    n_collections = 0
    for collection_key in data['collections'].items():
        n_collections += 1
    #
    # aa = 0
    # for collection_key, _collection in data['collections'].items():
    #     aa += 1
    #     if collection == collection_key:
    #         break
    #     elif aa == n_collections:
    #         print("ERROR: collection selected doesn't exist in the json file with the input dataset.")
    #         exit(0)

    # ---------------------------------------
    # --- FILTER only te two cameras of interest  (this is not strictly necessary)
    # ---------------------------------------
    deleted = []
    for sensor_key, sensor in data['sensors'].items():
        if sensor_1 == sensor['_name']:
            continue
        elif sensor_2 == sensor['_name']:
            continue
        else:
            deleted.append(sensor['_name'])
            del data['sensors'][sensor_key]
    print("\nDeleted sensors: " + str(deleted) + "\n")

    n_cams = 0

    for sensor_key, sensor in data['sensors'].items():
        if sensor['msg_type'] == "Image":
            n_cams += 1

    # print("\nNumber of cameras: " + str(n_cams) + "\n")

    # -------------------------------------------------------------------
    # ------ INTRINSICS MATRIX
    # -------------------------------------------------------------------

    K_1 = np.zeros((3, 3), np.float32)
    K_2 = np.zeros((3, 3), np.float32)

    K_1[0, :] = data['sensors'][sensor_1]['camera_info']['K'][0:3]
    K_1[1, :] = data['sensors'][sensor_1]['camera_info']['K'][3:6]
    K_1[2, :] = data['sensors'][sensor_1]['camera_info']['K'][6:9]

    K_2[0, :] = data['sensors'][sensor_2]['camera_info']['K'][0:3]
    K_2[1, :] = data['sensors'][sensor_2]['camera_info']['K'][3:6]
    K_2[2, :] = data['sensors'][sensor_2]['camera_info']['K'][6:9]

    print("\nIntrinsic values: \n\n " + "K_" + sensor_1 + ":\n" + str(K_1) + "\n\n" + "K_" + sensor_2 + ":\n" + str(K_2) +
          "\n")

    n_points = data['chessboards']['number_corners']

    homography_matrix_model = np.zeros((3, 3*n_collections), np.float32)
    accepted_collections = 0

    points = np.zeros((2, 0), np.float32)
    points_model = np.zeros((2, 0), np.float32)
    leg = []
    for collection_key, collection in data['collections'].items():
        if not (collection['labels'][sensor_2]['detected'] and collection['labels'][sensor_1]['detected']):
            continue
        else:

            # -------------------------------------------------------------------------------
            # ------ TRANSFORMS FROM EACH SENSOR TO CHESSBOARD OBTAINED WITH THE CALIBRATION
            # -------------------------------------------------------------------------------

            root_T_s1 = utilities.getAggregateTransform(
                data['sensors'][sensor_1]['chain'], data['collections']['0']['transforms'])

            root_T_s2 = utilities.getAggregateTransform(
                data['sensors'][sensor_2]['chain'], data['collections']['0']['transforms'])

            root_T_chessboard = utilities.translationQuaternionToTransform(
                data['chessboards']['collections'][collection_key]['trans'],
                data['chessboards']['collections'][collection_key]['quat'])

            s1_T_chessboard_h = np.dot(inv(root_T_s1), root_T_chessboard)
            s2_T_chessboard_h = np.dot(inv(root_T_s2), root_T_chessboard)

            s2_T_chessboard = np.zeros((3, 3), np.float32)
            s1_T_chessboard = np.zeros((3, 3), np.float32)

            for i in range(0, 2):
                s1_T_chessboard[:, i] = s1_T_chessboard_h[0:3, i]
                s2_T_chessboard[:, i] = s2_T_chessboard_h[0:3, i]

            s1_T_chessboard[:, 2] = s1_T_chessboard_h[0:3, 3]
            s2_T_chessboard[:, 2] = s2_T_chessboard_h[0:3, 3]
            # print("\nVERDADEIRA TF s2 T chess:\n")
            # print(s1_T_chessboard)

            A = np.dot(K_2, s2_T_chessboard)
            B = np.dot(A, inv(s1_T_chessboard))
            homography_matrix = np.dot(B, inv(K_1))

            # -------------------------------------------------------------------
            # ------ MATLAB STEREO CALIBRATION
            # -------------------------------------------------------------------
            # homography_matrix_model = matlab_stereo(data_model, sensor_1, sensor_2, collection_key)

            # -------------------------------------------------------------------
            # ------ OPENCV HOMOGRAPHY FINDER
            # -------------------------------------------------------------------
            idx_s1 = np.ones((3, n_points), np.float32)
            idx_s2 = np.ones((3, n_points), np.float32)

            for idx, point in enumerate(data['collections'][collection_key]['labels'][sensor_2]['idxs']):
                idx_s2[0, idx] = point['x']
                idx_s2[1, idx] = point['y']

            # print '\nCOLLECTION:\n'
            # print(collection_key)

            for idx, point in enumerate(data['collections'][collection_key]['labels'][sensor_1]['idxs']):
                idx_s1[0, idx] = point['x']
                idx_s1[1, idx] = point['y']

            homography_matrix_model[0:3, (int(collection_key)*3):((int(collection_key)*3)+3)], status = cv2.findHomography(
                idx_s1[0:2, :].transpose(), idx_s2[0:2, :].transpose())

            # print '\nHMM:\n'
            # print(homography_matrix_model)
            # -------------------------------------------------------------------
            # ------ COMPARISON BETWEEN THE ERROR OF BOTH CALIBRATION
            # -------------------------------------------------------------------
            s_idx_s2_proj = np.dot(homography_matrix, idx_s1)

            soma = 0
            for i in range(0, n_points):
                soma += s_idx_s2_proj[2, i]
            media = soma / n_points
            s = 1 / media

            idx_s2_proj = s * s_idx_s2_proj

            # print 'HMM:\n'
            # print(homography_matrix_model)

            s_idx_s2_proj_model = np.dot(homography_matrix_model[0:3, (int(collection_key)*3):((int(collection_key)*3)+3)], idx_s1)

            soma_model = 0
            for ii in range(0, n_points):
                soma_model += s_idx_s2_proj_model[2, i]
            media_model = soma_model / n_points
            s_model = 1 / media_model

            idx_s2_proj_model = s_model * s_idx_s2_proj_model

            # print("\nNow, lets see if this works:\n\n")
            # print("real idx (first 3 points):\n")
            # print(idx_s2[:, 0:3])
            # print("\nidx with our calib (first 3 points):\n")
            # print(idx_s2_proj[:, 0:3])
            # print("\nidx with openCV homography (first 3 points):\n")
            # print(idx_s2_proj_model[:, 0:3])

            points_ = idx_s2_proj[0:2, :] - idx_s2[0:2, :]
            points_model_ = idx_s2_proj_model[0:2, :] - idx_s2[0:2, :]

            x_max_1 = np.amax(np.abs(points_[0, :]))
            y_max_1 = np.amax(np.abs(points_[1, :]))
            x_max_2 = np.amax(np.abs(points_model_[0, :]))
            y_max_2 = np.amax(np.abs(points_model_[1, :]))

            if x_max_1 > 50 or y_max_1 > 50:
                continue

            print '\nCOLLECTION:'
            print(collection_key)
            print ("\nx_max_1: " + str(x_max_1))
            print ("\ny_max_1: " + str(y_max_1))
            print ("\nx_max_2: " + str(x_max_2))
            print ("\ny_max_2: " + str(y_max_2))

            accepted_collections += 1
            leg.append(str(collection_key))

            for n in range(0, n_points+1):
                points = np.append(points, points_[:, n:n+1], 1)
                points_model = np.append(points_model, points_model_[:, n:n+1], 1)

    total_points = n_points * accepted_collections
    print '\nTotal studied points (for each procedure): '
    print(total_points)

    avg_error_x_1 = np.sum(np.abs(points[0, :]))/total_points
    avg_error_y_1 = np.sum(np.abs(points[1, :])) / total_points
    avg_error_x_2 = np.sum(np.abs(points_model[0, :])) / total_points
    avg_error_y_2 = np.sum(np.abs(points_model[1, :])) / total_points

    print("\n AVERAGE ERROR (our calib): \n")
    print("x = " + str(avg_error_x_1) + " pix ;   y = " + str(avg_error_y_1) + " pix")
    print("\n AVERAGE ERROR (openCV): \n")
    print("x = " + str(avg_error_x_2) + " pix ;   y = " + str(avg_error_y_2) + " pix")

    # -------------------------------------------------------------------
    # ------ SEE THE DIFFERENCE IN A SCATTER PLOT
    # -------------------------------------------------------------------
    colors = cm.rainbow(np.linspace(0, 1, (points.shape[1]/n_points)))

    fig, ax = plt.subplots()
    plt.xlabel('x offset [pixels]')
    plt.ylabel('y offset [pixels]')

    plt.grid(True, color='k', linestyle='--', linewidth=0.1)
    string = "Difference between the image pts and the reprojected pts"
    plt.title(string)
    x_max = np.amax(np.absolute(points_model[0, :]))
    y_max = np.amax(np.absolute(points_model[1, :]))
    delta = 20
    ax.set_xlim(-x_max - delta, x_max + delta)
    ax.set_ylim(-y_max - delta, y_max + delta)
    # print '\nCOLORS:\n'
    # print(colors)
    scatter_points = []

    for c in range(0, accepted_collections):
        l1 = plt.scatter(points[0, (c * n_points):((c + 1) * n_points)],
                         points[1, (c * n_points):((c + 1) * n_points)], marker='o', color=colors[c])
        l2 = plt.scatter(points_model[0, (c * n_points):((c + 1) * n_points)],
                         points_model[1, (c * n_points):((c + 1) * n_points)], marker='v', color=colors[c])

        scatter_points.append([l1, l2])

    legend1 = plt.legend(scatter_points[0], ["pixels error with our calibration",
                                             "pixels error with OpenCV homography finder"], loc="upper left", shadow=True)

    plt.legend([l[0] for l in scatter_points], leg, loc=4, title="Collections", shadow=True)
    plt.gca().add_artist(legend1)

    plt.show()
