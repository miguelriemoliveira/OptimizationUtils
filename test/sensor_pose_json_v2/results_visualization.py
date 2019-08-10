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
import numpy as np
from numpy.linalg import inv
import argparse


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-json", "--json_file", help="Json file containing input dataset.", type=str, required=True)
    ap.add_argument("-jm", "--json_model", help="Json file containing model dataset.", type=str, required=True)
    ap.add_argument("-fs", "--first_sensor", help="First Sensor: his evaluation points will be projected to the second "
                                                  "sensor data.", type=str, required=True)
    ap.add_argument("-ss", "--second_sensor", help="Second Sensor: his evaluation points will be compared with the "
                                                   "projected ones from the first sensor.", type=str, required=True)
    ap.add_argument("-collection", "--collection_choosed", help="Must choose one collection to compare the points from"
                    " MATLAB stereo calibration with our calibration method.", type=int, required=True)

    args = vars(ap.parse_args())
    print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file and read sensors that will be compared
    # ---------------------------------------
    """ Loads a json file containing the chessboards poses for each collection"""
    f = open(args['json_file'], 'r')
    data = json.load(f)

    """ Loads a json file containing the cameras infos obtained with a different software calibration (matlab)"""
    fm = open(args['json_model'], 'r')         # This is the matlab results of stero calibration
    data_model = json.load(fm)

    sensor_1 = args['first_sensor']
    sensor_2 = args['second_sensor']
    collection = str(args['collection_choosed'])
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

    n_collections_chess_detected = 0
    for collection_key in data_model['collections'].items():
        n_collections_chess_detected += 1

    aa = 0
    for collection_key, _collection in data_model['collections'].items():
        aa += 1
        if collection == collection_key:
            break
        elif aa == n_collections_chess_detected:
            print("ERROR: collection selected doesn't exist in the json file with the MATLAB stereo calibration results.")
            exit(0)
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

    print("\nNumber of cameras: " + str(n_cams) + "\n")

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

    print("\nIntrinsic values: \n " + "K_" + sensor_1 + ":\n" + str(K_1) + "\n" + "K_" + sensor_2 + ":\n" + str(K_2) +
          "\n")

    n_points = data['chessboards']['number_corners']

    # -------------------------------------------------------------------------------
    # ------ TRANSFORMS FROM EACH SENSOR TO CHESSBOARD OBTAINED WITH THE CALIBRATION
    # -------------------------------------------------------------------------------

    root_T_s1 = utilities.getAggregateTransform(data['sensors'][sensor_1]['chain'], data['collections']['0']['transforms'])
    root_T_s2 = utilities.getAggregateTransform(data['sensors'][sensor_2]['chain'], data['collections']['0']['transforms'])
    root_T_chessboard = utilities.translationQuaternionToTransform(data['chessboards']['collections'][collection]['trans'],
                                                                   data['chessboards']['collections'][collection]['quat'])
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

    # Intrinsics matrix:

    K_1_model = np.zeros((3, 3), np.float32)
    K_2_model = np.zeros((3, 3), np.float32)

    K_1_model[0, :] = data_model['K'][sensor_1][0:3]
    K_1_model[1, :] = data_model['K'][sensor_1][3:6]
    K_1_model[2, :] = data_model['K'][sensor_1][6:9]

    K_2_model[0, :] = data_model['K'][sensor_2][0:3]
    K_2_model[1, :] = data_model['K'][sensor_2][3:6]
    K_2_model[2, :] = data_model['K'][sensor_2][6:9]

    print("K_" + str(sensor_1) + "_model:\n")
    print(K_1_model)
    print("\nK_" + str(sensor_2) + "_model:\n")
    print(K_2_model)

    # Transforms from each sensor to chessboard:
    string_sensor_1 = str(sensor_1 + "_optical")
    string_sensor_2 = str(sensor_2 + "_optical")

    s1_T_chessboard_model_rot = utilities.rodriguesToMatrix(
        data_model['collections'][collection]['transforms'][string_sensor_1]['rodr'])
    s1_T_chessboard_model_trans = data_model['collections'][collection]['transforms'][string_sensor_1]['trans']
    s2_T_chessboard_model_rot = utilities.rodriguesToMatrix(
        data_model['collections'][collection]['transforms'][string_sensor_2]['rodr'])
    s2_T_chessboard_model_trans = data_model['collections'][collection]['transforms'][string_sensor_2]['trans']

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

    # -------------------------------------------------------------------
    # ------ COMPARISON BETWEEN THE ERROR OF BOTH CALIBRATION
    # -------------------------------------------------------------------

    idx_s1 = np.ones((3, n_points), np.float32)
    idx_s2 = np.ones((3, n_points), np.float32)

    for idx, point in enumerate(data['collections'][collection]['labels'][sensor_2]['idxs']):
        idx_s2[0, idx] = point['x']
        idx_s2[1, idx] = point['y']

    for idx, point in enumerate(data['collections'][collection]['labels'][sensor_1]['idxs']):
        idx_s1[0, idx] = point['x']
        idx_s1[1, idx] = point['y']

    s_idx_s2_proj = np.dot(homography_matrix, idx_s1)

    soma = 0
    for i in range(0, n_points):
        soma += s_idx_s2_proj[2, i]
    media = soma / n_points
    s = 1 / media

    idx_s2_proj = s * s_idx_s2_proj

    s_idx_s2_proj_model = np.dot(homography_matrix_model, idx_s1)

    soma_model = 0
    for ii in range(0, n_points):
        soma_model += s_idx_s2_proj_model[2, i]
    media_model = soma_model / n_points
    s_model = 1 / media_model

    idx_s2_proj_model = s_model * s_idx_s2_proj_model

    print("\nNow, lets see if this works:\n\n")
    print("real idx:\n")
    print(idx_s2[:, 0:3])
    print("\nidx with our calib:\n")
    print(idx_s2_proj[:, 0:3])
    print("\nidx with matlab calib:\n")
    print(idx_s2_proj_model[:, 0:3])

    points = idx_s2_proj[0:2, :] - idx_s2[0:2, :]
    points_model = idx_s2_proj_model[0:2, :] - idx_s2[0:2, :]

    fig, ax = plt.subplots()
    plt.scatter(points[0, :], points[1, :], label='points obtained with the proposed approach', color='r')
    plt.scatter(points_model[0, :], points_model[1, :], label='points obtained with MATLAB stereo calibration', color='b')
    plt.xlabel('x offset [pixels]')
    plt.ylabel('y offset [pixels]')
    string = "Difference between the image pts and the reprojected pts for collection " + collection
    plt.title(string)
    plt.legend()
    x_max = np.amax(np.absolute(points_model[0, :]))
    y_max = np.amax(np.absolute(points_model[1, :]))
    ax.set_xlim(-x_max - 50, 500)
    ax.set_ylim(-y_max - 50, 500)
    plt.grid(True, color='k', linestyle='--', linewidth=0.1)
    plt.show()
