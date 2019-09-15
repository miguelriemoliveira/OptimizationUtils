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
    ap.add_argument("-json_opt", "--json_file_opt",
                    help="Json file containing input dataset from optimization procedure.", type=str, required=True)
    ap.add_argument("-json_calibcam", "--json_file_calibcam",
                    help="Json file containing input dataset from opencv stereo calibration.", type=str, required=True)
    ap.add_argument("-fs", "--first_sensor", help="First Sensor: his evaluation points will be projected to the second "
                                                  "sensor data.", type=str, required=True)
    ap.add_argument("-ss", "--second_sensor", help="Second Sensor: his evaluation points will be compared with the "
                                                   "projected ones from the first sensor.", type=str, required=True)

    args = vars(ap.parse_args())
    print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file and read sensors that will be compared
    # ---------------------------------------
    """ Loads a json file containing the chessboards poses for each collection"""
    f = open(args['json_file_opt'], 'r')
    ff = open(args['json_file_calibcam'], 'r')
    data_opt = json.load(f)
    data_cc = json.load(ff)

    sensor_1 = args['first_sensor']
    sensor_2 = args['second_sensor']
    # collection = str(args['collection_choosed'])
    input_sensors = {'first_sensor': sensor_1, 'second_sensor': sensor_2}

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

    n_points = data_opt['chessboards']['number_corners']
    # ---------------------------------------
    # --- FILTER only te two cameras of interest  (this is not strictly necessary)
    # ---------------------------------------
    deleted = []
    for sensor_key, sensor in data_opt['sensors'].items():
        if sensor_1 == sensor['_name']:
            continue
        elif sensor_2 == sensor['_name']:
            continue
        else:
            deleted.append(sensor['_name'])
            del data_opt['sensors'][sensor_key]
    print("\nDeleted sensors: " + str(deleted) + "\n")

    n_cams = 0

    for sensor_key, sensor in data_opt['sensors'].items():
        if sensor['msg_type'] == "Image":
            n_cams += 1

    # print("\nNumber of cameras: " + str(n_cams) + "\n")

    # Intrinsic matrixes:
    K_1_opt = np.zeros((3, 3), np.float32)
    K_2_opt = np.zeros((3, 3), np.float32)

    K_1_opt[0, :] = data_opt['sensors'][sensor_1]['camera_info']['K'][0:3]
    K_1_opt[1, :] = data_opt['sensors'][sensor_1]['camera_info']['K'][3:6]
    K_1_opt[2, :] = data_opt['sensors'][sensor_1]['camera_info']['K'][6:9]

    K_2_opt[0, :] = data_opt['sensors'][sensor_2]['camera_info']['K'][0:3]
    K_2_opt[1, :] = data_opt['sensors'][sensor_2]['camera_info']['K'][3:6]
    K_2_opt[2, :] = data_opt['sensors'][sensor_2]['camera_info']['K'][6:9]

    K_1_cc = np.zeros((3, 3), np.float32)
    K_2_cc = np.zeros((3, 3), np.float32)

    K_1_cc[0, :] = data_cc['sensors'][sensor_1]['camera_info']['K'][0:3]
    K_1_cc[1, :] = data_cc['sensors'][sensor_1]['camera_info']['K'][3:6]
    K_1_cc[2, :] = data_cc['sensors'][sensor_1]['camera_info']['K'][6:9]

    K_2_cc[0, :] = data_cc['sensors'][sensor_2]['camera_info']['K'][0:3]
    K_2_cc[1, :] = data_cc['sensors'][sensor_2]['camera_info']['K'][3:6]
    K_2_cc[2, :] = data_cc['sensors'][sensor_2]['camera_info']['K'][6:9]

    accepted_collections = 0
    leg = []

    points = np.zeros((2, 0), np.float32)
    points_model = np.zeros((2, 0), np.float32)

    for collection_key, collection in data_opt['collections'].items():
        if not (collection['labels'][sensor_2]['detected'] and collection['labels'][sensor_1]['detected']):
            continue
        else:

            # -------------------------------------------------------------------
            # ------ OPTIMIZATION
            # -------------------------------------------------------------------

            root_T_s1 = utilities.getAggregateTransform(
                data_opt['sensors'][sensor_1]['chain'], data_opt['collections']['0']['transforms'])

            root_T_s2 = utilities.getAggregateTransform(
                data_opt['sensors'][sensor_2]['chain'], data_opt['collections']['0']['transforms'])

            root_T_chessboard = utilities.translationQuaternionToTransform(
                data_opt['chessboards']['collections'][collection_key]['trans'],
                data_opt['chessboards']['collections'][collection_key]['quat'])

            s1_T_chessboard_opt_h = np.dot(inv(root_T_s1), root_T_chessboard)
            s2_T_chessboard_opt_h = np.dot(inv(root_T_s2), root_T_chessboard)

            s1_T_chessboard_opt = np.zeros((3, 3), np.float32)
            s2_T_chessboard_opt = np.zeros((3, 3), np.float32)

            for c in range(0, 2):
                for l in range(0, 3):
                    s1_T_chessboard_opt[l, c] = s1_T_chessboard_opt_h[l, c]
                    s2_T_chessboard_opt[l, c] = s2_T_chessboard_opt_h[l, c]

            s1_T_chessboard_opt[:, 2] = s1_T_chessboard_opt_h[0:3, 3]
            s2_T_chessboard_opt[:, 2] = s2_T_chessboard_opt_h[0:3, 3]

            # -------------------------------------------------------------------
            # ------ CAMERA CALIBRATE
            # -------------------------------------------------------------------

            s1_T_chessboard_cc_h = utilities.translationQuaternionToTransform(
                data_cc['chessboards']['collections'][collection_key][sensor_1]['trans'],
                data_cc['chessboards']['collections'][collection_key][sensor_1]['quat'])

            s2_T_chessboard_cc_h = utilities.translationQuaternionToTransform(
                data_cc['chessboards']['collections'][collection_key][sensor_2]['trans'],
                data_cc['chessboards']['collections'][collection_key][sensor_2]['quat'])

            s1_T_chessboard_cc = np.zeros((3, 3), np.float32)

            s2_T_chessboard_cc = np.zeros((3, 3), np.float32)

            for c in range(0, 2):
                for l in range(0, 3):
                    s1_T_chessboard_cc[l, c] = s1_T_chessboard_cc_h[l, c]
                    s2_T_chessboard_cc[l, c] = s2_T_chessboard_cc_h[l, c]

            s1_T_chessboard_cc[:, 2] = s1_T_chessboard_cc_h[0:3, 3]
            s2_T_chessboard_cc[:, 2] = s2_T_chessboard_cc_h[0:3, 3]

            # -------------------------------------------------------------------
            # ------ PRINTING TFS MATRIXES
            # -------------------------------------------------------------------
            # print("\n Transform s1 T chess: (OPT)\n")
            # print(s1_T_chessboard_opt)
            # print("\n Transform s1 T chess: (CC)\n")
            # print(s1_T_chessboard_cc)
            #
            # print("\n Transform s2 T chess: (OPT)\n")
            # print(s2_T_chessboard_opt)
            # print("\n Transform s2 T chess: (CC)\n")
            # print(s2_T_chessboard_cc)

            # -------------------------------------------------------------------
            # ------ BUILDING HOMOGRAPHY MATRIXES
            # -------------------------------------------------------------------

            A1 = np.dot(K_2_opt, s2_T_chessboard_opt)
            B1 = np.dot(A1, inv(s1_T_chessboard_opt))
            C1 = np.dot(B1, inv(K_1_opt))
            homography_matrix = C1

            A2 = np.dot(K_2_cc, s2_T_chessboard_cc)
            B2 = np.dot(A2, inv(s1_T_chessboard_cc))
            C2 = np.dot(B2, inv(K_1_cc))
            homography_matrix_model = C2

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
            # ------ COMPARISON BETWEEN THE ERROR OF BOTH CALIBRATION (our opt with p idx and model with gt idx)
            # -------------------------------------------------------------------
            s_idx_s2_proj = np.dot(homography_matrix, idx_s1_gt)

            soma = 0
            for i in range(0, n_points):
                soma += s_idx_s2_proj[2, i]
            media = soma / n_points
            s = 1 / media

            idx_s2_proj = s * s_idx_s2_proj  # (*s)

            s_idx_s2_proj_model = np.dot(homography_matrix_model, idx_s1_gt)

            soma_model = 0
            for ii in range(0, n_points):
                soma_model += s_idx_s2_proj_model[2, i]
            media_model = soma_model / n_points
            s_model = 1 / media_model

            idx_s2_proj_model = s_model * s_idx_s2_proj_model  # s_model *

            # print('\n pontos opencv sem o fator escala:')
            # print(s_idx_s2_proj_model)

            points_ = idx_s2_proj[0:2, :] - idx_s2_gt[0:2, :]
            points_model_ = idx_s2_proj_model[0:2, :] - idx_s2_gt[0:2, :]

            x_max_1 = np.amax(np.abs(points_[0, :]))
            y_max_1 = np.amax(np.abs(points_[1, :]))
            x_max_2 = np.amax(np.abs(points_model_[0, :]))
            y_max_2 = np.amax(np.abs(points_model_[1, :]))

            # if x_max_1 > 50 or y_max_1 > 50:
            #     continue

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

    print("\n AVERAGE ERROR (our optimization): \n")
    print("x = " + str(avg_error_x_1) + " pix ;   y = " + str(avg_error_y_1) + " pix")
    print("\n AVERAGE ERROR (openCV calibrate camera): \n")
    print("x = " + str(avg_error_x_2) + " pix ;   y = " + str(avg_error_y_2) + " pix")

    # -------------------------------------------------------------------
    # ------ SEE THE DIFFERENCE IN A SCATTER PLOT
    # -------------------------------------------------------------------
    colors = cm.tab20b(np.linspace(0, 1, (points.shape[1]/n_points)))

    fig, ax = plt.subplots()
    plt.xlabel('x error (pixels)')
    plt.ylabel('y error (pixels)')

    plt.grid(True, color='k', linestyle='--', linewidth=0.1)
    string = "Difference between the image pts and the reprojected pts"
    plt.title(string)
    x_max = np.amax(np.absolute(points[0, :]))
    y_max = np.amax(np.absolute(points[1, :]))
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

    legend1 = plt.legend(scatter_points[0], ["proposed approach",
                                             "OpenCV calibrate camera"], loc="upper left", shadow=True)

    plt.legend([l[0] for l in scatter_points], leg, loc=4, title="Collections", shadow=True)
    plt.gca().add_artist(legend1)

    plt.show()
