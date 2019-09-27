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
    # TODO chess size should be also in json files with the chess info

    chess_size = 0.1054
    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-json_opt", "--json_file_opt",
                    help="Json file containing input dataset from optimization procedure.", type=str, required=True)
    ap.add_argument("-json_stereo", "--json_file_stereo",
                    help="Json file containing input dataset from opencv stereo calibration.", type=str, required=True)
    ap.add_argument("-json_calibcam", "--json_file_calibcam",
                    help="Json file containing input dataset from opencv calibrate camera func.", type=str, required=True)
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
    fff = open(args['json_file_stereo'], 'r')
    data_opt = json.load(f)
    data_cc = json.load(ff)
    data_stereo = json.load(fff)

    sensor_1 = args['first_sensor']
    sensor_2 = args['second_sensor']
    # collection = str(args['collection_choosed'])
    input_sensors = {'first_sensor': sensor_1, 'second_sensor': sensor_2}
    input_datas = {'data_opt': data_opt, 'data_stereo': data_stereo, 'data_cc': data_cc}

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
        elif data_key == 'data_stereo':
            K_1_stereo = K_1
            K_2_stereo = K_2
            D_1_stereo = D_1
            D_2_stereo = D_2
        elif data_key == 'data_cc':
            K_1_cc = K_1
            K_2_cc = K_2
            D_1_cc = D_1
            D_2_cc = D_2

    num_x = data_opt['chessboards']['chess_num_x']
    num_y = data_opt['chessboards']['chess_num_y']

    tf_sensors = str(sensor_1 + '-' + sensor_2)

    for collection_key, collection in data_opt['collections'].items():
        if not (collection['labels'][sensor_2]['detected'] and collection['labels'][sensor_1]['detected']):
            continue
        else:
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

            for data_key, data in input_datas.items():
                if data_key == 'data_opt':

                    # Finding transform from sensor 1 to chessboard:
                    ret, rvecs, tvecs = cv2.solvePnP(object_points, img_points_1, K_1_opt, D_1_opt)
                    if not ret:
                        print ("ERROR: Chessboard wasn't found on collection" + str(collection_key))
                        exit(0)
                    s1_T_chess_h_opt = np.zeros((4, 4), np.float32)
                    s1_T_chess_h_opt[3, 3] = 1
                    s1_T_chess_h_opt[0:3, 3] = tvecs[:, 0]
                    s1_T_chess_h_opt[0:3, 0:3] = utilities.rodriguesToMatrix(rvecs)

                    root_T_s2 = utilities.getAggregateTransform(
                        data_opt['sensors'][sensor_2]['chain'], data_opt['transforms'])

                    root_T_chessboard = utilities.translationQuaternionToTransform(
                        data_opt['chessboards']['collections'][collection_key]['trans'],
                        data_opt['chessboards']['collections'][collection_key]['quat'])

                    s2_T_chess_h_opt = np.dot(inv(root_T_s2), root_T_chessboard)

                    s1_T_chess_opt = np.zeros((3, 3), np.float32)
                    s2_T_chess_opt = np.zeros((3, 3), np.float32)

                    for c in range(0, 2):
                        for l in range(0, 3):
                            s1_T_chess_opt[l, c] = s1_T_chess_h_opt[l, c]
                            s2_T_chess_opt[l, c] = s2_T_chess_h_opt[l, c]

                    s1_T_chess_opt[:, 2] = s1_T_chess_h_opt[0:3, 3]
                    s2_T_chess_opt[:, 2] = s2_T_chess_h_opt[0:3, 3]

                elif data_key == 'data_stereo':

                    # Finding transform from sensor 1 to chessboard:
                    ret, rvecs, tvecs = cv2.solvePnP(object_points, img_points_1, K_1_stereo, D_1_stereo)
                    if not ret:
                        print ("ERROR: Chessboard wasn't found on collection" + str(collection_key))
                        exit(0)
                    s1_T_chess_h_stereo = np.zeros((4, 4), np.float32)
                    s1_T_chess_h_stereo[3, 3] = 1
                    s1_T_chess_h_stereo[0:3, 3] = tvecs[:, 0]
                    s1_T_chess_h_stereo[0:3, 0:3] = utilities.rodriguesToMatrix(rvecs)

                    s1_T_s2_h = root_T_chessboard = utilities.translationQuaternionToTransform(
                        data_stereo['transforms'][tf_sensors]['trans'],
                        data_stereo['transforms'][tf_sensors]['quat'])

                    s2_T_chess_h_stereo = np.dot(inv(s1_T_s2_h), s1_T_chess_h_stereo)

                    s1_T_chess_stereo = np.zeros((3, 3), np.float32)
                    s2_T_chess_stereo = np.zeros((3, 3), np.float32)

                    for c in range(0, 2):
                        for l in range(0, 3):
                            s1_T_chess_stereo[l, c] = s1_T_chess_h_stereo[l, c]
                            s2_T_chess_stereo[l, c] = s2_T_chess_h_stereo[l, c]

                    s1_T_chess_stereo[:, 2] = s1_T_chess_h_stereo[0:3, 3]
                    s2_T_chess_stereo[:, 2] = s2_T_chess_h_stereo[0:3, 3]

                elif data_key == 'data_cc':

                    # Finding transform from sensor 1 to chessboard:
                    ret, rvecs, tvecs = cv2.solvePnP(object_points, img_points_1, K_1_cc, D_1_cc)
                    if not ret:
                        print ("ERROR: Chessboard wasn't found on collection" + str(collection_key))
                        exit(0)
                    s1_T_chess_h_cc = np.zeros((4, 4), np.float32)
                    s1_T_chess_h_cc[3, 3] = 1
                    s1_T_chess_h_cc[0:3, 3] = tvecs[:, 0]
                    s1_T_chess_h_cc[0:3, 0:3] = utilities.rodriguesToMatrix(rvecs)

                    s2_T_chess_h_cc = utilities.translationQuaternionToTransform(
                        data_cc['chessboards']['collections'][collection_key][sensor_2]['trans'],
                        data_cc['chessboards']['collections'][collection_key][sensor_2]['quat'])

                    s1_T_chess_cc = np.zeros((3, 3), np.float32)

                    s2_T_chess_cc = np.zeros((3, 3), np.float32)

                    for c in range(0, 2):
                        for l in range(0, 3):
                            s1_T_chess_cc[l, c] = s1_T_chess_h_cc[l, c]
                            s2_T_chess_cc[l, c] = s2_T_chess_h_cc[l, c]

                    s1_T_chess_cc[:, 2] = s1_T_chess_h_cc[0:3, 3]
                    s2_T_chess_cc[:, 2] = s2_T_chess_h_cc[0:3, 3]

            # -------------------------------------------------------------------
            # ------ PRINTING TFS MATRIXES
            # -------------------------------------------------------------------
            print("\n Transform s1 T chess: (OPT)")
            print(s1_T_chess_opt)
            print("\n Transform s1 T chess: (STEREO)")
            print(s1_T_chess_stereo)
            print("\n Transform s1 T chess: (CC)")
            print(s1_T_chess_cc)

            print("\n Transform s2 T chess: (OPT)")
            print(s2_T_chess_opt)
            print("\n Transform s2 T chess: (STEREO)")
            print(s2_T_chess_stereo)
            print("\n Transform s2 T chess: (CC)")
            print(s2_T_chess_cc)

            # -------------------------------------------------------------------
            # ------ BUILDING HOMOGRAPHY MATRIXES
            # -------------------------------------------------------------------

            A1 = np.dot(K_2_opt, s2_T_chess_opt)
            B1 = np.dot(A1, inv(s1_T_chess_opt))
            C1 = np.dot(B1, inv(K_1_opt))
            homography_matrix_opt = C1

            A2 = np.dot(K_2_cc, s2_T_chess_cc)
            B2 = np.dot(A2, inv(s1_T_chess_cc))
            C2 = np.dot(B2, inv(K_1_cc))
            homography_matrix_cc = C2

            A3 = np.dot(K_2_stereo, s2_T_chess_stereo)
            B3 = np.dot(A3, inv(s1_T_chess_stereo))
            C3 = np.dot(B3, inv(K_1_stereo))
            homography_matrix_stereo = C3

            accepted_collections = 0
            leg = []

            points_opt = np.zeros((2, 0), np.float32)
            points_stereo = np.zeros((2, 0), np.float32)
            points_cc = np.zeros((2, 0), np.float32)

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
            s_idx_s2_proj_opt = np.dot(homography_matrix_opt, idx_s1_gt)
            soma_opt = 0
            for i in range(0, n_points):
                soma_opt += s_idx_s2_proj_opt[2, i]
            media_opt = soma_opt / n_points
            s_opt = 1 / media_opt
            idx_s2_proj_opt = s_opt * s_idx_s2_proj_opt  # (*s_opt)

            # STEREO CALIBRATION:
            s_idx_s2_proj_stereo = np.dot(homography_matrix_stereo, idx_s1_gt)
            soma_stereo = 0
            for ii in range(0, n_points):
                soma_stereo += s_idx_s2_proj_stereo[2, ii]
            media_stereo = soma_stereo / n_points
            s_stereo = 1 / media_stereo
            idx_s2_proj_stereo = s_stereo * s_idx_s2_proj_stereo  # s_stereo *

            # CAMERA CALIBRATION:
            s_idx_s2_proj_cc = np.dot(homography_matrix_cc, idx_s1_gt)
            soma_cc = 0
            for iii in range(0, n_points):
                soma_cc += s_idx_s2_proj_stereo[2, iii]
            media_cc = soma_cc / n_points
            s_cc = 1 / media_cc
            idx_s2_proj_cc = s_cc * s_idx_s2_proj_cc  # s_cc *

            # -------------------------------------------------------------------
            # ------ ERROR!!!

            points_opt_ = idx_s2_proj_opt[0:2, :] - idx_s2_gt[0:2, :]
            points_stereo_ = idx_s2_proj_stereo[0:2, :] - idx_s2_gt[0:2, :]
            points_cc_ = idx_s2_proj_cc[0:2, :] - idx_s2_gt[0:2, :]
            # -------------------------------------------------------------------

            x_max_opt = np.amax(np.abs(points_opt_[0, :]))
            y_max_opt = np.amax(np.abs(points_opt_[1, :]))
            x_max_stereo = np.amax(np.abs(points_stereo_[0, :]))
            y_max_stereo = np.amax(np.abs(points_stereo_[1, :]))
            x_max_cc = np.amax(np.abs(points_cc_[0, :]))
            y_max_cc = np.amax(np.abs(points_cc_[1, :]))

            print '\nCOLLECTION:'
            print(collection_key)
            print ("\nx_max_opt: " + str(x_max_opt))
            print ("\ny_max_opt: " + str(y_max_opt))
            print ("\nx_max_stereo: " + str(x_max_stereo))
            print ("\ny_max_stereo: " + str(y_max_stereo))
            print ("\nx_max_cc: " + str(x_max_cc))
            print ("\ny_max_cc: " + str(y_max_cc))

            accepted_collections += 1
            leg.append(str(collection_key))

            for n in range(0, n_points+1):
                points_opt = np.append(points_opt, points_opt_[:, n:n+1], 1)
                points_stereo = np.append(points_stereo, points_stereo_[:, n:n+1], 1)
                points_cc = np.append(points_cc, points_cc_[:, n:n+1], 1)

    total_points = n_points * accepted_collections
    print '\nTotal studied points (for each procedure): '
    print(total_points)

    avg_error_x_opt = np.sum(np.abs(points_opt[0, :]))/total_points
    avg_error_y_opt = np.sum(np.abs(points_opt[1, :])) / total_points
    avg_error_x_stereo = np.sum(np.abs(points_stereo[0, :])) / total_points
    avg_error_y_stereo = np.sum(np.abs(points_stereo[1, :])) / total_points
    avg_error_x_cc = np.sum(np.abs(points_cc[0, :])) / total_points
    avg_error_y_cc = np.sum(np.abs(points_cc[1, :])) / total_points

    print("\n AVERAGE ERROR (our optimization): \n")
    print("x = " + str(avg_error_x_opt) + " pix ;   y = " + str(avg_error_y_opt) + " pix")
    print("\n AVERAGE ERROR (openCV stereo calibration): \n")
    print("x = " + str(avg_error_x_stereo) + " pix ;   y = " + str(avg_error_y_stereo) + " pix")
    print("\n AVERAGE ERROR (openCV calibrate camera): \n")
    print("x = " + str(avg_error_x_cc) + " pix ;   y = " + str(avg_error_y_cc) + " pix")

    # -------------------------------------------------------------------
    # ------ SEE THE DIFFERENCE IN A SCATTER PLOT
    # -------------------------------------------------------------------
    colors = cm.tab20b(np.linspace(0, 1, (points_opt.shape[1]/n_points)))

    fig, ax = plt.subplots()
    plt.xlabel('x error (pixels)')
    plt.ylabel('y error (pixels)')

    plt.grid(True, color='k', linestyle='--', linewidth=0.1)
    string = "Difference between the image pts and the reprojected pts"
    plt.title(string)
    x_max = np.amax(np.absolute(points_opt[0, :]))
    y_max = np.amax(np.absolute(points_opt[1, :]))
    delta = 20
    ax.set_xlim(-x_max - delta, x_max + delta)
    ax.set_ylim(-y_max - delta, y_max + delta)
    # print '\nCOLORS:\n'
    # print(colors)
    scatter_points = []

    for c in range(0, accepted_collections):
        l1 = plt.scatter(points_opt[0, (c * n_points):((c + 1) * n_points)],
                         points_opt[1, (c * n_points):((c + 1) * n_points)], marker='o', color=colors[c])
        l2 = plt.scatter(points_stereo[0, (c * n_points):((c + 1) * n_points)],
                         points_stereo[1, (c * n_points):((c + 1) * n_points)], marker='v', color=colors[c])
        l3 = plt.scatter(points_cc[0, (c * n_points):((c + 1) * n_points)],
                         points_cc[1, (c * n_points):((c + 1) * n_points)], marker='P', color=colors[c])

        scatter_points.append([l1, l2, l3])

    legend1 = plt.legend(scatter_points[0], ["proposed approach", "OpenCV stereo calibration",
                                             "OpenCV calibrate camera"], loc="upper left", shadow=True)

    plt.legend([l[0] for l in scatter_points], leg, loc=4, title="Collections", shadow=True)
    plt.gca().add_artist(legend1)

    plt.show()
