#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS
# -------------------------------------------------------------------------------
from copy import deepcopy
import json
import cv2
import argparse
from tf import transformations
import OptimizationUtils.utilities as utilities
import numpy as np

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
            if isinstance(item, np.ndarray) and key == 'data':    # to avoid saning images in the json
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
    ap.add_argument("-json", "--json_file", help="Json file containing input dataset.", type=str, required=True)
    ap.add_argument("-csize", "--chess_size", help="Size in meters of the side of the chessboard's squares.",
                    type=float, required=True)
    ap.add_argument("-cradius", "--chess_radius",
                    help="Radius in meters of the maximum side of the chessboard calibration pattern.",
                    type=float, required=True)
    ap.add_argument("-cnumx", "--chess_num_x", help="Chessboard's number of corners in horizontal dimension.",
                    type = int, required=True)
    ap.add_argument("-cnumy", "--chess_num_y", help="Chessboard's number of corners in vertical dimension.",
                    type=int, required=True)
    ap.add_argument("-fs", "--first_sensor", help="First Sensor: his evaluation points will be projected to the second "
                                                  "sensor data.", type=str, required=True)
    ap.add_argument("-ss", "--second_sensor", help="Second Sensor: his evaluation points will be compared with the "
                                                   "projected ones from the first sensor.", type=str, required=True)

    #
    # # Check https://stackoverflow.com/questions/52431265/how-to-use-a-lambda-as-parameter-in-python-argparse
    # def create_lambda_with_globals(s):
    #     return eval(s, globals())
    #
    # ap.add_argument("-ssf", "--sensor_selection_function", default=None, type=create_lambda_with_globals,
    #                 help='A string to be evaluated into a lambda function that receives a sensor name as input and '
    #                      'returns True or False to indicate if the sensor should be loaded (and used in the '
    #                      'optimization). The Syntax is lambda name: f(x), where f(x) is the function in python '
    #                      'language. Example: lambda name: name in ["left_laser", "frontal_camera"] , to load only '
    #                      'sensors left_laser and frontal_camera')
    # ap.add_argument("-csf", "--collection_selection_function", default=None, type=create_lambda_with_globals,
    #                 help='A string to be evaluated into a lambda function that receives a collection name as input and '
    #                      'returns True or False to indicate if the collection should be loaded (and used in the '
    #                      'optimization). The Syntax is lambda name: f(x), where f(x) is the function in python '
    #                      'language. Example: lambda name: int(name) > 5 , to load only collections 6, 7, and onward.')

    args = vars(ap.parse_args())
    print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file
    # ---------------------------------------
    """ Loads a json file containing the detections"""
    f = open(args['json_file'], 'r')
    dataset_sensors = json.load(f)

    sensor_1 = args['first_sensor']
    sensor_2 = args['second_sensor']
    chess_size = args['chess_size']
    num_x = args['chess_num_x']
    num_y = args['chess_num_y']
    n_points = num_x * num_y
    s1 = str(sensor_1)
    s2 = str(sensor_2)

    input_sensors = {'first_sensor': sensor_1, 'second_sensor': sensor_2}

    dataset_sensors['chessboards'] = {'chess_num_x': num_x, 'chess_num_y': num_y,
                                      'number_corners': n_points, 'square_size': chess_size, 'collections': {}}

    dataset_chessboards = dataset_sensors['chessboards']

    n_sensors = 0
    for sensor_key in dataset_sensors['sensors'].keys():
        n_sensors += 1

    for i_sensor_key, i_sensor in input_sensors.items():
        a = 0
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            a += 1
            if i_sensor == sensor['_name']:
                break
            elif a == n_sensors:
                print("ERROR: " + i_sensor + " doesn't exist on the input sensors list from the json file.")
                exit(0)

    # for collection_key, collection in dataset_sensors['collections'].items():
    #     if collection_key == '0':
    #         continue
    #     else:
    #         del dataset_sensors['collections'][collection_key]

    n_collections = 0
    for collection_key in dataset_sensors['collections'].items():
        n_collections += 1

    # ---------------------------------------
    # --- FILTER only te two cameras of interest  (this is not strictly necessary)
    # ---------------------------------------
    deleted = []
    for sensor_key, sensor in dataset_sensors['sensors'].items():
        if sensor_1 == sensor['_name']:
            continue
        elif sensor_2 == sensor['_name']:
            continue
        else:
            deleted.append(sensor['_name'])
            del dataset_sensors['sensors'][sensor_key]
    print("\nDeleted sensors: " + str(deleted) + "\n")

    # -------------------------------------------------------------------
    # ------ INTRINSICS MATRIX
    # -------------------------------------------------------------------

    K_1 = np.zeros((3, 3), np.float32)
    K_2 = np.zeros((3, 3), np.float32)

    K_1[0, :] = dataset_sensors['sensors'][sensor_1]['camera_info']['K'][0:3]
    K_1[1, :] = dataset_sensors['sensors'][sensor_1]['camera_info']['K'][3:6]
    K_1[2, :] = dataset_sensors['sensors'][sensor_1]['camera_info']['K'][6:9]

    K_2[0, :] = dataset_sensors['sensors'][sensor_2]['camera_info']['K'][0:3]
    K_2[1, :] = dataset_sensors['sensors'][sensor_2]['camera_info']['K'][3:6]
    K_2[2, :] = dataset_sensors['sensors'][sensor_2]['camera_info']['K'][6:9]

    # -------------------------------------------------------------------
    # ------ DISTORTION PARAMETERS
    # -------------------------------------------------------------------

    D_1 = np.zeros((5, 1), np.float32)
    D_2 = np.zeros((5, 1), np.float32)

    D_1[:, 0] = dataset_sensors['sensors'][sensor_1]['camera_info']['D'][0:5]

    D_2[:, 0] = dataset_sensors['sensors'][sensor_2]['camera_info']['D'][0:5]

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_1 = []  # 2d points in image plane.
    imgpoints_2 = []  # 2d points in image plane.

    for collection_key, collection in dataset_sensors['collections'].items():

        if not (collection['labels'][sensor_2]['detected'] and collection['labels'][sensor_1]['detected']):
            continue
        else:
            # -------------------------------------------------------------------
            # ------ Image Points
            # -------------------------------------------------------------------
            image_points_1 = np.ones((n_points, 2), np.float32)
            image_points_2 = np.ones((n_points, 2), np.float32)

            for idx, point in enumerate(dataset_sensors['collections'][collection_key]['labels'][sensor_2]['idxs']):
                image_points_2[idx, 0] = point['x']
                image_points_2[idx, 1] = point['y']

            for idx, point in enumerate(dataset_sensors['collections'][collection_key]['labels'][sensor_1]['idxs']):
                image_points_1[idx, 0] = point['x']
                image_points_1[idx, 1] = point['y']

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

            # If found, add object points, image points (after refining them)
            objpoints.append(object_points)
            imgpoints_1.append(image_points_1)
            imgpoints_2.append(image_points_2)

    height = dataset_sensors['sensors'][sensor_1]['height']
    width = dataset_sensors['sensors'][sensor_1]['width']
    image_size = (height, width)

    print("\n K_1: ")
    print (K_1)
    print("\n K_2: ")
    print (K_2)
    print("\n D_1: ")
    print (D_1)
    print("\n D_2: ")
    print (D_2)
    print("\n image_size: ")
    print (image_size)

    for sensor_key, sensor in dataset_sensors['sensors'].items():
        if sensor['_name'] == sensor_1:
            # flags = 0
            # flags |= cv2.CALIB_FIX_INTRINSIC
            # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_ASPECT_RATIO
            # flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # flags |= cv2.CALIB_RATIONAL_MODEL
            # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_K3
            # flags |= cv2.CALIB_FIX_K4
            # flags |= cv2.CALIB_FIX_K5

            calibcamera_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

            print(len(imgpoints_1))
            print(imgpoints_1[0].shape)
            # Calibrate the camera now using cv2 method
            ret_1, cameraMatrix_1, distCoeffs_1, rvecs_1, tvecs_1 = cv2.calibrateCamera(objpoints, imgpoints_1,
                                                                                        image_size, K_1, D_1,
                                                                                        criteria=calibcamera_criteria)
            # ,flags)

            print("\n ret_1: ")
            print (ret_1)
            print("\n M1: ")
            print (cameraMatrix_1)
            print("\n D_1: ")
            print (distCoeffs_1)
            print("\n R_1: ")
            print (rvecs_1)
            print("\n T_1: ")
            print (tvecs_1)

            dataset_sensors['sensors'][sensor_1]['camera_info']['K'][0:3] = cameraMatrix_1[0, :]
            dataset_sensors['sensors'][sensor_1]['camera_info']['K'][3:6] = cameraMatrix_1[1, :]
            dataset_sensors['sensors'][sensor_1]['camera_info']['K'][6:9] = cameraMatrix_1[2, :]

            dataset_sensors['sensors'][sensor_1]['camera_info']['D'][0:5] = distCoeffs_1[:, 0]

        elif sensor['_name'] == sensor_2:
            # flags = 0
            # flags |= cv2.CALIB_FIX_INTRINSIC
            # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_ASPECT_RATIO
            # flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # flags |= cv2.CALIB_RATIONAL_MODEL
            # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_K3
            # flags |= cv2.CALIB_FIX_K4
            # flags |= cv2.CALIB_FIX_K5

            calibcamera_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

            # Calibrate the camera now using cv2 method
            ret_2, cameraMatrix_2, distCoeffs_2, rvecs_2, tvecs_2 = cv2.calibrateCamera(objpoints, imgpoints_2,
                                                                                        image_size, K_2, D_2,
                                                                                        criteria=calibcamera_criteria)
            # ,flags)

            print("\n ret_2: ")
            print (ret_2)
            print("\n M2: ")
            print (cameraMatrix_2)
            print("\n D_2: ")
            print (distCoeffs_2)
            print("\n R_2: ")
            print (rvecs_2)
            print("\n T_2: ")
            print (tvecs_2)

            dataset_sensors['sensors'][sensor_2]['camera_info']['K'][0:3] = cameraMatrix_2[0, :]
            dataset_sensors['sensors'][sensor_2]['camera_info']['K'][3:6] = cameraMatrix_2[1, :]
            dataset_sensors['sensors'][sensor_2]['camera_info']['K'][6:9] = cameraMatrix_2[2, :]

            dataset_sensors['sensors'][sensor_2]['camera_info']['D'][0:5] = distCoeffs_2[:, 0]

    # print("\nTESTE:\n")
    # print("\nSo o primeiro rodrigues da primeiro rvecs:\n")
    # print(float(tvecs_1[0][0]))
    # print(rvecs_1[0])

    n = 0
    for collection_key, collection in dataset_sensors['collections'].items():

        dataset_chessboards['collections'][collection_key] = {s1: {}, s2: {}}

        d1 = {}
        trans1 = [float(tvecs_1[n][0]), float(tvecs_1[n][1]), float(tvecs_1[n][2])]
        d1['trans'] = trans1
        T1 = np.zeros((4, 4), np.float32)
        T1[3, 3] = 1
        T1[0:3, 0:3] = utilities.rodriguesToMatrix(rvecs_1[n])
        d1['quat'] = transformations.quaternion_from_matrix(T1).tolist()
        dataset_chessboards['collections'][collection_key][s1] = d1

        d2 = {}
        trans2 = [float(tvecs_2[n][0]), float(tvecs_2[n][1]), float(tvecs_2[n][2])]
        d2['trans'] = trans2
        T2 = np.zeros((4, 4), np.float32)
        T2[3, 3] = 1
        T2[0:3, 0:3] = utilities.rodriguesToMatrix(rvecs_2[n])
        d2['quat'] = transformations.quaternion_from_matrix(T2).tolist()
        dataset_chessboards['collections'][collection_key][sensor_2] = d2
        n += 1

    # ---------------------------------------
    # --- Save Results
    # ---------------------------------------
    # Write json file with updated dataset_sensors
    createJSONFile('test/sensor_pose_json_v2/results/opencv_calibcamera.json', dataset_sensors)


if __name__ == "__main__":
    main()