#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS
# -------------------------------------------------------------------------------

from copy import deepcopy
import json
import cv2
import argparse
from tf import transformations
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
    ap.add_argument("-kalibr", "--kalibr_file", help="kalibr txt file containing calibration results.", type=str, required=True)
    ap.add_argument("-csize", "--chess_size", help="Size in meters of the side of the chessboard's squares.",
                    type=float, required=True)
    # ap.add_argument("-cradius", "--chess_radius",
    #                 help="Radius in meters of the maximum side of the chessboard calibration pattern.",
    #                 type=float, required=True)
    ap.add_argument("-cnumx", "--chess_num_x", help="Chessboard's number of corners in horizontal dimension.",
                    type=int, required=True)
    ap.add_argument("-cnumy", "--chess_num_y", help="Chessboard's number of corners in vertical dimension.",
                    type=int, required=True)
    # ap.add_argument("-fs", "--first_sensor", help="First Sensor: his evaluation points will be projected to the second "
    #                                               "sensor data.", type=str, required=True)
    # ap.add_argument("-ss", "--second_sensor", help="Second Sensor: his evaluation points will be compared with the "
    #                                                "projected ones from the first sensor.", type=str, required=True)

    # Check https://stackoverflow.com/questions/52431265/how-to-use-a-lambda-as-parameter-in-python-argparse
    def create_lambda_with_globals(s):
        return eval(s, globals())

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

    sensor_1 = 'top_left_camera'
    sensor_2 = 'top_right_camera'
    chess_size = args['chess_size']
    num_x = args['chess_num_x']
    num_y = args['chess_num_y']
    kalibr_file = args['kalibr_file']
    n_points = num_x * num_y
    s1 = str(sensor_1)
    s2 = str(sensor_2)

    input_sensors = {'first_sensor': sensor_1, 'second_sensor': sensor_2}

    dataset_sensors['chessboards'] = {'chess_num_x': num_x, 'chess_num_y': num_y,
                                      'number_corners': n_points, 'square_size': chess_size}

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

    if not args['collection_selection_function'] is None:
        deleted = []
        for collection_key in dataset_sensors['collections'].keys():
            if not args['collection_selection_function'](collection_key):  # use the lambda expression csf
                deleted.append(collection_key)
                del dataset_sensors['collections'][collection_key]
        print("Deleted collections: " + str(deleted))

    # DELETING COLLECTIONS WHERE THE CHESSBOARD WAS NOT FOUND BY BOTH CAMERAS:

    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][sensor_key]['detected']:
                del dataset_sensors['collections'][collection_key]
                break
    print("\nCollections studied:\n")
    for collection_key, collection in dataset_sensors['collections'].items():
        print(collection_key)

    # -------------------------------------------------------------------
    # ------ READ INFORMATION FROM KALIBR TXT FILE
    # -------------------------------------------------------------------

    distortion_s1 = np.zeros((1, 5))
    distortion_s2 = np.zeros((1, 5))
    projection_s1 = np.zeros((1, 4))
    projection_s2 = np.zeros((1, 4))
    q_s2_to_s1 = np.zeros((1, 4))
    t_s2_to_s1 = np.zeros((1, 3))

    file = open(kalibr_file)
    lines = file.readlines()
    for n in range(0, 4):
        distortion_s1[0, n] = (float(lines[6 + n]))

    for n in range(0, 4):
        projection_s1[0, n] = (float(lines[13 + n]))

    for n in range(0, 4):
        distortion_s2[0, n] = (float(lines[24 + n]))

    for n in range(0, 4):
        projection_s2[0, n] = (float(lines[30 + n]))

    for n in range(0, 4):
        q_s2_to_s1[0, n] = (float(lines[39 + n]))

    for n in range(0, 3):
        t_s2_to_s1[0, n] = (float(lines[45 + n]))

    print('distortion_s1:\n')
    print(distortion_s1)
    print('projection_s1:\n')
    print(projection_s1)
    print('distortion_s2:\n')
    print(distortion_s2)
    print('projection_s2:\n')
    print(projection_s2)
    print ('q\n')
    print(q_s2_to_s1)
    print ('t\n')
    print(t_s2_to_s1)


    dataset_sensors['sensors'][sensor_2]['camera_info']['K'][0:3] = [projection_s2[0,0], 0, projection_s2[0,2]]
    dataset_sensors['sensors'][sensor_2]['camera_info']['K'][3:6] = [0, projection_s2[0,1], projection_s2[0,3]]
    dataset_sensors['sensors'][sensor_2]['camera_info']['K'][6:9] = [0, 0, 1]

    dataset_sensors['sensors'][sensor_1]['camera_info']['K'][0:3] = [projection_s1[0,0], 0, projection_s1[0,2]]
    dataset_sensors['sensors'][sensor_1]['camera_info']['K'][3:6] = [0, projection_s1[0,1], projection_s1[0,3]]
    dataset_sensors['sensors'][sensor_1]['camera_info']['K'][6:9] = [0, 0, 1]

    dataset_sensors['sensors'][sensor_2]['camera_info']['D'][0:5] = distortion_s2[0, :]
    dataset_sensors['sensors'][sensor_1]['camera_info']['D'][0:5] = distortion_s1[0, :]

    d1 = {}
    d1['trans'] = t_s2_to_s1[0, :]
    d1['quat'] = q_s2_to_s1[0, :]

    for collection_key, collection in dataset_sensors['collections'].items():
        collection['transforms'][str(s2) + '-' + str(s1)] = d1

    # ---------------   ------------------------
    # --- Save Results
    # ---------------------------------------
    # Write json file with updated dataset_sensors
    createJSONFile('test/sensor_pose_json_v2/results/kalibr2_calib.json', dataset_sensors)


if __name__ == "__main__":
    main()
