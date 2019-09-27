#!/usr/bin/env python

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import json
import cv2
from copy import deepcopy
import numpy as np


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


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


def createJSONFile(output_file, input):
    D = deepcopy(input)

    walk(D)

    print("Saving the json output file to " + str(output_file) + ", please wait, it could take a while ...")
    f = open(output_file, 'w')
    json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)  # to get only four decimal places on the json file
    print >> f, json.dumps(D, indent=2, sort_keys=True)
    f.close()
    print("Completed.")


if __name__ == "__main__":
    data_sample = {'collections': {}, 'transforms': {}}

    data_sample['transforms'] = {'car_center-frontal_camera': {'quat': [], 'trans': []},
                                 'car_center-top_right_camera': {'quat': [], 'trans': []},
                                 'frontal_camera-frontal_camera_optical':
                                     {'quat': [], 'trans': []},
                                 'top_right_camera-top_right_camera_optical':
                                     {'quat': [], 'trans': []}}

    data_sample['transforms']['car_center-frontal_camera']['quat'] = [0, 0, -0.7071, 0.7071]
    data_sample['transforms']['car_center-frontal_camera']['trans'] = [0, 0, 0.85]
    data_sample['transforms']['car_center-top_right_camera']['quat'] = [0.1405, 0.1405, -0.693, 0.693]
    data_sample['transforms']['car_center-top_right_camera']['trans'] = [0, 1.5, 1.65]
    data_sample['transforms']['frontal_camera-frontal_camera_optical']['quat'] = [-0.5, 0.5, -0.5, 0.5]
    data_sample['transforms']['frontal_camera-frontal_camera_optical']['trans'] = [0, 0, 0]
    data_sample['transforms']['top_right_camera-top_right_camera_optical']['quat'] = [-0.5, 0.5, -0.5, 0.5]
    data_sample['transforms']['top_right_camera-top_right_camera_optical']['trans'] = [0, 0, 0]

    for i in range(0, 9):
        data_sample['collections'][str(i)] = {'data': {'frontal_camera': {'data_file': "", "height": 480, "width": 640},
                                              'top_right_camera': {'data_file': "", "height": 480, "width": 640}},
                                              'labels': {'frontal_camera': {'detected': False, 'idxs': []},
                                                         'top_right_camera': {'detected': False, 'idxs': []}}}

    for c, c_key in data_sample['collections'].items():

        c_key['labels']['frontal_camera']['detected'] = False
        c_key['labels']['frontal_camera']['idxs'] = []
        c_key['labels']['top_right_camera']['detected'] = False
        c_key['labels']['top_right_camera']['idxs'] = []
        c_key['data']['frontal_camera']['data_file'] = ""
        c_key['data']['top_right_camera']['data_file'] = ""

        for sensor in ['frontal_camera', 'top_right_camera']:

            if sensor == 'frontal_camera':
                image_name = str(c) + '_Left.ppm'

            elif sensor == 'top_right_camera':
                image_name = str(c) + '_Right.ppm'

            # read image
            image = cv2.imread('/home/afonso/Desktop/calibration_sequence_I/' + image_name)

            print'\nImage name: '
            print(image_name)

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            found, corners = cv2.findChessboardCorners(image_gray, (8, 9))
            cv2.drawChessboardCorners(image, (8, 9), corners, found)  # Draw and display the corners

            if found is True:
                print'\nFound chess in collection:' + str(c) + '\n'
                # print('Found chessboard for ' + self.name)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(image_gray, corners, (8, 9), (-1, -1), criteria)
                corners2_d = []
                print(corners2_d)
                for corner in corners2:
                    corners2_d.append({'x': float(corner[0][0]), 'y': float(corner[0][1])})

                x = int(round(corners2_d[0]['x']))
                y = int(round(corners2_d[0]['y']))
                cv2.line(image, (x, y), (x, y), (0, 255, 255), 20)

                # Update the dictionary with the labels
                c_key['labels'][sensor]['detected'] = True
                c_key['labels'][sensor]['idxs'] = corners2_d
            c_key['data'][sensor]['data_file'] = image_name

    for collection, collection_key in data_sample['collections'].items():
        a = 0
        for sensor in ['frontal_camera', 'top_right_camera']:
            if not collection_key['labels'][sensor]['detected']:
                if a == 0:
                    del data_sample['collections'][collection]
                    a = 1
            else:
                continue

    # ---------------------------------------
    # --- Save Results
    # ---------------------------------------
    # Write json file with updated dataset_sensors
    createJSONFile('/tmp/dataset_sample.json', data_sample)
