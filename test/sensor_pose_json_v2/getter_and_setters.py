#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import numpy as np
from scipy.spatial import distance
from tf import transformations

import OptimizationUtils.utilities as utilities


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

# ------------  Sensors -----------------
# Each sensor will have a position (tx,ty,tz) and a rotation (r1,r2,r3)

def getterSensorTranslation(data, sensor_key, collection_key):
    calibration_parent = data['sensors'][sensor_key]['calibration_parent']
    calibration_child = data['sensors'][sensor_key]['calibration_child']
    transform_key = calibration_parent + '-' + calibration_child
    # We use collection selected_collection and assume they are all the same
    return data['collections'][collection_key]['transforms'][transform_key]['trans']


def setterSensorTranslation(data, value, sensor_key):
    assert len(value) == 3, "value must be a list with length 3."

    calibration_parent = data['sensors'][sensor_key]['calibration_parent']
    calibration_child = data['sensors'][sensor_key]['calibration_child']
    transform_key = calibration_parent + '-' + calibration_child

    for _collection_key in data['collections']:
        data['collections'][_collection_key]['transforms'][transform_key]['trans'] = value


def getterSensorRotation(data, sensor_key, collection_key):
    calibration_parent = data['sensors'][sensor_key]['calibration_parent']
    calibration_child = data['sensors'][sensor_key]['calibration_child']
    transform_key = calibration_parent + '-' + calibration_child

    # We use collection selected_collection and assume they are all the same
    quat = data['collections'][collection_key]['transforms'][transform_key]['quat']
    hmatrix = transformations.quaternion_matrix(quat)
    matrix = hmatrix[0:3, 0:3]

    return utilities.matrixToRodrigues(matrix)


def setterSensorRotation(data, value, sensor_key):
    assert len(value) == 3, "value must be a list with length 3."

    matrix = utilities.rodriguesToMatrix(value)
    hmatrix = np.identity(4)
    hmatrix[0:3, 0:3] = matrix
    quat = transformations.quaternion_from_matrix(hmatrix)

    calibration_parent = data['sensors'][sensor_key]['calibration_parent']
    calibration_child = data['sensors'][sensor_key]['calibration_child']
    transform_key = calibration_parent + '-' + calibration_child

    for _collection_key in data['collections']:
        data['collections'][_collection_key]['transforms'][transform_key]['quat'] = quat


def getterCameraPMatrix(data, sensor_key):
    fx_p = data['sensors'][sensor_key]['camera_info']['P'][0]
    fy_p = data['sensors'][sensor_key]['camera_info']['P'][5]
    cx_p = data['sensors'][sensor_key]['camera_info']['P'][2]
    cy_p = data['sensors'][sensor_key]['camera_info']['P'][6]
    intrinsics = [fx_p, fy_p, cx_p, cy_p]
    return intrinsics


def setterCameraPMatrix(data, value, sensor_key):
    assert len(value) == 4, "value must be a list with length 4."
    data['sensors'][sensor_key]['camera_info']['P'][0] = value[0]  # fx_p
    data['sensors'][sensor_key]['camera_info']['P'][5] = value[1]  # fy_p
    data['sensors'][sensor_key]['camera_info']['P'][2] = value[2]  # cx_p
    data['sensors'][sensor_key]['camera_info']['P'][6] = value[3]  # cy_p


def getterCameraIntrinsics(data, sensor_key):
    fx = data['sensors'][sensor_key]['camera_info']['K'][0]
    fy = data['sensors'][sensor_key]['camera_info']['K'][4]
    cx = data['sensors'][sensor_key]['camera_info']['K'][2]
    cy = data['sensors'][sensor_key]['camera_info']['K'][5]
    D = data['sensors'][sensor_key]['camera_info']['D']
    intrinsics = [fx, fy, cx, cy]
    intrinsics.extend(D)
    return intrinsics


def setterCameraIntrinsics(data, value, sensor_key):
    assert len(value) == 9, "value must be a list with length 9."
    data['sensors'][sensor_key]['camera_info']['K'][0] = value[0]
    data['sensors'][sensor_key]['camera_info']['K'][4] = value[1]
    data['sensors'][sensor_key]['camera_info']['K'][2] = value[2]
    data['sensors'][sensor_key]['camera_info']['K'][5] = value[3]
    data['sensors'][sensor_key]['camera_info']['D'] = value[4:]


# ------------  Chessboards -----------------
# Each sensor will have a position (tx,ty,tz) and a rotation (r1,r2,r3)

def getterChessBoardTranslation(data, collection_key):
    return data['chessboards']['collections'][collection_key]['trans']


def setterChessBoardTranslation(data, value, collection_key):
    assert len(value) == 3, "value must be a list with length 3."
    data['chessboards']['collections'][collection_key]['trans'] = value


def getterChessBoardRotation(data, collection_key):
    quat = data['chessboards']['collections'][collection_key]['quat']
    hmatrix = transformations.quaternion_matrix(quat)
    matrix = hmatrix[0:3, 0:3]
    return utilities.matrixToRodrigues(matrix)


def setterChessBoardRotation(data, value, collection_key):
    assert len(value) == 3, "value must be a list with length 3."

    matrix = utilities.rodriguesToMatrix(value)
    hmatrix = np.identity(4).astype(np.float)
    hmatrix[0:3, 0:3] = matrix
    quat = transformations.quaternion_from_matrix(hmatrix)
    data['chessboards']['collections'][collection_key]['quat'] = quat
