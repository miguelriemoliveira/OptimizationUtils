#!/usr/bin/env python

import sys
import os.path
import argparse
import json
import copy
import numpy as np
import cv2

from OptimizationUtils import utilities
from OptimizationUtils.OptimizationUtils import Optimizer

from OptimizationUtils.tf import TFTree, Transform

# build the chessboard corners
objp = np.zeros((9*7, 3), np.float32)
objp[:, :2] = 0.019 * np.mgrid[0:9, 0:7].T.reshape(-1,2)

# Homegeneous chess points
hcp = [x + [1] for x in objp.tolist()]


def setter(obj, name, value):
    obj[name] = value


def objectiveFunction(models):

    parameters = models['parameters']
    collections = models['collections']
    sensors = models['sensors']

    residual = []

    for name, sensor in sensors.items():
        for idx, collection in enumerate(collections):
            if sensor['msg_type'] == 'Image':
                if not collection['labels'][name]['detected']:
                    continue

                tree = collection['tf_tree']

                param = copy.deepcopy(parameters[name]['trans'])
                param.extend(parameters[name]['rot'])
                tree.add_transform(sensor['calibration_parent'], sensor['calibration_child'], Transform.from_position_euler(*param))

                param = copy.deepcopy(parameters['ee_link-chessboard_link']['trans'])
                param.extend(parameters['ee_link-chessboard_link']['rot'])
                tree.add_transform('ee_link', 'chessboard_link', Transform.from_position_euler(*param))

                # sensor to root transformation
                rTs = tree.lookup_transform(sensor['camera_info']['header']['frame_id'], 'base_link').matrix
                # chessboard to root transformation
                rTc = tree.lookup_transform('chessboard_link', 'base_link').matrix


                # TEST MIGUEL - ALTERNATIVE to tf
                rTs_miguel = utilities.getTransform(sensor['camera_info']['header']['frame_id'], 'base_link', collection['transforms'])
                print('rts = ' + str(rTs))
                print('rts_miguel = ' + str(rTs_miguel))


                # convert chessboard corners from pixels to sensor coordinates.
                K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(sensor['camera_info']['K']))
                D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(sensor['camera_info']['D']))

                corners = np.zeros((len(collection['labels'][name]['idxs']),1,2), dtype=np.float)
                for idx, point in enumerate(collection['labels'][name]['idxs']):
                    corners[idx,0,0] = point['x']
                    corners[idx,0,1] = point['y']

                ret, rvecs, tvecs = cv2.solvePnP(objp, corners, K, D)
                sTc = utilities.traslationRodriguesToTransform(tvecs, rvecs)

                rTs = np.dot(rTs, sTc)

                p = np.stack((hcp[0], hcp[9], hcp[6*9], hcp[6*9+8])).T
                error = np.apply_along_axis(np.linalg.norm, 0,
                                            np.dot(rTs, p) - np.dot(rTc, p))

                residual.extend( error.tolist() )

    return residual


def load_data(jsonfile, sensor_filter=None, collection_filter=None):
    """Load data from a JSON file.

    Parameters
    ----------
    jsonfile: str
        Path to the JSON file that contains the data.
    sensor_filter: callable, optional
        Used to filter the data by de name of the sensor.
        For example: `lambda name: name == 'sensor_name'` will remove
        all information and data associated with this a sensor which
        name is 'sensor_name'.
    collection_filter: callable, optional
        Used to filter the data by collection number.
        For example: `lambda idx: idx != 0 ` will remove the collection with
        index 0.

    Returns
    -------
    sensors: dict
        Sensors metadata.
    collections: list of dicts
        List of collections ordered by index.
    """

    try:
        with open(jsonfile, 'r') as f:
            dataset = json.load(f)
    except IOError as e:
        print(str(e))
        sys.exit(1)

    # Sensors metadata.
    # Contains information such as their links, topics, intrinsics (if camera), etc..
    sensors = dataset['sensors']

    # A collection is a list of data.
    # The capture order is maintained in the list.
    collections = [x for _, x in sorted(dataset['collections'].items())]

    # Filter the sensors and collection by sensor name
    if sensor_filter is not None:
        sensors = dict(filter(sensor_filter, sensors.items()))
        for c in collections:
            c['data'] = dict(filter(sensor_filter, c['data'].items()))
            c['labels'] = dict(filter(sensor_filter, c['data'].items()))

    if collection_filter is not None:
        collections = [x for idx, x in enumerate(collections) if not collection_filter(idx)]

    for collection in collections:

        # Build a transformation tree for this collection
        tree = TFTree()
        for _, tf in collection['transforms'].items():
            param = copy.deepcopy(tf['trans'])
            param.extend(tf['quat'])
            tree.add_transform(tf['parent'], tf['child'], Transform(*param))

        collection['tf_tree'] = tree

        # Handle chessboard data
        for sensor_name in collection['data'].keys():
            if sensors[sensor_name]['msg_type'] != 'Image':
                continue  # we are only interested in images, remember?

            # Image data is not stored in the JSON file for practical reasons. Mainly, too much data.
            # Instead, the image is stored in a compressed format and its name is part of the collection.
            filename = os.path.dirname(jsonfile) + '/' + collection['data'][sensor_name]['data_file']
            collection['data'][sensor_name]['data'] = cv2.imread(filename)

    return sensors, collections


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-json", "--json_file", help="JSON file containing input dataset.", type=str, required=True)

    args = vars(ap.parse_args())

    # Sensor information and calibration data is saved in a json file.
    # The data is organized as a collection of data.
    sensors, collections = load_data(args['json_file'])

    # Setup optimizer
    opt = Optimizer()

    parameters = {}
    opt.addDataModel('parameters', parameters)
    opt.addDataModel('collections', collections)
    opt.addDataModel('sensors', sensors)

    # In this scenario we want to optimize the pose of the camera(s) and the link
    # that joins the end-effector to the chessboard.
    for name, sensor in sensors.items():
        t = collections[0]['transforms']['base_link-camera_link']['trans']
        r = collections[0]['transforms']['base_link-camera_link']['quat']

        xform = Transform(t[0], t[1], t[2], r[0], r[1], r[2], r[3])
        r = xform.euler

        parameters[name] = {'trans': t, 'rot': r}
        opt.pushParamVector(group_name='S_' + name + '_trans', data_key='parameters',
                            getter=lambda m: m[name]['trans'],
                            setter=lambda m,v: setter(m[name], 'trans', v),
                            suffix=['x', 'y', 'z'])
        opt.pushParamVector(group_name='S_' + name + '_rot', data_key='parameters',
                            getter=lambda m: m[name]['rot'],
                            setter=lambda m,v: setter(m[name],'rot', v),
                            suffix=['1', '2', '3'])


    # end-effector to chessboard first guess.
    t = collections[0]['transforms']['ee_link-chessboard_link']['trans']
    r = collections[0]['transforms']['ee_link-chessboard_link']['quat']

    xform = Transform(t[0], t[1], t[2], r[0], r[1], r[2], r[3])
    r = xform.euler

    ee_name = 'ee_link-chessboard_link'
    parameters[ee_name] = {'trans': t, 'rot': r}
    opt.pushParamVector(group_name='L_' + ee_name + '_trans', data_key='parameters',
                            getter=lambda m: m[ee_name]['trans'],
                            setter=lambda m,v: setter(m[ee_name],'trans', v),
                            suffix=['x', 'y', 'z'])

    opt.pushParamVector(group_name='L_' + name + '_rot', data_key='parameters',
                            getter=lambda m: m[ee_name]['rot'],
                            setter=lambda m,v: setter(m[ee_name],'rot' , v),
                            suffix=['1', '2', '3'])

    # Declare residuals
    for name, sensor in sensors.items():
        for idx, collection in enumerate(collections):
            if sensor['msg_type'] == 'Image':
                if not collection['labels'][name]['detected']:
                    continue

                params = opt.getParamsContainingPattern('S_' + name)
                params.extend(opt.getParamsContainingPattern('L_'))
                for i in range(0, 4):
                    opt.pushResidual(name=str(idx) + '_' + name + '_' + str(i), params=params)

    opt.computeSparseMatrix()
    opt.setObjectiveFunction(objectiveFunction)

    options = {'ftol': 1e-4, 'xtol': 1e-8, 'gtol': 1e-5, 'diff_step': 1e-4, 'x_scale': 'jac'}
    opt.startOptimization(options)

    print parameters


if __name__ == '__main__':
    main()
