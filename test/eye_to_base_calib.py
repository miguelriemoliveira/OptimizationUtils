#!/usr/bin/env python

import sys
import os.path
import argparse
import json
import numpy as np
import cv2

from functools import partial

from OptimizationUtils import utilities
from OptimizationUtils.OptimizationUtils import Optimizer

from OptimizationUtils.tf import TFTree, Transform


def getPose(model, name):
    # A pose is represented by a translation vector and a quaternion.
    # However, the rotation's optimization is done on a rotation vector (see Rodrigues).
    xform = Transform(*model[name])
    return model[name][0:3] + utilities.matrixToRodrigues(xform.rotation_matrix).tolist()


def setPose(model, pose, name):
    # A pose is represented by a translation vector and a quaternion.
    # However, the optimization of the rotation is done on a rotation vector (see Rodrigues).
    mat = utilities.traslationRodriguesToTransform(pose[0:3], pose[3:])
    xform = Transform.from_matrix(mat)
    model[name] = xform.position_quaternion


def printOriginTag(name, pose):
    xform = Transform(*pose)
    t = [str(x) for x in list(xform.position)]
    r = [str(x) for x in list(xform.euler)]
    print('<!-- {} -->'.format(name))
    print('<origin xyz="{}" rpy="{}" />'.format(" ".join(t), " ".join(r)))


def objectiveFunction(models):

    parameters = models['parameters']
    collections = models['collections']
    sensors = models['sensors']
    pattern = models['pattern']
    config = models['config']

    residual = []

    for idx, collection in enumerate(collections):

        tree = collection['tf_tree']

        # chessboard to root transformation
        tree.add_transform(pattern['parent_link'], pattern['link'], Transform(*parameters['pattern']))
        rTc = tree.lookup_transform(pattern['link'], config['world_link']).matrix

        for sensor_name, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            xform = Transform(*parameters[sensor_name])
            tree.add_transform(sensors[sensor_name]['calibration_parent'], sensors[sensor_name]['calibration_child'], xform)

<<<<<<< HEAD

                # TEST MIGUEL - ALTERNATIVE to tf
                rTs_miguel = utilities.getTransform(sensor['camera_info']['header']['frame_id'], 'base_link', collection['transforms'])
                print('rts = ' + str(rTs))
                print('rts_miguel = ' + str(rTs_miguel))


                # convert chessboard corners from pixels to sensor coordinates.
                K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(sensor['camera_info']['K']))
                D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(sensor['camera_info']['D']))
=======
            # sensor to root transformation
            rTs = tree.lookup_transform(sensors[sensor_name]['camera_info']['header']['frame_id'], config['world_link']).matrix
>>>>>>> Use Rodrigues in optimization. Code refactoring.

            # convert chessboard corners from pixels to sensor coordinates.
            K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(sensors[sensor_name]['camera_info']['K']))
            D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(sensors[sensor_name]['camera_info']['D']))

            corners = np.zeros((len(labels['idxs']), 1, 2), dtype=np.float)
            for idx, point in enumerate(labels['idxs']):
                corners[idx, 0, 0] = point['x']
                corners[idx, 0, 1] = point['y']

            ret, rvecs, tvecs = cv2.solvePnP(pattern['grid'], corners, K, D)
            sTc = utilities.traslationRodriguesToTransform(tvecs, rvecs)

            rTs = np.dot(rTs, sTc)

            w = pattern['dimension'][0]
            h = pattern['dimension'][1] - 1

            hcp = pattern['hgrid']
            p = np.stack((hcp[0], hcp[w-1], hcp[w*h], hcp[h*w + w - 1])).T

            error = np.apply_along_axis(np.linalg.norm, 0,
                                        np.dot(rTs, p) - np.dot(rTc, p))

            residual.extend(error.tolist())

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
    config: dict
        Calibration configuration.
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

    # A collection is a list of data. The capture order is maintained in the list.
    collections = [x for _, x in sorted(dataset['collections'].items())]

    # Filter the sensors and collection by sensor name
    if sensor_filter is not None:
        sensors = dict(filter(sensor_filter, sensors.items()))
        for c in collections:
            c['data'] = dict(filter(sensor_filter, c['data'].items()))
            c['labels'] = dict(filter(sensor_filter, c['data'].items()))

    if collection_filter is not None:
        collections = [x for idx, x in enumerate(collections) if not collection_filter(idx)]

    # Image data is not stored in the JSON file for practical reasons. Mainly, too much data.
    # Instead, the image is stored in a compressed format and its name is in the collection.
    for collection in collections:
        for sensor_name in collection['data'].keys():
            if sensors[sensor_name]['msg_type'] != 'Image':
                continue  # we are only interested in images, remember?

            filename = os.path.dirname(jsonfile) + '/' + collection['data'][sensor_name]['data_file']
            collection['data'][sensor_name]['data'] = cv2.imread(filename)

    return sensors, collections, dataset['calibration_config']


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-json", "--json_file", help="JSON file containing input dataset.", type=str, required=True)

    args = vars(ap.parse_args())

    # Sensor information and calibration data is saved in a json file.
    # The dataset is organized as a collection of data.
    sensors, collections, config = load_data(args['json_file'])

    # The best way to keep track of all transformation pairs is to a have a
    # transformation tree similar to what ROS offers (github.com/safijari/tiny_tf).
    for collection in collections:
        tree = TFTree()
        for _, tf in collection['transforms'].items():
            param = tf['trans'] + tf['quat']
            tree.add_transform(tf['parent'], tf['child'], Transform(*param))

        collection['tf_tree'] = tree

    # Setup optimizer
    opt = Optimizer()

    parameters = {}
<<<<<<< HEAD
    opt.addDataModel('parameters', parameters)
    opt.addDataModel('collections', collections)
    opt.addDataModel('sensors', sensors)
=======

    # should the models be unified? (eurico)
    opt.addModelData('parameters', parameters)
    opt.addModelData('collections', collections)
    opt.addModelData('sensors', sensors)
    opt.addModelData('config', config)
>>>>>>> Use Rodrigues in optimization. Code refactoring.

    # The pose of several sensors are explicit optimization parameters.
    # These poses are the the same for all collections.
    # We will save these parameters in a global model.
    for name, sensor in sensors.items():
        # All collections have the same pose for the given sensor.
        key = utilities.generateKey(sensor['calibration_parent'], sensor['calibration_child'])
        t = collections[0]['transforms'][key]['trans']
        r = collections[0]['transforms'][key]['quat']
        parameters[name] = t + r

        opt.pushParamVector(group_name='S_' + name, data_key='parameters',
                            getter=partial(getPose, name=name),
                            setter=partial(setPose, name=name),
                            suffix=['x', 'y', 'z', 'rx', 'ry', 'rz'])

    pattern = config['calibration_pattern']
    # cache the chessboard grid
    size = (pattern['dimension'][0], pattern['dimension'][1])

    grid = np.zeros((size[0]*size[1], 3), np.float32)
    grid[:, :2] = pattern['size'] * np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

    pattern['grid'] = grid
    pattern['hgrid'] = [x + [1] for x in grid.tolist()]  # homogeneous coordinates

    opt.addModelData('pattern', pattern)

    # Pattern pose, for now we assume fixed pose
    parameters['pattern'] = Transform.from_position_euler(*pattern['origin']).position_quaternion
    opt.pushParamVector(group_name='L_pattern', data_key='parameters',
                        getter=partial(getPose, name='pattern'),
                        setter=partial(setPose, name='pattern'),
                        suffix=['x', 'y', 'z', 'rx', 'ry', 'rz'])

    # Declare residuals
    for idx, collection in enumerate(collections):
        for key, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            params = opt.getParamsContainingPattern('S_' + name)
            params.extend(opt.getParamsContainingPattern('L_'))
            for i in range(0, 4):
                opt.pushResidual(name=str(idx) + '_' + name + '_' + str(i), params=params)

    opt.computeSparseMatrix()
    opt.setObjectiveFunction(objectiveFunction)

    options = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-4, 'x_scale': 'jac'}
    opt.startOptimization(options)

    print('')
    for name in sensors.keys():
        printOriginTag(name, parameters[name])

    printOriginTag('pattern', parameters['pattern'])


if __name__ == '__main__':
    main()
