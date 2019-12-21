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


def getPose(model, name, idx = None):
    # A pose is represented by a translation vector and a quaternion.
    # However, the rotation's optimization is done on a rotation vector (see Rodrigues).
    if idx is None:
        xform = Transform(*model[name])
        return model[name][0:3] + utilities.matrixToRodrigues(xform.rotation_matrix).tolist()
    else:
        xform = Transform(*model[name][idx])
        return model[name][idx][0:3] + utilities.matrixToRodrigues(xform.rotation_matrix).tolist()


def setPose(model, pose, name, idx = None):
    # A pose is represented by a translation vector and a quaternion.
    # However, the optimization of the rotation is done on a rotation vector (see Rodrigues).
    mat = utilities.traslationRodriguesToTransform(pose[0:3], pose[3:])
    xform = Transform.from_matrix(mat)
    if idx is None:
        model[name] = xform.position_quaternion
    else:
        model[name][idx] = xform.position_quaternion


def printOriginTag(name, pose):
    xform = Transform(*pose)
    t = [str(x) for x in list(xform.position)]
    r = [str(x) for x in list(xform.euler)]
    print('<!-- {} -->'.format(name))
    print('<origin xyz="{}" rpy="{}"/>'.format(" ".join(t), " ".join(r)))


def getPatternPose(tf_tree, root_link, camera, pattern, labels):
    # sensor to root transformation
    rTs = tf_tree.lookup_transform(camera['camera_info']['header']['frame_id'], root_link).matrix

    # convert chessboard corners from pixels to sensor coordinates.
    K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(camera['camera_info']['K']))
    D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(camera['camera_info']['D']))
    width = camera['camera_info']['width']
    height = camera['camera_info']['height']

    corners = np.zeros( (2, len(labels['idxs'])), dtype=np.float32)
    for idx, point in enumerate(labels['idxs']):
        corners[0, idx] = point['x']
        corners[1, idx] = point['y']

    _, rvecs, tvecs = cv2.solvePnP(np.array(pattern['grid'][:3,:].T), corners.T.reshape(-1,1,2), K, D)
    sTc = utilities.traslationRodriguesToTransform(tvecs, rvecs)

    return Transform.from_matrix(np.dot(rTs, sTc)).position_quaternion


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
        if pattern['fixed']:
            tree.add_transform(pattern['parent_link'], pattern['link'], Transform(*parameters['pattern']))
            rTc = tree.lookup_transform(pattern['link'], config['world_link']).matrix
        elif parameters['pattern'][idx] is not None:
            tree.add_transform(pattern['parent_link'], pattern['link'] + str(idx), Transform(*parameters['pattern'][idx]))
            rTc = tree.lookup_transform(pattern['link'] + str(idx), config['world_link']).matrix

        # TODO MIGUEL's tests. To delete after solving the problem

        # lets print the transform pool
        print("all the transforms:\n'" + str(collection['transforms'].keys()))

        parent = 'ee_link'
        child = 'chessboard_link'

        # Eurico's approach (first the child, then the parent) TODO Eurico, please confirm
        T1a = tree.lookup_transform(child, parent).matrix
        # T1a = tree.lookup_transform(parent, child).matrix
        print('\nT1a (using Euricos approach) =\n' + str(T1a))

        # Miguel's approach (first the parent, then the child)
        T1b = utilities.getTransform(parent, child, collection['transforms'])
        print('\nT1b (using Miguels approach)=\n' + str(T1b))

        tranform_key = parent + '-' + child
        print("\nFrom collection['transforms'] =\n" + str(collection['transforms'][tranform_key]))
        trans = collection['transforms'][tranform_key]['trans']
        quat = collection['transforms'][tranform_key]['quat']

        T1c = utilities.translationQuaternionToTransform(trans, quat)
        print('\nT1c (extracted directly from dictionary)=\n' + str(T1c))

        exit(0)
        # --------------------------------------------------------

        for sensor_name, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            xform = Transform(*parameters[sensor_name])
            tree.add_transform(sensors[sensor_name]['calibration_parent'], sensors[sensor_name]['calibration_child'],
                               xform)

            # sensor to root transformation
            rTs = tree.lookup_transform(sensors[sensor_name]['camera_info']['header']['frame_id'],
                                        config['world_link']).matrix
            sTr = np.linalg.inv(rTs)

            # chess to camera transformation
            sTc = np.dot(sTr, rTc)

            # convert chessboard corners from pixels to sensor coordinates.
            K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(sensors[sensor_name]['camera_info']['K']))
            D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(sensors[sensor_name]['camera_info']['D']))
            width = sensors[sensor_name]['camera_info']['width']
            height = sensors[sensor_name]['camera_info']['height']

            corners = np.zeros( (2, len(labels['idxs'])), dtype=np.float32)
            for idx, point in enumerate(labels['idxs']):
                corners[0, idx] = point['x']
                corners[1, idx] = point['y']

            if config['ef'] == 'projection':
                projected, _, _ = utilities.projectToCamera(K, D, width, height, np.dot(sTc, pattern['grid']))
                error = np.apply_along_axis(np.linalg.norm, 0, projected - corners)
            else:
                ret, rvecs, tvecs = cv2.solvePnP(np.array(pattern['grid'][:3,:].T), corners.T.reshape(-1,1,2), K, D)
                sTc = utilities.traslationRodriguesToTransform(tvecs, rvecs)
                rTs = np.dot(rTs, sTc)
                error = np.apply_along_axis(np.linalg.norm, 0, np.dot(rTs, pattern['grid']) - np.dot(rTc, pattern['grid']))

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
        sensors = dict(filter(lambda x: not sensor_filter(x), sensors.items()))
        for c in collections:
            c['data'] = dict(filter(lambda x: not sensor_filter(x), c['data'].items()))
            c['labels'] = dict(filter(lambda x: not sensor_filter(x), c['labels'].items()))

    if collection_filter is not None:
        collections = [x for idx, x in enumerate(collections) if collection_filter(idx)]

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
    ap.add_argument("-p", "--print", help="Print optimized parameters as URDF tags.", dest='print', action='store_true')
    ap.add_argument("-json", "--json_file", help="JSON file containing input dataset.", type=str, required=True)
    ap.add_argument("-ef", "--error_function", type=str, default='projection', choices=['projection', 'distance'],
                    help="Error function to use with images.")

    # Check https://stackoverflow.com/questions/52431265/how-to-use-a-lambda-as-parameter-in-python-argparse
    def create_lambda_with_globals(s):
        return eval(s, globals())

    ap.add_argument("-ssf", "--sensor_selection_function", default=None, type=create_lambda_with_globals,
                    help='A string to be evaluated into a lambda function that receives a sensor name as input and '
                         'returns True or False to indicate if the sensor should be loaded (and used in the '
                         'optimization). The Syntax is lambda name: f(x), where f(x) is the function in python '
                         'language. Example: lambda name: name in ["left_laser", "frontal_camera"] , to load only '
                         'sensors left_laser and frontal_camera')
    ap.add_argument("-csf", "--collection_selection_function", default=None, type=create_lambda_with_globals,
                    help='A string to be evaluated into a lambda function that receives a collection name as input and '
                         'returns True or False to indicate if the collection should be loaded (and used in the '
                         'optimization). The Syntax is lambda name: f(x), where f(x) is the function in python '
                         'language. Example: lambda idx: idx > 5 , to load only collections 6, 7 and onward.')

    args = vars(ap.parse_args())

    # Sensor information and calibration data is saved in a json file.
    # The dataset is organized as a collection of data.
    sensors, collections, config = load_data(args['json_file'],
                                             args['sensor_selection_function'],
                                             args['collection_selection_function'])

    config['ef'] = args['error_function']

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

    # should the models be unified? (eurico)
    # Miguel: I vote yes, definitely, the work of maintaining this is daunting
    opt.addDataModel('parameters', parameters)
    opt.addDataModel('collections', collections)
    opt.addDataModel('sensors', sensors)
    opt.addDataModel('config', config)

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
    size = (pattern['dimension']["x"], pattern['dimension']["y"])

    grid = np.zeros((size[0]*size[1], 4), np.float32)
    grid[:, :2] = pattern['size'] * np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    grid[:,3] = 1

    pattern['grid'] = grid.T

    opt.addDataModel('pattern', pattern)

    if pattern['fixed']:
        parameters['pattern'] = Transform.from_position_euler(*pattern['origin']).position_quaternion
        opt.pushParamVector(group_name='L_pattern', data_key='parameters',
                            getter=partial(getPose, name='pattern'),
                            setter=partial(setPose, name='pattern'),
                            suffix=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    else:
        parameters['pattern'] = [None] * len(collections)
        for idx, collection in enumerate(collections):
            for sensor_name, labels in collection['labels'].items():
                if not labels['detected'] or sensors[sensor_name]['msg_type'] != 'Image':
                    continue
                parameters['pattern'][idx] = getPatternPose(collection['tf_tree'], config['world_link'],
                                                            sensors[sensor_name], pattern, labels)

                opt.pushParamVector(group_name='L_' + str(idx) + '_pattern', data_key='parameters',
                                    getter=partial(getPose, name='pattern', idx=idx),
                                    setter=partial(setPose, name='pattern', idx=idx),
                                    suffix=['x', 'y', 'z', 'rx', 'ry', 'rz'])
                break

    # Declare residuals
    for idx, collection in enumerate(collections):
        for sensor_name, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            params = opt.getParamsContainingPattern('S_' + sensor_name)
            if pattern['fixed']:
                params.extend(opt.getParamsContainingPattern('L_'))
            else:
                params.extend(opt.getParamsContainingPattern('L_' + str(idx)))

            if sensors[sensor_name]['msg_type'] == 'Image':
                nr = config['calibration_pattern']['dimension']["x"] * config['calibration_pattern']['dimension']["y"]
                for i in range(0, nr):
                    opt.pushResidual(name=str(idx) + '_' + sensor_name + '_' + str(i), params=params)

    opt.computeSparseMatrix()
    opt.setObjectiveFunction(objectiveFunction)

    options = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-4}
    opt.startOptimization(options)

    if args['print']:
        print('')
        for name in sensors.keys():
            printOriginTag(name, parameters[name])

        if pattern['fixed']:
            printOriginTag('pattern', parameters['pattern'])
        else:
            for i in range(0, len(collections)):
                printOriginTag('pattern' + str(i), parameters['pattern'][i])



if __name__ == '__main__':
    main()
