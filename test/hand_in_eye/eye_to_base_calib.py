#!/usr/bin/env python

import sys
import argparse
import json
import numpy as np
import cv2

from functools import partial

from OptimizationUtils import utilities
from OptimizationUtils.OptimizationUtils import Optimizer, addArguments
from OptimizationUtils.tf import TFTree, Transform
from visualization import setupVisualization, visualizationFunction


def getPose(model, name, idx=None):
    # A pose is represented by a translation vector and a quaternion.
    # However, the rotation's optimization is done on a rotation vector (see Rodrigues).
    if idx is None:
        xform = Transform(*model[name])
        return model[name][0:3] + utilities.matrixToRodrigues(xform.rotation_matrix).tolist()
    else:
        xform = Transform(*model[name][idx])
        return model[name][idx][0:3] + utilities.matrixToRodrigues(xform.rotation_matrix).tolist()


def setPose(model, pose, name, idx=None):
    # A pose is represented by a translation vector and a quaternion.
    # However, the optimization of the rotation is done on a rotation vector (see Rodrigues).
    mat = utilities.traslationRodriguesToTransform(pose[0:3], pose[3:])
    xform = Transform.from_matrix(mat)
    if idx is None:
        model[name] = xform.position_quaternion
    else:
        model[name][idx] = xform.position_quaternion


def getCameraIntrinsics(model, name):
    fx = model[name]['camera_info']['K'][0]
    fy = model[name]['camera_info']['K'][4]
    cx = model[name]['camera_info']['K'][2]
    cy = model[name]['camera_info']['K'][5]
    D = model[name]['camera_info']['D']
    intrinsics = [fx, fy, cx, cy]
    intrinsics.extend(D)
    return intrinsics


def setCameraIntrinsics(model, value, name):
    assert len(value) == 9, "value must be a list with length 9."
    model[name]['camera_info']['K'][0] = value[0]
    model[name]['camera_info']['K'][4] = value[1]
    model[name]['camera_info']['K'][2] = value[2]
    model[name]['camera_info']['K'][5] = value[3]
    model[name]['camera_info']['D'] = value[4:]


def printOriginTag(name, pose):
    xform = Transform(*pose)
    t = [str(x) for x in list(xform.position)]
    r = [str(x) for x in list(xform.euler)]
    print('<!-- {} -->'.format(name))
    print('<origin xyz="{}" rpy="{}"/>'.format(" ".join(t), " ".join(r)))


def getPatternPose(tf_tree, frame_id, camera, pattern, labels):
    # sensor to root transformation
    rTs = tf_tree.lookup_transform(camera['camera_info']['header']['frame_id'], frame_id).matrix

    # convert chessboard corners from pixels to sensor coordinates.
    K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(camera['camera_info']['K']))
    D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(camera['camera_info']['D']))

    corners = np.zeros((2, len(labels['idxs'])), dtype=np.float32)
    ids = [0] * len(labels['idxs'])
    for idx, point in enumerate(labels['idxs']):
        corners[0, idx] = point['x']
        corners[1, idx] = point['y']
        ids[idx] = point['id']

    _, rvecs, tvecs = cv2.solvePnP(np.array(pattern['grid'][:3, :].T[ids]), corners.T.reshape(-1, 1, 2), K, D)
    sTc = utilities.traslationRodriguesToTransform(tvecs, rvecs)

    return Transform.from_matrix(np.dot(rTs, sTc)).position_quaternion


def objectiveFunction(models):
    parameters = models['parameters']
    collections = models['collections']
    sensors = models['sensors']
    pattern = models['pattern']
    config = models['config']

    print(parameters)
    # exit(0)
    residual = []

    for cid, collection in collections.items():
        tree = collection['tf_tree']

        # chessboard to root transformation
        if pattern['fixed']:
            tree.add_transform(pattern['parent_link'], pattern['link'], Transform(*parameters['pattern']))
            rTc = tree.lookup_transform(pattern['link'], config['world_link']).matrix
        elif cid in parameters['pattern']:
            tree.add_transform(pattern['parent_link'], pattern['link'] + cid, Transform(*parameters['pattern'][cid]))
            rTc = tree.lookup_transform(pattern['link'] + cid, config['world_link']).matrix

        for sensor_name, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            xform = Transform(*parameters[sensor_name])
            tree.add_transform(sensors[sensor_name]['calibration_parent'], sensors[sensor_name]['calibration_child'],
                               xform)

            # root to sensor transformation
            sTr = tree.lookup_transform(config['world_link'],
                                        sensors[sensor_name]['camera_info']['header']['frame_id']).matrix

            # chess to camera transformation
            sTc = np.dot(sTr, rTc)

            # convert chessboard corners from pixels to sensor coordinates.
            K = np.ndarray((3, 3), dtype=np.float, buffer=np.array(sensors[sensor_name]['camera_info']['K']))
            D = np.ndarray((5, 1), dtype=np.float, buffer=np.array(sensors[sensor_name]['camera_info']['D']))

            width = sensors[sensor_name]['camera_info']['width']
            height = sensors[sensor_name]['camera_info']['height']

            corners = np.zeros((2, len(labels['idxs'])), dtype=np.float32)
            ids = range(0, len(labels['idxs']))
            for idx, point in enumerate(labels['idxs']):
                corners[0, idx] = point['x']
                corners[1, idx] = point['y']
                ids[idx] = point['id']

            projected, _, _ = utilities.projectToCamera(K, D, width, height, np.dot(sTc, pattern['grid'].T[ids].T))
            diff = projected - corners
            error = np.apply_along_axis(np.linalg.norm, 0, diff)

            labels['error'] = diff.tolist()

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
    # collections = [x for _, x in sorted(dataset['collections'].items())]
    collections = dataset['collections']

    # Filter the sensors and collection by sensor name
    if sensor_filter is not None:
        sensors = {k: v for k, v in sensors.items() if sensor_filter(k)}
        for c in collections:
            c['data'] = dict(filter(lambda x: not sensor_filter(x), c['data'].items()))
            c['labels'] = dict(filter(lambda x: not sensor_filter(x), c['labels'].items()))

    if collection_filter is not None:
        collections = {k: v for k, v in collections.items() if collection_filter(k)}

    # Image data is not stored in the JSON file for practical reasons. Mainly, too much data.
    # Instead, the image is stored in a compressed format and its name is in the collection.
    for collection in collections.values():
        for sensor_name in collection['data'].keys():
            if sensors[sensor_name]['msg_type'] != 'Image':
                continue  # we are only interested in images, remember?

            # filename = os.path.dirname(jsonfile) + '/' + collection['data'][sensor_name]['data_file']
            # collection['data'][sensor_name]['data'] = cv2.imread(filename)

    return sensors, collections, dataset['calibration_config']


def main():
    ap = argparse.ArgumentParser()
    ap = addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-i", "--intrinsic", help="Optimize the intrinsic parameters of cameras.", dest='intrinsic',
                    action='store_true')
    ap.add_argument("-p", "--print", help="Print optimized parameters as URDF tags.", dest='print', action='store_true')
    ap.add_argument("-json", "--json_file", help="JSON file containing input dataset.", type=str, required=True)
    ap.add_argument("-e", "--error", help="Final errors output file. (JSON format)", type=str)

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

    # The best way to keep track of all transformation pairs is to a have a
    # transformation tree similar to what ROS offers (github.com/safijari/tiny_tf).
    for collection in collections.values():
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
    # TODO this should be removed! Will work on it. asdasd
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
        t = collections.values()[0]['transforms'][key]['trans']
        r = collections.values()[0]['transforms'][key]['quat']
        parameters[name] = t + r

        opt.pushParamVector(group_name='S_' + name, data_key='parameters',
                            getter=partial(getPose, name=name),
                            setter=partial(setPose, name=name),
                            suffix=['x', 'y', 'z', 'rx', 'ry', 'rz'])

        if sensors[name]['msg_type'] == 'Image' and args['intrinsic']:
            opt.pushParamVector(group_name='S_' + name + '_intrinsics_', data_key='sensors',
                                getter=partial(getCameraIntrinsics, name=name),
                                setter=partial(setCameraIntrinsics, name=name),
                                suffix=['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 't1', 't2', 'k3'])

    pattern = config['calibration_pattern']
    # cache the chessboard grid
    size = (pattern['dimension']["x"], pattern['dimension']["y"])

    grid = np.zeros((size[0] * size[1], 4), np.float32)
    grid[:, :2] = pattern['size'] * np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    grid[:, 3] = 1

    pattern['grid'] = grid.T

    opt.addDataModel('pattern', pattern)

    if pattern['fixed']:
        parameters['pattern'] = Transform.from_position_euler(*pattern['origin']).position_quaternion
        opt.pushParamVector(group_name='L_pattern_', data_key='parameters',
                            getter=partial(getPose, name='pattern'),
                            setter=partial(setPose, name='pattern'),
                            suffix=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    else:
        parameters['pattern'] = {}
        for idx, collection in collections.items():
            for sensor_name, labels in collection['labels'].items():
                if not labels['detected'] or sensors[sensor_name]['msg_type'] != 'Image':
                    continue
                parameters['pattern'][idx] = getPatternPose(collection['tf_tree'], pattern['parent_link'],
                                                            sensors[sensor_name], pattern, labels)

                opt.pushParamVector(group_name='L_' + idx + '_pattern_', data_key='parameters',
                                    getter=partial(getPose, name='pattern', idx=idx),
                                    setter=partial(setPose, name='pattern', idx=idx),
                                    suffix=['x', 'y', 'z', 'rx', 'ry', 'rz'])

                break

    # Declare residuals
    for idx, collection in collections.items():
        for sensor_name, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            params = opt.getParamsContainingPattern('S_' + sensor_name)
            if pattern['fixed']:
                params.extend(opt.getParamsContainingPattern('L_'))
            else:
                params.extend(opt.getParamsContainingPattern('L_' + idx))

            if sensors[sensor_name]['msg_type'] == 'Image':
                nr = len(labels['idxs'])
                for i in range(0, nr):
                    opt.pushResidual(name=idx + '_' + sensor_name + '_' + str(i), params=params)

    opt.computeSparseMatrix()
    opt.setObjectiveFunction(objectiveFunction)

    # Start optimization
    options = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': None, 'jac': '3-point', 'x_scale': 'jac'}
    opt.startOptimization(options)

    rmse = np.sqrt(np.mean(np.array(objectiveFunction(opt.data_models)) ** 2))
    print("RMSE {}".format(rmse))

    if args['error']:
        # Build summary.
        summary = {'collections': {k: {kk: vv['error'] for kk, vv in v['labels'].items() if vv['detected']} for k, v in
                                   collections.items()}}
        ## remove empty collections
        summary['collections'] = {k: v for k, v in summary['collections'].items() if len(v) > 0}

        with open(args['error'], 'w') as f:
            print >> f, json.dumps(summary, indent=2, sort_keys=True, separators=(',', ': '))

    if args['print']:
        if args['intrinsic']:
            print('\nIntrinsics')
            print('----------')
            for name, sensor in sensors.items():
                print(name)
                print(getCameraIntrinsics(sensors, name))

        print('\nExtrinsics')
        print('----------')
        for name in sensors.keys():
            printOriginTag(name, parameters[name])

        # printOriginTag('pattern', parameters['pattern'])


if __name__ == '__main__':
    main()
