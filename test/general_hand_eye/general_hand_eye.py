#!/usr/bin/env python
import os
import sys
import argparse
import json
import numpy as np
from copy import deepcopy

import cv2

from functools import partial
from tf import transformations
from OptimizationUtils import utilities
from OptimizationUtils.OptimizationUtils import Optimizer, addArguments
from OptimizationUtils.tf import TFTree, Transform
from visualization import setupVisualization, visualizationFunction


def getterTransform(dataset, transform_key, collection_name):
    # The pose must be returned as a list of translation vector and rotation, i.e. [tx, ty, tz, r1, r2, r3] where r1,
    # r2,r3 are angles of the Rodrigues rotation vector.

    trans = dataset['collections'][collection_name]['transforms'][transform_key]['trans']
    quat = dataset['collections'][collection_name]['transforms'][transform_key]['quat']

    # Convert from (trans, quat) to [tx, ty, tz, r1, r2, r3] format
    h_matrix = transformations.quaternion_matrix(quat)  # quaternion to homogeneous matrix
    matrix = h_matrix[0:3, 0:3]  # non-homogeneous matrix 3x3
    rod = utilities.matrixToRodrigues(matrix).tolist()  # matrix to Rodrigues
    return trans + rod


def setterTransform(dataset, values, transform_key, collection_name=None):
    # The pose must be returned as a list of translation vector and rotation, i.e. [tx, ty, tz, r1, r2, r3] where r1,
    # r2,r3 are angles of the Rodrigues rotation vector.

    # Convert from [tx, ty, tz, r1, r2, r3] format to trans and quat format
    trans, rod = values[0:3], values[3:]
    matrix = utilities.rodriguesToMatrix(rod)
    h_matrix = np.identity(4)
    h_matrix[0:3, 0:3] = matrix
    quat = transformations.quaternion_from_matrix(h_matrix)

    if collection_name is None:  # if collection_name is None, set all collections with the same value
        for collection_key in dataset['collections']:
            dataset['collections'][collection_key]['transforms'][transform_key]['trans'] = trans  # set the translation
            dataset['collections'][collection_key]['transforms'][transform_key]['quat'] = quat  # set the quaternion
    else:
        dataset['collections'][collection_name]['transforms'][transform_key]['trans'] = trans  # set the translation
        dataset['collections'][collection_name]['transforms'][transform_key]['quat'] = quat  # set the quaternion


def getterCameraIntrinsics(dataset, sensor_name):
    fx = dataset['sensors'][sensor_name]['camera_info']['K'][0]
    fy = dataset['sensors'][sensor_name]['camera_info']['K'][4]
    cx = dataset['sensors'][sensor_name]['camera_info']['K'][2]
    cy = dataset['sensors'][sensor_name]['camera_info']['K'][5]
    D = dataset['sensors'][sensor_name]['camera_info']['D']
    intrinsics = [fx, fy, cx, cy]
    intrinsics.extend(D)
    return intrinsics


def setterCameraIntrinsics(dataset, value, sensor_name):
    assert len(value) == 9, "value must be a list with length 9."
    dataset['sensors'][sensor_name]['camera_info']['K'][0] = value[0]
    dataset['sensors'][sensor_name]['camera_info']['K'][4] = value[1]
    dataset['sensors'][sensor_name]['camera_info']['K'][2] = value[2]
    dataset['sensors'][sensor_name]['camera_info']['K'][5] = value[3]
    dataset['sensors'][sensor_name]['camera_info']['D'] = value[4:]


def printOriginTag(name, sensor, transforms):
    parent_sensor = sensor['calibration_parent']
    child_sensor = sensor['calibration_child']
    transform_key = utilities.generateKey(parent_sensor, child_sensor)

    trans = transforms[transform_key]['trans']
    quat = transforms[transform_key]['quat']
    pose = np.array(trans).tolist() + np.array(quat).tolist()

    xform = Transform(*pose)
    t = [str(x) for x in list(xform.position)]
    r = [str(x) for x in list(xform.euler)]
    print('<!-- {} -->'.format(name))
    print('<origin xyz="{}" rpy="{}"/>'.format(" ".join(t), " ".join(r)))


def getPatternFirstGuessPose(tf_tree, frame_id, camera, pattern, labels):
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
    collections = models['dataset']['collections']
    sensors = models['dataset']['sensors']
    pattern = models['dataset']['pattern']
    config = models['dataset']['calibration_config']
    args = models['args']
    residuals = []

    for collection_key, collection in collections.items():
        tree = collection['tf_tree']
        # print("visiting collection " + collection_key)

        # Chessboard to root transformation.
        # We do not need to worry about the calibration pattern being fixed or not since we have a stored transform
        # for each collection. If the pattern is fixed, these transforms are all the same. But his is handled inside
        # the optimization utils because of the parameters configuration, we don't have to worry about it.
        parent_pattern = config['calibration_pattern']['parent_link']
        child_pattern = config['calibration_pattern']['link']
        transform_key = utilities.generateKey(parent_pattern, child_pattern)
        trans = collection['transforms'][transform_key]['trans']
        quat = collection['transforms'][transform_key]['quat']
        tree.add_transform(parent_pattern, child_pattern, Transform(trans[0], trans[1], trans[2],
                                                                    quat[0], quat[1], quat[2], quat[3]))
        rTc = tree.lookup_transform(config['calibration_pattern']['link'], config['world_link']).matrix

        for sensor_name, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            parent_sensor = sensors[sensor_name]['calibration_parent']
            child_sensor = sensors[sensor_name]['calibration_child']
            transform_key = utilities.generateKey(parent_sensor, child_sensor)

            trans = collection['transforms'][transform_key]['trans']
            quat = collection['transforms'][transform_key]['quat']
            tree.add_transform(parent_sensor, child_sensor, Transform(trans[0], trans[1], trans[2],
                                                                      quat[0], quat[1], quat[2], quat[3]))
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
            # error = np.apply_along_axis(np.linalg.norm, 0, diff)
            error = np.sum(diff * diff, 0)  # big boost

            if not labels.has_key('error'):
                labels['init_error'] = diff.tolist()

            labels['error'] = diff.tolist()

            # Required by the visualization function to publish annotated images
            if args['ros_visualization']:
                idxs_projected = []
                for idx in range(0, projected.shape[1]):
                    idxs_projected.append({'x': projected[0][idx], 'y': projected[1][idx]})
                collection['labels'][sensor_name]['idxs_projected'] = idxs_projected  # store projections

                if 'idxs_initial' not in collection['labels'][sensor_name]:  # store the first projections
                    collection['labels'][sensor_name]['idxs_initial'] = deepcopy(idxs_projected)

            residuals.extend(error.tolist())

            ## ---
            # A = Transform(*pparam)
            # A = Transform(*getPatternFirstGuessPose(tree, sensors[sensor_name]['calibration_child'], sensors[sensor_name], pattern, labels))
            # X = tree.lookup_transform(config['world_link'] , config['calibration_pattern']['link'])
            # Z = tree.lookup_transform(sensors[sensor_name]['calibration_parent'], sensors[sensor_name]['calibration_child'])
            # B = tree.lookup_transform(config['world_link'] , sensors[sensor_name]['calibration_parent'])

            # diff = np.dot(A.matrix, X.matrix) - np.dot(Z.matrix, B.matrix)
            # err = np.linalg.norm(diff, ord='fro')

            # residual.append(float(err)*100)
            ## ---

    return residuals


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
    # collections = dataset['collections']

    # Filter the sensors and collection by sensor name
    if sensor_filter is not None:
        dataset['sensors'] = {k: v for k, v in dataset['sensors'].items() if sensor_filter(k)}
        for c in dataset['collections']:
            c['data'] = dict(filter(lambda x: not sensor_filter(x), c['data'].items()))
            c['labels'] = dict(filter(lambda x: not sensor_filter(x), c['labels'].items()))

    if collection_filter is not None:
        dataset['collections'] = {k: v for k, v in dataset['collections'].items() if collection_filter(k)}

    # Image data is not stored in the JSON file for practical reasons. Mainly, too much data.
    # Instead, the image is stored in a compressed format and its name is in the collection.
    for collection in dataset['collections'].values():
        for sensor_name in collection['data'].keys():
            if dataset['sensors'][sensor_name]['msg_type'] != 'Image':
                continue  # we are only interested in images, remember?

            filename = os.path.dirname(jsonfile) + '/' + collection['data'][sensor_name]['data_file']
            collection['data'][sensor_name]['data'] = cv2.imread(filename)

    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap = addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-i", "--intrinsic", help="Optimize the intrinsic parameters of cameras.", dest='intrinsic',
                    action='store_true')
    ap.add_argument("-p", "--print", help="Print optimized parameters as URDF tags.", dest='print', action='store_true')
    ap.add_argument("-json", "--json_file", help="JSON file containing input dataset.", type=str, required=True)
    ap.add_argument("-e", "--error", help="Final errors output file. (JSON format)", type=str)
    ap.add_argument("-rv", "--ros_visualization", help="Publish ros visualization markers.", action='store_true')
    ap.add_argument("-si", "--show_images", help="shows images for each camera", action='store_true', default=False)
    ap.add_argument("-sp", "--single_pattern", help="show a single pattern instead of one per collection.",
                    action='store_true', default=False)

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
    # TODO Eurico, I think it is better to have a single data model which correponds to the json file. To split the
    #  content of the json into pieces does not make a lot of sense. For example, in the calibration example we save
    #  the altered json, if you wanted to do that you'd have to put it back toghether. We should discuss this also in
    #  the scope of #55 and #100 and may end up revising, but we shoul be consistent throughout the several examples.
    #  I see additional data models as a higher level stuff: for example, supose you have two jsons ...
    dataset = load_data(args['json_file'], args['sensor_selection_function'], args['collection_selection_function'])

    # The best way to keep track of all transformation pairs is to a have a
    # transformation tree similar to what ROS offers (github.com/safijari/tiny_tf).
    for collection in dataset['collections'].values():
        tree = TFTree()
        for _, tf in collection['transforms'].items():
            param = tf['trans'] + tf['quat']
            tree.add_transform(tf['parent'], tf['child'], Transform(*param))

        collection['tf_tree'] = tree

    # Setup optimizer
    opt = Optimizer()
    opt.addDataModel('dataset', dataset)  # a single data model containing the full dataset.
    opt.addDataModel('args', args)  # a model containing the input args.

    # In some cases, all collections have the same transform values. For example, the pose of a given sensor. We define
    # a collection to be the one used to get the value of the transform. Arbitrarly, we select the first key in the
    # collections dictionary
    selected_collection_key = dataset['collections'].keys()[0]

    # Steaming from the config json, we define a transform to be optimized for each sensor. It could happen that two
    # or more sensors define the same transform to be optimized (#120). To cope with this we first create a list of
    # transformations to be optimized and then compute the unique set of that list.
    transforms_set = set()
    for sensor_name, sensor in dataset['sensors'].items():
        parent = sensor['calibration_parent']
        child = sensor['calibration_child']
        transform_key = utilities.generateKey(parent, child)
        transforms_set.add(transform_key)

    for transform_key in transforms_set:  # push six parameters for each transform to be optimized.
        opt.pushParamVector(group_name=transform_key, data_key='dataset',
                            getter=partial(getterTransform, transform_key=transform_key,
                                           collection_name=selected_collection_key),
                            setter=partial(setterTransform, transform_key=transform_key, collection_name=None),
                            suffix=['_x', '_y', '_z', '_r1', '_r2', '_r3'])

    # Add intrinsics as optimization parameters if sensor is a camera
    if args['intrinsic']:
        for sensor_name, sensor in dataset['sensors'].items():

            if dataset['sensors'][sensor_name]['msg_type'] == 'Image':
                opt.pushParamVector(group_name='S_' + sensor_name + '_intrinsics', data_key='dataset',
                                    getter=partial(getterCameraIntrinsics, sensor_name=sensor_name),
                                    setter=partial(setterCameraIntrinsics, sensor_name=sensor_name),
                                    suffix=['_fx', '_fy', '_cx', '_cy', '_k1', '_k2', '_t1', '_t2', '_k3'])

    # Cache the chessboard grid
    # TODO Eurico, changed this a bit to make sure the data model pattern does not contain the same information as the
    #  calibration_config, calibration pattern.
    pattern = {}
    dx = dataset['calibration_config']['calibration_pattern']['dimension']["x"]
    dy = dataset['calibration_config']['calibration_pattern']['dimension']["y"]
    size = dataset['calibration_config']['calibration_pattern']['size']
    grid = np.zeros((dx * dy, 4), np.float32)
    grid[:, :2] = size * np.mgrid[0:dx, 0:dy].T.reshape(-1, 2)
    grid[:, 3] = 1
    pattern['grid'] = grid.T
    dataset['pattern'] = pattern

    # Create first guesses for the pattern poses and the corresponding optimization parameters
    if dataset['calibration_config']['calibration_pattern']['fixed']:

        parent = dataset['calibration_config']['calibration_pattern']['parent_link']
        child = dataset['calibration_config']['calibration_pattern']['link']
        transform_key = utilities.generateKey(parent, child)

        found = False
        for collection_key, collection in dataset['collections'].items():
            for sensor_name, labels in collection['labels'].items():
                if not labels['detected'] or not dataset['sensors'][sensor_name]['msg_type'] == 'Image':
                    continue
                pose = getPatternFirstGuessPose(collection['tf_tree'], parent, dataset['sensors'][sensor_name], pattern,
                                                labels)
                trans = pose[0:3]
                quat = pose[3:]

                found = True

            if found:
                break

        if not found:
            trans = dataset['calibration_config']['calibration_pattern']['origin'][0:3]
            rpy = dataset['calibration_config']['calibration_pattern']['origin'][3:]
            quat = transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

        # The pattern is fixed but we have a transform for each collection (although they are all the same)
        for collection_key, collection in dataset['collections'].items():
            collection['transforms'][transform_key] = {'parent': parent, 'child': child, 'trans': trans,
                                                       'quat': quat}

        opt.pushParamVector(group_name=transform_key, data_key='dataset',
                            getter=partial(getterTransform, transform_key=transform_key,
                                           collection_name=selected_collection_key),
                            setter=partial(setterTransform, transform_key=transform_key, collection_name=None),
                            suffix=['_x', '_y', '_z', '_r1', '_r2', '_r3'])

    else:
        parent = dataset['calibration_config']['calibration_pattern']['parent_link']
        child = dataset['calibration_config']['calibration_pattern']['link']
        transform_key = utilities.generateKey(parent, child)

        for collection_key, collection in dataset['collections'].items():
            for sensor_name, labels in collection['labels'].items():
                if not labels['detected'] or not dataset['sensors'][sensor_name]['msg_type'] == 'Image':
                    continue

                # TODO this function could return a tuple of (trans,quat)
                # For first guess, get pattern pose using solvePNP
                pose = getPatternFirstGuessPose(collection['tf_tree'], parent, dataset['sensors'][sensor_name], pattern,
                                                labels)
                trans = pose[0:3]
                quat = pose[3:]
                collection['transforms'][transform_key] = {'parent': parent, 'child': child, 'trans': trans,
                                                           'quat': quat}

                opt.pushParamVector(group_name='c' + collection_key + '_' + transform_key, data_key='dataset',
                                    getter=partial(getterTransform, transform_key=transform_key,
                                                   collection_name=collection_key),
                                    setter=partial(setterTransform, transform_key=transform_key,
                                                   collection_name=collection_key),
                                    suffix=['_x', '_y', '_z', '_r1', '_r2', '_r3'])

                break  # only need one first guess of the pattern pose, even if there are more sensors

        if trans is None or quat is None:
            raise ValueError("Could not find a first guess for the pattern pose?")

    # Declare residuals
    for collection_key, collection in dataset['collections'].items():
        for sensor_name, labels in collection['labels'].items():
            if not labels['detected']:
                continue

            # Params related to the sensor. Parameter name is the transform_key of the sensors optimized transform
            params = opt.getParamsContainingPattern(
                utilities.generateKey(dataset['sensors'][sensor_name]['calibration_parent'],
                                      dataset['sensors'][sensor_name]['calibration_child']))

            params.extend(opt.getParamsContainingPattern(sensor_name))  # intrinsics have the sensor name

            # Params related to the pattern
            transform_key = utilities.generateKey(dataset['calibration_config']['calibration_pattern']['parent_link'],
                                                  dataset['calibration_config']['calibration_pattern']['link'])

            if dataset['calibration_config']['calibration_pattern']['fixed']:
                params.extend(opt.getParamsContainingPattern(transform_key))
            else:
                params.extend(opt.getParamsContainingPattern('c' + collection_key + '_' + transform_key))

            if dataset['sensors'][sensor_name]['msg_type'] == 'Image':
                for i in range(0, len(labels['idxs'])):
                    opt.pushResidual(name='c' + collection_key + '_' + sensor_name + '_' + str(i), params=params)

                # opt.pushResidual(name='c' + collection_key + '_' + sensor_name + '_AXZB', params=params)

    opt.computeSparseMatrix()
    opt.setObjectiveFunction(objectiveFunction)

    # opt.printParameters()
    # opt.printResiduals()
    # opt.printSparseMatrix()

    # Visualization
    if args['view_optimization']:
        opt.setInternalVisualization(True)
    else:
        opt.setInternalVisualization(False)

    if args['ros_visualization']:
        print("Configuring visualization ... ")
        graphics = setupVisualization(dataset, args)
        opt.addDataModel('graphics', graphics)
        opt.setVisualizationFunction(visualizationFunction, args['ros_visualization'], niterations=1, figures=[])

    # Start optimization
    options = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': None, 'jac': '2-point', 'x_scale': 'jac'}
    opt.startOptimization(options)

    rmse = np.sqrt(np.mean(np.array(objectiveFunction(opt.data_models))))
    print("RMSE {}".format(rmse))

    if args['error']:
        # Build summary.
        summary = {'collections': {k: {kk: {'errors': vv['error'], 'init_errors': vv['init_error']}
                                       for kk, vv in v['labels'].items() if vv['detected']} for k, v in
                                   dataset['collections'].items()}}
        # remove empty collections
        summary['collections'] = {k: v for k, v in summary['collections'].items() if len(v) > 0}

        for key, collection in dataset['collections'].items():
            tree = collection['tf_tree']

            for sensor_name, labels in collection['labels'].items():
                if not labels['detected']:
                    continue

                summary['collections'][key][sensor_name]['B'] = getPatternFirstGuessPose(tree, dataset['sensors'][
                    sensor_name]['calibration_child'],
                                                                                         dataset['sensors'][
                                                                                             sensor_name], pattern,
                                                                                         labels)
                summary['collections'][key][sensor_name]['X'] = tree.lookup_transform(
                    dataset['calibration_config']['calibration_pattern']['link'],
                    dataset['calibration_config']['world_link']).position_quaternion
                summary['collections'][key][sensor_name]['Z'] = tree.lookup_transform(
                    dataset['sensors'][sensor_name]['calibration_child'],
                    dataset['sensors'][sensor_name]['calibration_parent']).position_quaternion
                summary['collections'][key][sensor_name]['A'] = tree.lookup_transform(
                    dataset['calibration_config']['world_link'],
                    dataset['sensors'][sensor_name]['calibration_parent']).position_quaternion

        with open(args['error'], 'w') as f:
            print >> f, json.dumps(summary, indent=2, sort_keys=True, separators=(',', ': '))

    if args['print']:
        if args['intrinsic']:
            print('\nIntrinsics')
            print('----------')
            for sensor_name, sensor in dataset['sensors'].items():
                print(sensor_name)
                print(getterCameraIntrinsics(dataset, sensor_name))

        print('\nExtrinsics')
        print('----------')
        for sensor_name, sensor in dataset['sensors'].items():
            printOriginTag(sensor_name, sensor, dataset['collections'][selected_collection_key]['transforms'])

        # printOriginTag('pattern', parameters['pattern'])


if __name__ == '__main__':
    main()
