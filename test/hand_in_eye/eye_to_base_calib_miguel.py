#!/usr/bin/env python

import sys
import argparse
import json
import numpy as np
import cv2

from functools import partial

from tf import transformations

from OptimizationUtils import utilities
from OptimizationUtils.OptimizationUtils import Optimizer, addArguments
from OptimizationUtils.tf import TFTree, Transform
from visualization import setupVisualization, visualizationFunction


def getTransform(dataset, transform_key, collection_name):
    # The pose must be returned as a list of translation vector and rotation, i.e. [tx, ty, tz, r1, r2, r3] where r1,
    # r2,r3 are angles of the Rodrigues rotation vector.

    t = dataset['collections'][collection_name]['transforms'][transform_key]['trans']
    q = dataset['collections'][collection_name]['transforms'][transform_key]['quat']

    # Convert to [tx, ty, tz, r1, r2, r3] format
    h_matrix = transformations.quaternion_matrix(q)  # quaternion to homogeneous matrix
    matrix = h_matrix[0:3, 0:3]  # non-homogeneous matrix 3x3
    rod = utilities.matrixToRodrigues(matrix).tolist()  # matrix to Rodrigues
    return t + rod


def setTransform(dataset, values, transform_key, collection_name=None):
    # The pose must be returned as a list of translation vector and rotation, i.e. [tx, ty, tz, r1, r2, r3] where r1,
    # r2,r3 are angles of the Rodrigues rotation vector.

    # Convert from [tx, ty, tz, r1, r2, r3] format to trans and quat format
    trans = values[0:3]
    rod = values[3:]
    matrix = utilities.rodriguesToMatrix(rod)
    h_matrix = np.identity(4)
    h_matrix[0:3, 0:3] = matrix
    quat = transformations.quaternion_from_matrix(h_matrix)

    if collection_name is None:  # if no collection is given set all collections with the same value
        for collection_key in dataset['collections']:
            dataset['collections'][collection_key]['transforms'][transform_key]['trans'] = trans  # set the translation
            dataset['collections'][collection_key]['transforms'][transform_key]['quat'] = quat  # set the quaternion
    else:
        dataset['collections'][collection_name]['transforms'][transform_key]['trans'] = trans  # set the translation
        dataset['collections'][collection_name]['transforms'][transform_key]['quat'] = quat  # set the quaternion

def getCameraIntrinsics(dataset, sensor_name):
    fx = dataset['sensors'][sensor_name]['camera_info']['K'][0]
    fy = dataset['sensors'][sensor_name]['camera_info']['K'][4]
    cx = dataset['sensors'][sensor_name]['camera_info']['K'][2]
    cy = dataset['sensors'][sensor_name]['camera_info']['K'][5]
    D = dataset['sensors'][sensor_name]['camera_info']['D']
    intrinsics = [fx, fy, cx, cy]
    intrinsics.extend(D)
    return intrinsics


def setCameraIntrinsics(dataset, value, sensor_name):
    assert len(value) == 9, "value must be a list with length 9."
    dataset['sensors'][sensor_name]['camera_info']['K'][0] = value[0]
    dataset['sensors'][sensor_name]['camera_info']['K'][4] = value[1]
    dataset['sensors'][sensor_name]['camera_info']['K'][2] = value[2]
    dataset['sensors'][sensor_name]['camera_info']['K'][5] = value[3]
    dataset['sensors'][sensor_name]['camera_info']['D'] = value[4:]


def printOriginTag(name, pose):
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
    residuals = []

    for collection_key, collection in collections.items():
        tree = collection['tf_tree']
        print("visiting collection " + collection_key)

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
            error = np.apply_along_axis(np.linalg.norm, 0, diff)

            labels['error'] = diff.tolist()

            residuals.extend(error.tolist())

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

            # filename = os.path.dirname(jsonfile) + '/' + collection['data'][sensor_name]['data_file']
            # collection['data'][sensor_name]['data'] = cv2.imread(filename)

    return dataset


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
    # TODO Eurico, changed to use a single datamodel dataset, where we have inside all the others you had before.
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
    opt.addDataModel('dataset', dataset)

    # All collections have the same pose for the given sensor. For the getters we only need to get one collection.
    # Lets take the first key on the dictionary and always get that transformation.
    selected_collection_key = dataset['collections'].keys()[0]

    # The pose of several sensors are explicit optimization parameters.
    # These poses are the the same for all collections.
    # We will save these parameters in a global model.
    for sensor_name, sensor in dataset['sensors'].items():
        parent = sensor['calibration_parent']
        child = sensor['calibration_child']
        transform_key = utilities.generateKey(parent, child)

        # Six parameters containing the optimized transformation for this sensor
        opt.pushParamVector(group_name=transform_key, data_key='dataset',
                            getter=partial(getTransform, transform_key=transform_key,
                                           collection_name=selected_collection_key),
                            setter=partial(setTransform, transform_key=transform_key, collection_name=None),
                            suffix=['_x', '_y', '_z', '_r1', '_r2', '_r3'])

        # Add intrinsics if sensor is a camera
        if dataset['sensors'][sensor_name]['msg_type'] == 'Image' and args['intrinsic']:
            opt.pushParamVector(group_name='S_' + sensor_name + '_intrinsics', data_key='dataset',
                                getter=partial(getCameraIntrinsics, sensor_name=sensor_name),
                                setter=partial(setCameraIntrinsics, sensor_name=sensor_name),
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

    # TODO just for debugging
    # dataset['calibration_config']['calibration_pattern']['fixed'] = False

    # Create first guesses for the pattern poses
    # TODO I think it is better that each piece of code of loop does a
    #  single thing. Before your code would do two things: set up the first guess and create the parameters. For
    #  clarity, I have separated this into two pieces of code.
    if dataset['calibration_config']['calibration_pattern']['fixed']:

        parent = dataset['calibration_config']['calibration_pattern']['parent_link']
        child = dataset['calibration_config']['calibration_pattern']['link']
        transform_key = utilities.generateKey(parent, child)

        # TODO origin is used as a first guess when the pattern is fixed, right? Why not use the getpose as bellow?
        trans = dataset['calibration_config']['calibration_pattern']['origin'][0:3]
        rpy = dataset['calibration_config']['calibration_pattern']['origin'][3:]
        quat = transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

        # The pattern is fixed but we have a transform for each collection (although they are all the same)
        for collection_key, collection in dataset['collections'].items():
            collection['transforms'][transform_key] = {'parent': parent, 'child': child, 'trans': trans,
                                                       'quat': quat}

        opt.pushParamVector(group_name=transform_key, data_key='dataset',
                            getter=partial(getTransform, transform_key=transform_key,
                                           collection_name=selected_collection_key),
                            setter=partial(setTransform, transform_key=transform_key, collection_name=None),
                            suffix=['_x', '_y', '_z', '_r1', '_r2', '_r3'])

    else:
        parent = dataset['calibration_config']['calibration_pattern']['parent_link']
        child = dataset['calibration_config']['calibration_pattern']['link']
        transform_key = utilities.generateKey(parent, child)

        for collection_key, collection in dataset['collections'].items():
            for sensor_name, labels in collection['labels'].items():
                if not labels['detected'] or not dataset['sensors'][sensor_name]['msg_type'] == 'Image':
                    continue

                #TODO this function could return a tuple of (trans,quat)
                # For first guess, get pattern pose using solvePNP
                pose = getPatternFirstGuessPose(collection['tf_tree'], parent, dataset['sensors'][sensor_name], pattern,
                                                labels)
                trans = pose[0:3]
                quat = pose[3:]
                collection['transforms'][transform_key] = {'parent': parent, 'child': child, 'trans': trans,
                                                           'quat': quat}

                opt.pushParamVector(group_name='c' + collection_key + '_' + transform_key, data_key='dataset',
                                    getter=partial(getTransform, transform_key=transform_key,
                                                   collection_name=selected_collection_key),
                                    setter=partial(setTransform, transform_key=transform_key,
                                                   collection_name=selected_collection_key),
                                    suffix=['_x', '_y', '_z', '_r1', '_r2', '_r3'])

                break  # only need one first guess of the pattern pose, even if there are more sensors

        if trans is None or quat is None:
            raise ValueError("Could not find a first guess for the pattern pose?")

    # Declare residuals
    for collection_key, collection in dataset['collections'].items():

        for sensor_name, labels in collection['labels'].items():

            print('num_labels = ' + str(len(collection['labels'][sensor_name]['idxs'])))
            if not labels['detected']:
                continue

            # Params related to the sensor. Parameter name is the transform_key of the sensors optimized transform
            params = opt.getParamsContainingPattern(utilities.generateKey(sensor['calibration_parent'],
                                                                          sensor['calibration_child']))
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

    opt.computeSparseMatrix()
    opt.setObjectiveFunction(objectiveFunction)



    opt.printParameters()
    opt.printResiduals()
    opt.printSparseMatrix()

    # Visualization
    if args['view_optimization']:
        print("Configuring visualization ... ")
        graphics = setupVisualization(dataset, args)
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(dataset_graphics)
        opt.addDataModel('graphics', graphics)

    opt.setVisualizationFunction(visualizationFunction, args['view_optimization'], niterations=1, figures=[])

    # Start optimization
    options = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': None, 'jac': '3-point', 'x_scale': 'jac'}
    opt.startOptimization(options)


    exit(0)

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
            for sensor_name, sensor in sensors.items():
                print(sensor_name)
                print(getCameraIntrinsics(sensors, sensor_name))

        print('\nExtrinsics')
        print('----------')
        for sensor_name in sensors.keys():
            printOriginTag(sensor_name, parameters[sensor_name])

        # printOriginTag('pattern', parameters['pattern'])


if __name__ == '__main__':
    main()
