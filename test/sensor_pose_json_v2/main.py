#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then own modules)
# -------------------------------------------------------------------------------
from numpy import inf
from functools import partial
import json
import sys
import argparse
from colorama import Fore, Style

import rospkg
from urdf_parser_py.urdf import URDF

import OptimizationUtils.OptimizationUtils as OptimizationUtils
from getters_and_setters import *
from objective_function import *
from test.sensor_pose_json_v2.chessboard import createChessBoardData
from test.sensor_pose_json_v2.visualization import *


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
            if isinstance(item, np.ndarray) and key == 'data':  # to avoid saving images in the json
                del node[key]

            elif isinstance(item, np.ndarray):
                node[key] = item.tolist()
            pass


# Save to json file

def createJSONFile(output_file, input):
    D = deepcopy(input)

    walk(D)

    print("Saving the json output file to " + str(output_file) + ", please wait, it could take a while ...")
    f = open(output_file, 'w')
    json.encoder.FLOAT_REPR = lambda f: ("%.6f" % f)  # to get only four decimal places on the json file
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
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-json", "--json_file", help="Json file containing input dataset.", type=str, required=True)
    ap.add_argument("-rv", "--ros_visualization", help="Publish ros visualization markers.", action='store_true')
    ap.add_argument("-si", "--show_images", help="shows images for each camera", action='store_true', default=False)
    ap.add_argument("-sp", "--single_pattern", help="show a single pattern instead of one per collection.",
                    action='store_true', default=False)
    ap.add_argument("-uic", "--use_incomplete_collections",
                    help="Remove any collection which does not have a detection for all sensors.",
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
                         'language. Example: lambda name: int(name) > 5 , to load only collections 6, 7, and onward.')

    args = vars(ap.parse_args())
    print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- Reading robot description file from param /robot_description
    # ---------------------------------------
    # rospy.loginfo('Reading /robot_description ros param')
    # xml_robot = URDF.from_parameter_server()  # needed to create the optimized xacro at the end of the optimization

    # ---------------------------------------
    # --- INITIALIZATION Read data from file
    # ---------------------------------------
    """ Loads a json file containing the detections"""
    f = open(args['json_file'], 'r')
    dataset_sensors = json.load(f)

    # Load images from files into memory. Images in the json file are stored in separate png files and in their place
    # a field "data_file" is saved with the path to the file. We must load the images from the disk.
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not sensor['msg_type'] == 'Image':  # nothing to do here.
                continue

            filename = os.path.dirname(args['json_file']) + '/' + collection['data'][sensor_key]['data_file']
            collection['data'][sensor_key]['data'] = cv2.imread(filename)

    if not args['collection_selection_function'] is None:
        deleted = []
        for collection_key in dataset_sensors['collections'].keys():
            if not args['collection_selection_function'](collection_key):  # use the lambda expression csf
                deleted.append(collection_key)
                del dataset_sensors['collections'][collection_key]
        print("Deleted collections: " + str(deleted))

    if not args['use_incomplete_collections']:
        # Deleting collections where the chessboard was not found by both cameras:
        for collection_key, collection in dataset_sensors['collections'].items():
            for sensor_key, sensor in dataset_sensors['sensors'].items():
                if not collection['labels'][sensor_key]['detected']:
                    print(Fore.RED + "Removing collection " + collection_key + ' -> pattern was not found in sensor ' +
                          sensor_key + ' (must be found in all sensors).' + Style.RESET_ALL)
                    del dataset_sensors['collections'][collection_key]
                    break

    print("\nCollections studied:\n " + str(dataset_sensors['collections'].keys()))

    # ---------------------------------------
    # --- CREATE CHESSBOARD DATASET
    # ---------------------------------------
    dataset_sensors['chessboards'], dataset_chessboard_points = createChessBoardData(args, dataset_sensors)

    # ---------------------------------------
    # --- FILTER SOME OF THE ELEMENTS LOADED, TO USE ONLY A SUBSET IN THE CALIBRATION
    # ---------------------------------------
    if not args['sensor_selection_function'] is None:
        deleted = []
        for sensor_key in dataset_sensors['sensors'].keys():
            if not args['sensor_selection_function'](sensor_key):  # use the lambda expression ssf
                deleted.append(sensor_key)
                del dataset_sensors['sensors'][sensor_key]
        print("Deleted sensors: " + str(deleted))

    print('Loaded dataset containing ' + str(len(dataset_sensors['sensors'].keys())) + ' sensors and ' + str(
        len(dataset_sensors['collections'].keys())) + ' collections.')

    # ---------------------------------------
    # --- DETECT EDGES IN THE LASER SCANS
    # ---------------------------------------
    for sensor_key, sensor in dataset_sensors['sensors'].items():
        if sensor['msg_type'] == 'LaserScan':  # only for lasers
            for collection_key, collection in dataset_sensors['collections'].items():
                idxs = collection['labels'][sensor_key]['idxs']
                edges = []  # a list of edges
                for i in range(0, len(idxs) - 1):
                    if (idxs[i + 1] - idxs[i]) != 1:
                        edges.append(i)
                        edges.append(i + 1)

                # Remove first (right most) and last (left most) edges, since these are often false edges.
                if len(edges) > 0:
                    edges.pop(0)  # remove the first element.
                if len(edges) > 0:
                    edges.pop()  # if the index is not given, then the last element is popped out and removed.
                collection['labels'][sensor_key]['edge_idxs'] = edges

    # ---------------------------------------
    # --- Detect corners in velodyne data
    # ---------------------------------------
    for sensor_key, sensor in dataset_sensors['sensors'].items():
        if sensor['msg_type'] == 'PointCloud2':  # only for 3D Lidars and RGBD cameras
            for collection_key, collection in dataset_sensors['collections'].items():
                from rospy_message_converter import message_converter

                # Convert 3D cloud data on .json dictionary to ROS message type
                cloud_msg = message_converter.convert_dictionary_to_ros_message("sensor_msgs/PointCloud2",
                                                                                collection['data'][sensor_key])

                idxs = collection['labels'][sensor_key]['idxs']
                # pc = ros_numpy.numpify(cloud_msg)[idxs]
                # points = np.zeros((pc.shape[0], 4))
                # points[:, 0] = pc['x']
                # points[:, 1] = pc['y']
                # points[:, 2] = pc['z']
                # points[:, 3] = 1

                # TODO really compute the idxs


                collection['labels'][sensor_key]['idx_top_left'] = idxs[0]
                collection['labels'][sensor_key]['idx_top_right'] = idxs[1]
                collection['labels'][sensor_key]['idx_bottom_left'] = idxs[2]
                collection['labels'][sensor_key]['idx_bottom_right'] = idxs[3]

    # ---------------------------------------
    # --- SETUP OPTIMIZER
    # ---------------------------------------
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('args', args)
    opt.addDataModel('dataset_sensors', dataset_sensors)
    opt.addDataModel('dataset_chessboard_points', dataset_chessboard_points)

    # For the getters we only need to get one collection. Lets take the first key on the dictionary and always get that
    # transformation.
    selected_collection_key = dataset_sensors['collections'].keys()[0]

    # ------------  Sensors -----------------
    # Each sensor will have a position (tx,ty,tz) and a rotation (r1,r2,r3)

    # Add parameters related to the sensors
    translation_delta = 0.01

    # TODO temporary placement of top_left_camera
    # for collection_key, collection in dataset_sensors['collections'].items():
    #     collection['transforms']['base_link-top_left_camera']['trans'] = [-1.48, 0.22, 1.35]
    # dataset_sensors['calibration_config']['anchored_sensor'] = 'right_laser'
    print('Anchored sensor is ' + Fore.GREEN + dataset_sensors['calibration_config'][
        'anchored_sensor'] + Style.RESET_ALL)

    print('Creating parameters ...')
    for sensor_key, sensor in dataset_sensors['sensors'].items():

        # Translation -------------------------------------
        initial_translation = getterSensorTranslation(dataset_sensors, sensor_key=sensor_key,
                                                      collection_key=selected_collection_key)

        if sensor_key == dataset_sensors['calibration_config']['anchored_sensor']:
            bound_max = [x + sys.float_info.epsilon for x in initial_translation]
            bound_min = [x - sys.float_info.epsilon for x in initial_translation]
        else:
            # bound_max = [x + translation_delta for x in initial_translation]
            # bound_min = [x - translation_delta for x in initial_translation]
            bound_max = [+inf for x in initial_translation]
            bound_min = [-inf for x in initial_translation]
            if 'laser' in sensor_key:
                bound_max[2] = initial_translation[2] + translation_delta
                bound_min[2] = initial_translation[2] - translation_delta

        opt.pushParamVector(group_name='S_' + sensor_key + '_t', data_key='dataset_sensors',
                            getter=partial(getterSensorTranslation, sensor_key=sensor_key,
                                           collection_key=selected_collection_key),
                            setter=partial(setterSensorTranslation, sensor_key=sensor_key),
                            suffix=['x', 'y', 'z'],
                            bound_max=bound_max, bound_min=bound_min)

        # Rotation --------------------------------------
        initial_rotation = getterSensorRotation(dataset_sensors, sensor_key=sensor_key,
                                                collection_key=selected_collection_key)

        if sensor_key == dataset_sensors['calibration_config']['anchored_sensor']:
            bound_max = [x + sys.float_info.epsilon for x in initial_rotation]
            bound_min = [x - sys.float_info.epsilon for x in initial_rotation]
        else:
            bound_max = [+inf for x in initial_rotation]
            bound_min = [-inf for x in initial_rotation]

        opt.pushParamVector(group_name='S_' + sensor_key + '_r', data_key='dataset_sensors',
                            getter=partial(getterSensorRotation, sensor_key=sensor_key,
                                           collection_key=selected_collection_key),
                            setter=partial(setterSensorRotation, sensor_key=sensor_key),
                            suffix=['1', '2', '3'],
                            bound_max=bound_max, bound_min=bound_min)

        if sensor['msg_type'] == 'Image':  # if sensor is a camera add intrinsics
            # opt.pushParamVector(group_name='S_' + sensor_key + '_I_', data_key='dataset_sensors',
            #                     getter=partial(getterCameraIntrinsics, sensor_key=sensor_key),
            #                     setter=partial(setterCameraIntrinsics, sensor_key=sensor_key),
            #                     suffix=['fx', 'fy', 'cx', 'cy', 'd0', 'd1', 'd2', 'd3', 'd4'])
            opt.pushParamVector(group_name='S_' + sensor_key + '_P_', data_key='dataset_sensors',
                                getter=partial(getterCameraPMatrix, sensor_key=sensor_key),
                                setter=partial(setterCameraPMatrix, sensor_key=sensor_key),
                                suffix=['fx_p', 'fy_p', 'cx_p', 'cy_p'])

    # ------------  Chessboard -----------------
    # Each Chessboard will have the position (tx,ty,tz) and rotation (r1,r2,r3)

    # Add translation and rotation parameters related to the Chessboards
    for collection_key in dataset_sensors['chessboards']['collections']:
        # bound_max = [x + translation_delta for x in initial_values]
        # bound_min = [x - translation_delta for x in initial_values]

        # initial_translation = getterChessBoardTranslation(dataset_sensors,collection_key)
        # initial_translation[0] += 0.9
        # initial_translation[1] += 0.7
        # initial_translation[2] += 0.7
        # setterChessBoardTranslation(dataset_sensors, initial_translation, collection_key)

        opt.pushParamVector(group_name='C_' + collection_key + '_t', data_key='dataset_sensors',
                            getter=partial(getterChessBoardTranslation, collection_key=collection_key),
                            setter=partial(setterChessBoardTranslation, collection_key=collection_key),
                            suffix=['x', 'y', 'z'])
        # ,bound_max=bound_max, bound_min=bound_min)

        opt.pushParamVector(group_name='C_' + collection_key + '_r', data_key='dataset_sensors',
                            getter=partial(getterChessBoardRotation, collection_key=collection_key),
                            setter=partial(setterChessBoardRotation, collection_key=collection_key),
                            suffix=['1', '2', '3'])

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    # Each error is computed after the sensor and the chessboard of a collection. Thus, each error will be affected
    # by the parameters tx,ty,tz,r1,r2,r3 of the sensor and the chessboard

    print("Creating residuals ... ")
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][sensor_key]['detected']:  # if chessboard not detected by sensor in collection
                continue

            params = opt.getParamsContainingPattern('S_' + sensor_key)  # sensor related params
            params.extend(opt.getParamsContainingPattern('C_' + collection_key + '_'))  # chessboard related params

            if sensor['msg_type'] == 'Image':  # if sensor is a camera use four residuals
                # for idx in range(0, dataset_chessboards['number_corners']):
                for idx in range(0, 4):
                    opt.pushResidual(name=collection_key + '_' + sensor_key + '_' + str(idx), params=params)

            elif sensor['msg_type'] == 'LaserScan':  # if sensor is a 2D lidar add two residuals

                # Extrema points (longitudinal error)
                opt.pushResidual(name=collection_key + '_' + sensor_key + '_eleft', params=params)
                opt.pushResidual(name=collection_key + '_' + sensor_key + '_eright', params=params)

                # Inner points, use detection of edges (longitudinal error)
                for idx, _ in enumerate(collection['labels'][sensor_key]['edge_idxs']):
                    opt.pushResidual(name=collection_key + '_' + sensor_key + '_inner_' + str(idx), params=params)

                # Laser beam (orthogonal error)
                for idx in range(0, len(collection['labels'][sensor_key]['idxs'])):
                    opt.pushResidual(name=collection_key + '_' + sensor_key + '_beam_' + str(idx), params=params)

            elif sensor['msg_type'] == 'PointCloud2':  # if sensor is a 3D lidar add two residuals

                # Laser beam error
                for idx in range(0, len(collection['labels'][sensor_key]['idxs'])):
                    opt.pushResidual(name=collection_key + '_' + sensor_key + '_oe_' + str(idx), params=params)
                # Extrema displacement error
                for idx in range(0, 4):
                    opt.pushResidual(name=collection_key + '_' + sensor_key + '_cd_' + str(idx), params=params)

    # opt.printResiduals()

    # ---------------------------------------
    # --- Compute the SPARSE MATRIX
    # ---------------------------------------
    print("Computing sparse matrix ... ")
    opt.computeSparseMatrix()
    # opt.printSparseMatrix()

    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    if args['view_optimization']:
        opt.setInternalVisualization(True)
    else:
        opt.setInternalVisualization(False)

    if args['ros_visualization']:
        print("Configuring visualization ... ")
        graphics = setupVisualization(dataset_sensors, args)
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(dataset_graphics)
        opt.addDataModel('graphics', graphics)

        # opt.setVisualizationFunction(visualizationFunction, False, niterations=1, figures=[])
        opt.setVisualizationFunction(visualizationFunction, args['ros_visualization'], niterations=1, figures=[])

    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print('Initializing optimization ...')
    # opt.startOptimization(optimization_options={'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-4,
    #                                             'x_scale': 'jac'})
    opt.startOptimization(optimization_options={'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'diff_step': 1e-4,
                                                'x_scale': 'jac'})
    # TODO diff step as None

    # print('\n-----------------')
    # opt.printParameters(opt.x0, text='Initial parameters')
    # print('\n')
    # opt.printParameters(opt.xf, text='Final parameters')

    # ---------------------------------------
    # --- Save updated JSON file
    # ---------------------------------------
    # Write json file with updated dataset_sensors
    print('Saving json file ...')
    createJSONFile('test/sensor_pose_json_v2/results/dataset_sensors_results.json', dataset_sensors)
    print('File saved.')

    # # ---------------------------------------
    # # --- Save updated xacro
    # # ---------------------------------------
    # # Cycle all sensors in calibration config, and for each replace the optimized transform in the original xacro
    # print('Save updated xacro file ...')
    # for sensor_key in dataset_sensors['calibration_config']['sensors']:
    #     child = dataset_sensors['calibration_config']['sensors'][sensor_key]['child_link']
    #     parent = dataset_sensors['calibration_config']['sensors'][sensor_key]['parent_link']
    #     transform_key = parent + '-' + child
    #
    #     trans = list(dataset_sensors['collections'][selected_collection_key]['transforms'][transform_key]['trans'])
    #     quat = list(dataset_sensors['collections'][selected_collection_key]['transforms'][transform_key]['quat'])
    #     found = False
    #
    #     for joint in xml_robot.joints:
    #         if joint.parent == parent and joint.child == child:
    #             found = True
    #             print('Found joint: ' + str(joint.name))
    #
    #             print('Replacing xyz = ' + str(joint.origin.xyz) + ' by ' + str(trans))
    #             joint.origin.xyz = trans
    #
    #             rpy = list(tf.transformations.euler_from_quaternion(quat, axes='sxyz'))
    #             print('Replacing rpy = ' + str(joint.origin.rpy) + ' by ' + str(rpy))
    #             joint.origin.rpy = rpy
    #             break
    #
    #     if not found:
    #         raise ValueError('Could not find transform ' + str(transform_key) + ' in /robot_description')
    #
    # # TODO remove hardcoded xacro name
    # outfile = rospkg.RosPack().get_path('interactive_calibration') + '/calibrations/atlascar2/optimized.urdf.xacro'
    # with open(outfile, 'w') as out:
    #     print("Writing optimized urdf to " + str(outfile))
    #     out.write(URDF.to_xml_string(xml_robot))
    #
    # print('Xacro file saved.')


if __name__ == "__main__":
    main()
