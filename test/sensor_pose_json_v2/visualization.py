#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import copy
import cv2
import math
import os
import pprint

import rospy
import tf
import visualization_msgs
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, sensor_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import KeyPressManager.KeyPressManager as KeyPressManager
import matplotlib.pyplot as plt
from matplotlib import cm
from open3d import *

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities


def setupVisualization(dataset_sensors, args):
    """
    Creates the necessary variables in a dictionary "dataset_graphics", which will be passed onto the visualization
    function
    """

    # Create a python dictionary that will contain all the visualization related information
    dataset_graphics = {'collections': {}, 'sensors': {}, 'chessboard': {}, 'ros': {}, 'args': args}
    for collection_key, collection in dataset_sensors['collections'].items():
        dataset_graphics['collections'][str(collection_key)] = {}  # Create a new key for each collection

    # Initialize ROS stuff
    rospy.init_node("optimization_node")
    dataset_graphics['ros']['tf_broadcaster'] = tf.TransformBroadcaster()

    # Create colormaps to be used for colloring the elements. Each collection contains a color, each sensor likewise.
    dataset_graphics['chessboard']['colormap'] = cm.plasma(np.linspace(0, 1, args['chess_num_x'] * args['chess_num_y']))

    dataset_graphics['collections']['colormap'] = cm.Set3(np.linspace(0, 1, len(dataset_sensors['collections'].keys())))
    for row_idx, collection_key in enumerate(sorted(dataset_sensors['collections'].keys())):
        dataset_graphics['collections'][str(collection_key)] = {
            'color': dataset_graphics['collections']['colormap'][row_idx, :]}

    color_map_sensors = cm.gist_rainbow(np.linspace(0, 1, len(dataset_sensors['sensors'].keys())))
    for row_idx, sensor_key in enumerate(sorted(dataset_sensors['sensors'].keys())):
        sensor_key = str(sensor_key)
        dataset_graphics['sensors'][sensor_key] = {'color': color_map_sensors[row_idx, :]}

    # Create image publishers ----------------------------------------------------------
    # We need to republish a new image at every visualization
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, _sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][str(sensor_key)]['detected']:  # not detected by sensor in collection
                continue

            if _sensor['msg_type'] == 'Image':
                msg_type = sensor_msgs.msg.Image
                topic_name = 'c' + str(collection_key) + '/' + str(sensor_key)
                dataset_graphics['collections'][collection_key][str(sensor_key)] = {'publisher': rospy.Publisher(
                    topic_name, msg_type, queue_size=0, latch=True)}

    # Create Lasers MarkerArray -----------------------------------------------------------
    markers = MarkerArray()
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, _sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][str(sensor_key)]['detected']:  # not detected by sensor in collection
                continue

            if _sensor['msg_type'] == 'LaserScan':

                marker = Marker()
                marker.header.frame_id = sensor_key
                marker.header.stamp = rospy.Time.now()
                marker.ns = str(collection_key) + '-' + str(sensor_key)
                marker.id = 0
                marker.frame_locked = True
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.lifetime = rospy.Duration(0)
                marker.pose.position.x = 0
                marker.pose.position.y = 0
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.0

                marker.color.r = dataset_graphics['collections'][collection_key]['color'][0]
                marker.color.g = dataset_graphics['collections'][collection_key]['color'][1]
                marker.color.b = dataset_graphics['collections'][collection_key]['color'][2]
                marker.color.a = 1.0

                # Get laser points that belong to the chessboard
                idxs = collection['labels'][sensor_key]['idxs']
                rhos = [collection['data'][sensor_key]['ranges'][row_idx] for row_idx in idxs]
                thetas = [collection['data'][sensor_key]['angle_min'] +
                          collection['data'][sensor_key]['angle_increment'] * row_idx for row_idx in idxs]

                for row_idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                    p = Point()
                    p.z = 0
                    p.x = rho * math.cos(theta)
                    p.y = rho * math.sin(theta)
                    marker.points.append(p)

                markers.markers.append(marker)

    dataset_graphics['ros']['MarkersLaserScans'] = markers
    dataset_graphics['ros']['PubLaserScans'] = rospy.Publisher('LaserScans', MarkerArray, queue_size=0, latch=True)

    # Create Chessboards MarkerArray -----------------------------------------------------------

    # a general drawing of a chessboard. Will be replicated for each collection based on each
    # collections's chessboard pose. Fields ".header.frame_id" to be filled for each collection
    marker = Marker()
    marker.id = 0
    marker.frame_locked = True
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(0)
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.01

    pts = dataset_sensors['chessboards']['evaluation_points']
    num_x = dataset_sensors['chessboards']['chess_num_x']
    num_y = dataset_sensors['chessboards']['chess_num_y']

    for row_idx in range(0, num_y):  # visit all rows and draw an horizontal line for each
        p = Point()
        p.x = pts[0, num_x * row_idx]
        p.y = pts[1, num_x * row_idx]
        p.z = pts[2, num_x * row_idx]
        marker.points.append(p)
        p = Point()
        p.x = pts[0, num_x * row_idx + num_x - 1]
        p.y = pts[1, num_x * row_idx + num_x - 1]
        p.z = pts[2, num_x * row_idx + num_x - 1]
        marker.points.append(p)

    for col_idx in range(0, num_x):  # visit all columns and draw a vertical line for each
        p = Point()
        p.x = pts[0, col_idx]
        p.y = pts[1, col_idx]
        p.z = pts[2, col_idx]
        marker.points.append(p)
        p = Point()
        p.x = pts[0, col_idx + (num_y-1) * num_x]
        p.y = pts[1, col_idx + (num_y-1) * num_x]
        p.z = pts[2, col_idx + (num_y-1) * num_x]
        marker.points.append(p)

    # Create a marker array for drawing a chessboard for each collection
    markers = MarkerArray()
    now = rospy.Time.now()
    for collection_chess_key, collection_chess in dataset_sensors['chessboards']['collections'].items():
        marker.header.frame_id = 'chessboard_' + collection_chess_key
        marker.header.stamp = now
        marker.ns = str(collection_chess_key)
        marker.color.r = dataset_graphics['collections'][collection_chess_key]['color'][0]
        marker.color.g = dataset_graphics['collections'][collection_chess_key]['color'][1]
        marker.color.b = dataset_graphics['collections'][collection_chess_key]['color'][2]
        marker.color.a = 1.0
        markers.markers.append(copy.deepcopy(marker))

    dataset_graphics['ros']['MarkersChessboards'] = markers
    dataset_graphics['ros']['PubChessboards'] = rospy.Publisher('Chessboards', MarkerArray, queue_size=0, latch=True)

    return dataset_graphics


def visualizationFunction(data):
    # Get the data from the model
    dataset_sensors = data['dataset_sensors']
    dataset_chessboard = data['dataset_sensors']['chessboards']
    dataset_graphics = data['dataset_graphics']
    args = dataset_graphics['args']

    now = rospy.Time.now()

    # Publishes all tfs contained in the json
    # for _collection_key, _collection in _dataset_sensors['collections'].items():
    #     for _transform_key, _transform in _collection['transforms'].items():
    #         # TODO after https://github.com/lardemua/AtlasCarCalibration/issues/54 this will be unnecessary
    #         parent = _transform_key.split('-')[0]
    #         child = _transform_key.split('-')[1]
    #         br.sendTransform(_transform['trans'], _transform['quat'], now, child, parent)
    #
    #     break # All collections have the same transforms, so we only need to publish one collection

    # Publishes only the tfs which are being calibrated. Better than above, but requires a state_publisher to be
    # launched in parallel Draw sensor poses (use sensor pose from collection '0' since they are all the same)
    for sensor_key, sensor in dataset_sensors['sensors'].items():
        transform_key = sensor['calibration_parent'] + '-' + sensor['calibration_child']
        transform = dataset_sensors['collections']['0']['transforms'][transform_key]
        dataset_graphics['ros']['tf_broadcaster'].sendTransform(transform['trans'], transform['quat'],
                                                                now, sensor['calibration_child'],
                                                                sensor['calibration_parent'])

    # Publishes the chessboards transforms
    for idx, (collection_chess_key, collection_chess) in enumerate(dataset_chessboard['collections'].items()):
        parent = 'base_link'
        child = 'chessboard_' + collection_chess_key
        dataset_graphics['ros']['tf_broadcaster'].sendTransform(collection_chess['trans'], collection_chess['quat'],
                                                                now, child, parent)

    # Publish Laser Scans
    dataset_graphics['ros']['PubLaserScans'].publish(dataset_graphics['ros']['MarkersLaserScans'])

    # Publish Chessboards
    dataset_graphics['ros']['PubChessboards'].publish(dataset_graphics['ros']['MarkersChessboards'])

    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():

            if not collection['labels'][sensor_key]['detected']:  # not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'Image':
                if args['show_images']:
                    filename = os.path.dirname(args['json_file']) + '/' + collection['data'][sensor_key][
                        'data_file']

                    # TODO should not read image again from disk
                    image = cv2.imread(filename)
                    width = collection['data'][sensor_key]['width']
                    height = collection['data'][sensor_key]['height']
                    diagonal = math.sqrt(width ** 2 + height ** 2)
                    cm = dataset_graphics['chessboard']['colormap']

                    # Draw projected points (as dots)
                    for idx, point in enumerate(collection['labels'][sensor_key]['idxs_projected']):
                        x = int(round(point['x']))
                        y = int(round(point['y']))
                        color = (cm[idx, 2] * 255, cm[idx, 1] * 255, cm[idx, 0] * 255)
                        cv2.line(image, (x, y), (x, y), color, int(6E-3 * diagonal))

                    # Draw ground truth points (as squares)
                    for idx, point in enumerate(collection['labels'][sensor_key]['idxs']):
                        x = int(round(point['x']))
                        y = int(round(point['y']))
                        color = (cm[idx, 2] * 255, cm[idx, 1] * 255, cm[idx, 0] * 255)
                        utilities.drawSquare2D(image, x, y, int(8E-3 * diagonal), color=color, thickness=2)

                    # Draw initial projected points (as crosses)
                    for idx, point in enumerate(collection['labels'][sensor_key]['idxs_initial']):
                        x = int(round(point['x']))
                        y = int(round(point['y']))
                        color = (cm[idx, 2] * 255, cm[idx, 1] * 255, cm[idx, 0] * 255)
                        utilities.drawCross2D(image, x, y, int(8E-3 * diagonal), color=color, thickness=1)

                    window_name = sensor_key + '-' + collection_key

                    # cv2.imshow(window_name, image)
                    msg = CvBridge().cv2_to_imgmsg(image, "bgr8")

                    dataset_graphics['collections'][collection_key][sensor_key]['publisher'].publish(msg)
                    # print('publishing image')

            elif sensor['msg_type'] == 'LaserScan':
                pass

            else:
                raise ValueError("Unknown sensor msg_type")

    # color_collection = color_map_collections[idx, :]
    # # Draw chessboard poses
    # for idx, (_collection_key, _collection) in enumerate(_dataset_chessboard['collections'].items()):
    #     root_T_chessboard = utilities.translationQuaternionToTransform(_collection['trans'], _collection['quat'])
    #     color_collection = color_map_collections[idx, :]
    #     utilities.drawChessBoard(ax, root_T_chessboard, chessboard_points, 'C' + _collection_key,
    #                              chess_num_x=args['chess_num_x'], chess_num_y=args['chess_num_y'],
    #                              color=color_collection, axis_scale=0.3, line_width=2,
    #                              handles=dataset_graphics['collections'][_collection_key]['handle'])
    #
    #     # Transform limit points to root
    #     pts_l_chess_root = np.dot(root_T_chessboard, pts_l_chess)
    #
    #     utilities.drawPoints3D(ax, None, pts_l_chess_root, line_width=1.0,
    #                            handles=
    #                            dataset_graphics['collections'][_collection_key]['limit_handle'])