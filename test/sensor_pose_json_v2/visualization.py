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
from rospy_message_converter import message_converter
from std_msgs.msg import Header, ColorRGBA

import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, sensor_msgs, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
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
    dataset_graphics['chessboard']['colormap'] = cm.plasma(np.linspace(0, 1,
                                                                       dataset_sensors['calibration_config'][
                                                                           'calibration_pattern']['dimension']['x'] *
                                                                       dataset_sensors['calibration_config'][
                                                                           'calibration_pattern']['dimension']['y']))

    dataset_graphics['collections']['colormap'] = cm.tab20(
        np.linspace(0, 1, len(dataset_sensors['collections'].keys())))
    for idx, collection_key in enumerate(sorted(dataset_sensors['collections'].keys())):
        dataset_graphics['collections'][str(collection_key)] = {
            'color': dataset_graphics['collections']['colormap'][idx, :]}

    color_map_sensors = cm.gist_rainbow(np.linspace(0, 1, len(dataset_sensors['sensors'].keys())))
    for idx, sensor_key in enumerate(sorted(dataset_sensors['sensors'].keys())):
        sensor_key = str(sensor_key)
        dataset_graphics['sensors'][sensor_key] = {'color': color_map_sensors[idx, :]}

    # Create image publishers ----------------------------------------------------------
    # We need to republish a new image at every visualization
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, _sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][str(sensor_key)]['detected']:  # not detected by sensor in collection
                continue

            if _sensor['msg_type'] == 'Image':
                msg_type = sensor_msgs.msg.Image
                topic_name = 'c' + str(collection_key) + '/' + str(sensor_key) + '/image_raw'
                dataset_graphics['collections'][collection_key][str(sensor_key)] = {'publisher': rospy.Publisher(
                    topic_name, msg_type, queue_size=0, latch=True)}

                msg_type = sensor_msgs.msg.CameraInfo
                topic_name = 'c' + str(collection_key) + '/' + str(sensor_key) + '/camera_info'
                dataset_graphics['collections'][collection_key][str(sensor_key)]['publisher_camera_info'] = \
                    rospy.Publisher(topic_name, msg_type, queue_size=0, latch=True)

    # Create Lasers MarkerArray -----------------------------------------------------------
    markers = MarkerArray()
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, _sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][str(sensor_key)]['detected']:  # not detected by sensor in collection
                continue

            if _sensor['msg_type'] == 'LaserScan':

                marker = Marker(header=Header(frame_id=str(sensor_key), stamp=rospy.Time.now()),
                                ns=str(collection_key) + '-' + str(sensor_key), id=0, frame_locked=True,
                                type=Marker.POINTS, action=Marker.ADD, lifetime=rospy.Duration(0),
                                pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                                scale=Vector3(x=0.03, y=0.03, z=0),
                                color=ColorRGBA(r=dataset_graphics['collections'][collection_key]['color'][0],
                                                g=dataset_graphics['collections'][collection_key]['color'][1],
                                                b=dataset_graphics['collections'][collection_key]['color'][2], a=1.0)
                                )

                # Get laser points that belong to the chessboard (labelled)
                idxs = collection['labels'][sensor_key]['idxs']
                rhos = [collection['data'][sensor_key]['ranges'][idx] for idx in idxs]
                thetas = [collection['data'][sensor_key]['angle_min'] +
                          collection['data'][sensor_key]['angle_increment'] * idx for idx in idxs]

                for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                    p = Point()
                    p.z = 0
                    p.x = rho * math.cos(theta)
                    p.y = rho * math.sin(theta)
                    marker.points.append(p)

                markers.markers.append(copy.deepcopy(marker))

                # Draw extrema points
                marker.ns = str(collection_key) + '-' + str(sensor_key)
                marker.type = Marker.SPHERE_LIST
                marker.id = 1
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 0.5
                marker.points = [marker.points[0], marker.points[-1]]

                markers.markers.append(copy.deepcopy(marker))

                # Draw detected edges
                marker.ns = str(collection_key) + '-' + str(sensor_key)
                marker.type = Marker.CUBE_LIST
                marker.id = 2
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 0.5

                marker.points = []  # Reset the list of marker points
                for edge_idx in collection['labels'][sensor_key]['edge_idxs']:  # add edge points
                    p = Point()
                    p.x = rhos[edge_idx] * math.cos(thetas[edge_idx])
                    p.y = rhos[edge_idx] * math.sin(thetas[edge_idx])
                    p.z = 0
                    marker.points.append(p)
                markers.markers.append(copy.deepcopy(marker))

    dataset_graphics['ros']['MarkersLaserScans'] = markers
    dataset_graphics['ros']['PubLaserScans'] = rospy.Publisher('LaserScans', MarkerArray, queue_size=0, latch=True)

    # Create LaserBeams Publisher -----------------------------------------------------------
    # This one is recomputed every time in the objective function, so just create the generic properties.
    markers = MarkerArray()

    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][sensor_key]['detected']:  # chess not detected by sensor in collection
                continue
            if sensor['msg_type'] == 'LaserScan':
                marker = Marker(header=Header(frame_id=str(sensor_key), stamp=rospy.Time.now()),
                                ns=str(collection_key) + '-' + str(sensor_key), id=0, frame_locked=True,
                                type=Marker.LINE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
                                pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                                scale=Vector3(x=0.003, y=0, z=0),
                                color=ColorRGBA(r=dataset_graphics['collections'][collection_key]['color'][0],
                                                g=dataset_graphics['collections'][collection_key]['color'][1],
                                                b=dataset_graphics['collections'][collection_key]['color'][2], a=1.0)
                                )
                markers.markers.append(marker)

    dataset_graphics['ros']['MarkersLaserBeams'] = markers
    dataset_graphics['ros']['PubLaserBeams'] = rospy.Publisher('LaserBeams', MarkerArray, queue_size=0, latch=True)

    # Create Chessboards MarkerArray -----------------------------------------------------------

    markers = MarkerArray()
    now = rospy.Time.now()
    epts = dataset_sensors['chessboards']['evaluation_points']
    sx = dataset_sensors['chessboards']['chess_num_x']
    sy = dataset_sensors['chessboards']['chess_num_y']
    square_size = dataset_sensors['chessboards']['square_size']
    for collection_chess_key, collection_chess in dataset_sensors['chessboards']['collections'].items():

        # Chessboard drawing (just visual with lines)
        marker = Marker(header=Header(frame_id='chessboard_' + collection_chess_key, stamp=now),
                        ns=str(collection_chess_key), id=0, frame_locked=True,
                        type=Marker.LINE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
                        pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                        scale=Vector3(x=0.005, y=0, z=0),
                        color=ColorRGBA(r=dataset_graphics['collections'][collection_chess_key]['color'][0],
                                        g=dataset_graphics['collections'][collection_chess_key]['color'][1],
                                        b=dataset_graphics['collections'][collection_chess_key]['color'][2], a=1.0))

        w = sx - 1
        for idx in range(0, sy):  # visit all rows and draw an horizontal line for each
            marker.points.append(Point(x=epts[0, sx * idx], y=epts[1, sx * idx], z=epts[2, sx * idx]))
            marker.points.append(
                Point(x=epts[0, sx * idx + w], y=epts[1, sx * idx + w], z=epts[2, sx * idx + w]))

        h = (sy - 1) * sx
        for idx in range(0, sx):  # visit all columns and draw a vertical line for each
            marker.points.append(Point(x=epts[0, idx], y=epts[1, idx], z=epts[2, idx]))
            marker.points.append(Point(x=epts[0, idx + h], y=epts[1, idx + h], z=epts[2, idx + h]))

        # Pattern physical canvas: Top Line
        marker.points.append(Point(x=-1 * square_size, y=-1 * square_size, z=0))
        marker.points.append(Point(x=sx * square_size, y=-1 * square_size, z=0))

        # Pattern physical canvas: Bottom Line
        marker.points.append(Point(x=-1 * square_size, y=sy * square_size, z=0))
        marker.points.append(Point(x=sx * square_size, y=sy * square_size, z=0))

        # Pattern physical canvas: Left Line
        p = Point(x=-1 * square_size, y=-1 * square_size, z=0)
        marker.points.append(p)
        p = Point(x=-1 * square_size, y=sy * square_size, z=0)
        marker.points.append(p)

        # Pattern physical canvas: Right Line
        p = Point(x=sx * square_size, y=-1 * square_size, z=0)
        marker.points.append(p)
        p = Point(x=sx * square_size, y=sy * square_size, z=0)
        marker.points.append(p)

        markers.markers.append(marker)

        # Draw limit points used to compute the longitudinal distance
        marker = Marker(header=Header(frame_id='chessboard_' + collection_chess_key, stamp=now),
                        ns=str(collection_chess_key), id=1, frame_locked=True,
                        type=Marker.SPHERE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
                        pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                        scale=Vector3(x=0.015, y=0.015, z=0.015),
                        color=ColorRGBA(r=dataset_graphics['collections'][collection_chess_key]['color'][0],
                                        g=dataset_graphics['collections'][collection_chess_key]['color'][1],
                                        b=dataset_graphics['collections'][collection_chess_key]['color'][2], a=1.0))

        for idx in range(0, dataset_sensors['chessboards']['limit_points'].shape[1]):
            marker.points.append(Point(x=dataset_sensors['chessboards']['limit_points'][0, idx],
                                       y=dataset_sensors['chessboards']['limit_points'][1, idx],
                                       z=dataset_sensors['chessboards']['limit_points'][2, idx]))

        markers.markers.append(marker)

        # Draw inner points
        marker = Marker(header=Header(frame_id='chessboard_' + collection_chess_key, stamp=now),
                        ns=str(collection_chess_key), id=2, frame_locked=True,
                        type=Marker.SPHERE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
                        pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                        scale=Vector3(x=0.025, y=0.025, z=0.025),
                        color=ColorRGBA(r=dataset_graphics['collections'][collection_chess_key]['color'][0],
                                        g=dataset_graphics['collections'][collection_chess_key]['color'][1],
                                        b=dataset_graphics['collections'][collection_chess_key]['color'][2],
                                        a=1.0))

        for idx in range(0, dataset_sensors['chessboards']['inner_points'].shape[1]):
            marker.points.append(Point(x=dataset_sensors['chessboards']['inner_points'][0, idx],
                                       y=dataset_sensors['chessboards']['inner_points'][1, idx],
                                       z=dataset_sensors['chessboards']['inner_points'][2, idx]))

        markers.markers.append(marker)

    dataset_graphics['ros']['MarkersChessboards'] = markers
    dataset_graphics['ros']['PubChessboards'] = rospy.Publisher('Chessboards', MarkerArray, queue_size=0, latch=True)

    # Create Miscellaneous MarkerArray -----------------------------------------------------------
    markers = MarkerArray()

    # Text signaling the anchored sensor
    for _sensor_key, sensor in dataset_sensors['sensors'].items():
        if _sensor_key == dataset_sensors['calibration_config']['anchored_sensor']:
            marker = Marker(header=Header(frame_id=str(_sensor_key), stamp=now),
                            ns=str(_sensor_key), id=0, frame_locked=True,
                            type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, lifetime=rospy.Duration(0),
                            pose=Pose(position=Point(x=0, y=0, z=0.2), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                            scale=Vector3(x=0.0, y=0.0, z=0.1),
                            color=ColorRGBA(r=0.6, g=0.6, b=0.6,a=1.0), text='Anchored')

            markers.markers.append(marker)

    dataset_graphics['ros']['MarkersMiscellaneous'] = markers
    dataset_graphics['ros']['PubMiscellaneous'] = rospy.Publisher('Miscellaneous', MarkerArray, queue_size=0,
                                                                  latch=True)
    # Publish only once in latched mode
    dataset_graphics['ros']['PubMiscellaneous'].publish(dataset_graphics['ros']['MarkersMiscellaneous'])

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
    selected_collection_key = dataset_sensors['collections'].keys()[0]
    for sensor_key, sensor in dataset_sensors['sensors'].items():
        transform_key = sensor['calibration_parent'] + '-' + sensor['calibration_child']
        transform = dataset_sensors['collections'][selected_collection_key]['transforms'][transform_key]
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
    now = rospy.Time.now()
    for marker in dataset_graphics['ros']['MarkersLaserScans'].markers:
        marker.header.stamp = now
    dataset_graphics['ros']['PubLaserScans'].publish(dataset_graphics['ros']['MarkersLaserScans'])
    dataset_graphics['ros']['PubLaserScans'].publish(dataset_graphics['ros']['MarkersLaserScans'])

    # Publish Laser Beams
    now = rospy.Time.now()
    for marker in dataset_graphics['ros']['MarkersLaserBeams'].markers:
        marker.header.stamp = now
    dataset_graphics['ros']['PubLaserBeams'].publish(dataset_graphics['ros']['MarkersLaserBeams'])

    # Publish Chessboards
    now = rospy.Time.now()
    for marker in dataset_graphics['ros']['MarkersChessboards'].markers:
        marker.header.stamp = now

    dataset_graphics['ros']['PubChessboards'].publish(dataset_graphics['ros']['MarkersChessboards'])

    # Publish Annotated images
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():

            if not collection['labels'][sensor_key]['detected']:  # not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'Image':
                if args['show_images']:
                    image = copy.deepcopy(collection['data'][sensor_key]['data'])
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

                    msg = CvBridge().cv2_to_imgmsg(image, "bgr8")
                    msg.header.frame_id = sensor_key + '_optical'  # TODO should be automated
                    dataset_graphics['collections'][collection_key][sensor_key]['publisher'].publish(msg)

                    # Publish camera info message
                    camera_info_msg = CameraInfo()
                    camera_info_msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/CameraInfo',
                                                                                          sensor['camera_info'])
                    dataset_graphics['collections'][collection_key][sensor_key]['publisher_camera_info'].publish(
                        camera_info_msg)

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
