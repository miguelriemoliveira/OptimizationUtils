#!/usr/bin/env python

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import copy
import cv2
import math
import rospy
import urdf_parser_py
from rospy_message_converter import message_converter
from std_msgs.msg import Header, ColorRGBA

from rospy_urdf_to_rviz_converter.rospy_urdf_to_rviz_converter import urdfToMarkerArray
from tf import transformations
from urdf_parser_py.urdf import URDF

import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, sensor_msgs, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
from matplotlib import cm
from open3d import *
from OptimizationUtils import utilities


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


def setupVisualization(dataset, args):
    """
    Creates the necessary variables in a dictionary "dataset_graphics", which will be passed onto the visualization
    function
    """

    # Create a python dictionary that will contain all the visualization related information
    graphics = {'collections': {}, 'sensors': {}, 'pattern': {}, 'ros': {}, 'args': args}

    # Initialize ROS stuff
    rospy.init_node("optimization_node")
    graphics['ros']['tf_broadcaster'] = tf.TransformBroadcaster()
    graphics['ros']['publisher_models'] = rospy.Publisher('robot_meshes', MarkerArray, queue_size=0, latch=True)

    # Parse robot description from the ros parameter '/robot_description'
    # TODO the ros xacro file could be stored in the json file for usage here
    rospy.loginfo('Reading xml xacro file ...')
    xml_robot = URDF.from_parameter_server()


    pattern = dataset['calibration_config']['calibration_pattern']
    graphics['pattern']['colormap'] = cm.plasma(
        np.linspace(0, 1, pattern['dimension']['x'] * pattern['dimension']['y']))

    # Create colormaps to be used for coloring the elements. Each collection contains a color, each sensor likewise.
    graphics['collections']['colormap'] = cm.tab20b(np.linspace(0, 1, len(dataset['collections'].keys())))
    for idx, collection_key in enumerate(sorted(dataset['collections'].keys())):
        graphics['collections'][str(collection_key)] = {'color': graphics['collections']['colormap'][idx, :]}

    # color_map_sensors = cm.gist_rainbow(np.linspace(0, 1, len(dataset['sensors'].keys())))
    # for idx, sensor_key in enumerate(sorted(dataset['sensors'].keys())):
    #     dataset['sensors'][str(sensor_key)]['color'] = color_map_sensors[idx, :]

    # Create the markers array for visualizing the robot meshes on all collections
    markers = MarkerArray()
    for collection_key, collection in dataset['collections'].items():
        print("Collection : " + str(collection_key))
        rgba = graphics['collections'][collection_key]['color']
        # rgba[3] = 0.3 # change the alpha
        m = urdfToMarkerArray(xml_robot, frame_id_prefix='c' + collection_key + '_', namespace=collection_key,
                              rgba=rgba)



    # Draw the chessboard
    for collection_key, collection in dataset['collections'].items():
        m = Marker(header=Header(frame_id='c' + collection_key + '_chessboard_link', stamp=rospy.Time.now()),
                   ns='c' + collection_key + '_', id=999, frame_locked=True,
                   type=Marker.MESH_RESOURCE, action=Marker.ADD, lifetime=rospy.Duration(0),
                   pose=Pose(position=Point(x=0, y=0, z=0),
                             orientation=Quaternion(x=0, y=0, z=0, w=1)),
                   scale=Vector3(x=1.0, y=1.0, z=1.0),
                   color=ColorRGBA(r=1, g=1, b=1, a=1))
        m.mesh_resource = 'package://interactive_calibration/meshes/charuco_5x5.dae'
        m.mesh_use_embedded_materials = True
        markers.markers.append(m)
        break

    graphics['ros']['robot_mesh_markers'] = markers

    # Create image publishers ----------------------------------------------------------
    # We need to republish a new image at every visualization
    for collection_key, collection in dataset['collections'].items():
        for sensor_key, sensor in dataset['sensors'].items():
            if not collection['labels'][str(sensor_key)]['detected']:  # not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'Image':
                msg_type = sensor_msgs.msg.Image
                topic_name = 'c' + str(collection_key) + '/' + str(sensor_key) + '/image_raw'
                graphics['collections'][collection_key][str(sensor_key)] = {'publisher': rospy.Publisher(
                    topic_name, msg_type, queue_size=0, latch=True)}

                msg_type = sensor_msgs.msg.CameraInfo
                topic_name = 'c' + str(collection_key) + '/' + str(sensor_key) + '/camera_info'
                graphics['collections'][collection_key][str(sensor_key)]['publisher_camera_info'] = \
                    rospy.Publisher(topic_name, msg_type, queue_size=0, latch=True)

    return graphics


def visualizationFunction(models):
    # Get the data from the models
    args = models['args']
    collections = models['dataset']['collections']
    sensors = models['dataset']['sensors']
    pattern = models['dataset']['pattern']
    config = models['dataset']['calibration_config']
    graphics = models['graphics']

    now = rospy.Time.now()  # time used to publish all visualization messages

    for collection_key, collection in collections.items():

        # To have a fully connected tree, must connect the instances of the tf tree of every collection into a single
        # tree. We do this by publishing an identity transform between the configured world link and hte world link
        # of each collection.
        parent = config['world_link']
        child = 'c' + collection_key + '_' + parent
        graphics['ros']['tf_broadcaster'].sendTransform((0, 0, 0), (0, 0, 0, 1), now, child, parent)

        # Publish all current transforms
        for transform_key, transform in collection['transforms'].items():
            # TODO after https://github.com/lardemua/AtlasCarCalibration/issues/54 this will be unnecessary
            parent = 'c' + collection_key + '_' + transform_key.split('-')[0]
            child = 'c' + collection_key + '_' + transform_key.split('-')[1]
            graphics['ros']['tf_broadcaster'].sendTransform(transform['trans'], transform['quat'], now, child, parent)

    # Update markers stamp, so that rviz uses newer transforms to compute their poses.
    for marker in graphics['ros']['robot_mesh_markers'].markers:
        marker.header.stamp = now

    # Publish the models
    graphics['ros']['publisher_models'].publish(graphics['ros']['robot_mesh_markers'])

    # Publish Annotated images
    for collection_key, collection in collections.items():
        for sensor_key, sensor in sensors.items():

            if not collection['labels'][sensor_key]['detected']:  # not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'Image':
                if args['show_images']:
                    image = copy.deepcopy(collection['data'][sensor_key]['data'])
                    width = collection['data'][sensor_key]['width']
                    height = collection['data'][sensor_key]['height']
                    diagonal = math.sqrt(width ** 2 + height ** 2)
                    cm = graphics['pattern']['colormap']

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

                    msg.header.frame_id = 'c' + collection_key + '_' + sensor['parent']
                    graphics['collections'][collection_key][sensor_key]['publisher'].publish(msg)

                    # Publish camera info message
                    camera_info_msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/CameraInfo',
                                                                                          sensor['camera_info'])
                    camera_info_msg.header.frame_id = msg.header.frame_id
                    graphics['collections'][collection_key][sensor_key]['publisher_camera_info'].publish(
                        camera_info_msg)
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
