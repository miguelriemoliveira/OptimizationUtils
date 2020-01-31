#!/usr/bin/env python

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import copy
import cv2
import math
import os
import pprint

import rospy
import urdf_parser_py
from rospy_message_converter import message_converter
from std_msgs.msg import Header, ColorRGBA

from rospy_urdf_to_rviz_converter.rospy_urdf_to_rviz_converter import robot_description_to_marker_array
from tf import transformations
from urdf_parser_py.urdf import URDF

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


def setupVisualization(dataset, args):
    """
    Creates the necessary variables in a dictionary "dataset_graphics", which will be passed onto the visualization
    function
    """

    # Create a python dictionary that will contain all the visualization related information
    graphics = {'collections': {}, 'sensors': {}, 'chessboard': {}, 'ros': {}, 'args': args}
    # for collection_key, collection in dataset_sensors['collections'].items():
    #     dataset_graphics['collections'][str(collection_key)] = {}  # Create a new key for each collection
    #

    # markers2 = robot_description_to_marker_array(_robot_description='/robot_description', _frame_id_prefix='', _namespace=None,
    #                                   _rgba=None)

    # Initialize ROS stuff
    rospy.init_node("optimization_node")
    graphics['ros']['tf_broadcaster'] = tf.TransformBroadcaster()

    markers = MarkerArray()
    graphics['ros']['publisher_models'] = rospy.Publisher('test_meshes', MarkerArray, queue_size=0, latch=True)

    # Parse robot description from param /robot_description
    rospy.loginfo('Reading xml xacro file ...')
    xml_robot = URDF.from_parameter_server()
    # xml_robot2 = URDF.from_xml_file('/home/mike/catkin_ws/src/AtlasCarCalibration/interactive_calibration/calibrations/ur10e'
    #                    '/eye_in_hand_chess/ur10e.urdf.xacro')
    # URDF.from_xml('/home/mike/catkin_ws/src/AtlasCarCalibration/interactive_calibration/calibrations/ur10e'
    #                    '/eye_in_hand_chess/ur10e.urdf.xacro',)
    # URDF.fr


    # print(xml_robot)
    # print('-----------------------------------')
    # print(xml_robot2)
    # exit(0)

    # Create colormaps to be used for coloring the elements. Each collection contains a color, each sensor likewise.
    graphics['collections']['colormap'] = cm.tab20b(np.linspace(0, 1, len(dataset['collections'].keys())))
    for idx, collection_key in enumerate(sorted(dataset['collections'].keys())):
        graphics['collections'][str(collection_key)] = {'color': graphics['collections']['colormap'][idx, :]}

    for collection_key, collection in dataset['collections'].items():
        print("Collection : " + str(collection_key))
        count = 0
        for link in xml_robot.links:
            print("Analysing link: " + str(link.name))
            print(link.name + ' has ' + str(len(link.visuals)) + ' visuals.')

            for visual in link.visuals:  # iterate through all visuals in the list
                if not visual.geometry:
                    raise ValueError("Link name " + link.name + " contains visual without geometry.")
                else:
                    geometry = visual.geometry

                print("visual: " + str(visual))
                x = y = z = 0  # origin xyz default values
                qx = qy = qz = 0
                qw = 1  # default rotation values

                if visual.origin:  # check if there is an origin
                    x = visual.origin.xyz[0]
                    y = visual.origin.xyz[1]
                    z = visual.origin.xyz[2]

                    q = transformations.quaternion_from_euler(visual.origin.rpy[0],
                                                              visual.origin.rpy[1],
                                                              visual.origin.rpy[2],
                                                              axes='sxyz')
                    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

                if visual.material:
                    r,g,b,a = visual.material.color.rgba

                r,g,b,a = graphics['collections'][str(collection_key)]['color']
                a=0.3

                # define the frame_id name per collection. Add a prefix based on the collection key
                # TODO base_link should not be hardcoded
                frame_id = 'c' + collection_key + '_' + link.name

                # Test geometry type
                if isinstance(geometry, urdf_parser_py.urdf.Mesh):
                    print("Visual.geometry of type urdf_parser_py.urdf.Mesh")

                    m = Marker(header=Header(frame_id=str(frame_id), stamp=rospy.Time.now()),
                               ns=str('c' + collection_key), id=count, frame_locked=True,
                               type=Marker.MESH_RESOURCE, action=Marker.ADD, lifetime=rospy.Duration(0),
                               pose=Pose(position=Point(x=x, y=y, z=z),
                                         orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)),
                               scale=Vector3(x=1.0, y=1.0, z=1.0),
                               color=ColorRGBA(r=r, g=g, b=b, a=a))
                    m.mesh_resource = geometry.filename
                    m.mesh_use_embedded_materials = False
                    markers.markers.append(m)
                    count += 1
                elif isinstance(geometry, urdf_parser_py.urdf.Box):
                    print("Visual.geometry of type urdf_parser_py.urdf.Box")
                    sx = geometry.size[0]
                    sy = geometry.size[1]
                    sz = geometry.size[2]

                    m = Marker(header=Header(frame_id=str(frame_id), stamp=rospy.Time.now()),
                               ns=str('c' + collection_key), id=count, frame_locked=True,
                               type=Marker.CUBE, action=Marker.ADD, lifetime=rospy.Duration(0),
                               pose=Pose(position=Point(x=x, y=y, z=z),
                                         orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)),
                               scale=Vector3(x=sx, y=sy, z=sz),
                               color=ColorRGBA(r=r, g=g, b=b, a=a))
                    markers.markers.append(m)
                    count += 1
                else:
                    print("visuals:\n " + str(visual))
                    raise ValueError('Unknown visual.geometry type' + str(type(visual.geometry)) + " for link " +
                                     link.name)

    graphics['ros']['markers_models'] = markers

    print('There are ' + str(len(markers.markers)) + ' markers')



    # # Create image publishers ----------------------------------------------------------
    # # We need to republish a new image at every visualization
    # for collection_key, collection in dataset_sensors['collections'].items():
    #     for sensor_key, _sensor in dataset_sensors['sensors'].items():
    #         if not collection['labels'][str(sensor_key)]['detected']:  # not detected by sensor in collection
    #             continue
    #
    #         if _sensor['msg_type'] == 'Image':
    #             msg_type = sensor_msgs.msg.Image
    #             topic_name = 'c' + str(collection_key) + '/' + str(sensor_key) + '/image_raw'
    #             dataset_graphics['collections'][collection_key][str(sensor_key)] = {'publisher': rospy.Publisher(
    #                 topic_name, msg_type, queue_size=0, latch=True)}
    #
    #             msg_type = sensor_msgs.msg.CameraInfo
    #             topic_name = 'c' + str(collection_key) + '/' + str(sensor_key) + '/camera_info'
    #             dataset_graphics['collections'][collection_key][str(sensor_key)]['publisher_camera_info'] = \
    #                 rospy.Publisher(topic_name, msg_type, queue_size=0, latch=True)
    #
    return graphics


def visualizationFunction(models):
    # Get the data from the model
    # parameters = data['parameters']
    # collections = models['dataset']['collections']
    # sensors =  models['dataset']['sensors']
    # pattern =  models['dataset']['pattern']
    # config =   models['dataset']['config']
    # graphics = models['dataset']['graphics']

    collections = models['dataset']['collections']
    sensors = models['dataset']['sensors']
    pattern = models['dataset']['pattern']
    config = models['dataset']['calibration_config']
    graphics = models['graphics']

    now = rospy.Time.now()
    # print("Calling visualization")

    for collection_key, collection in collections.items():

        parent = 'base_link'
        child = 'c' + collection_key + '_' + 'base_link'
        graphics['ros']['tf_broadcaster'].sendTransform((0, 0, 0), (0, 0, 0, 1), now, child, parent)

        for transform_key, transform in collection['transforms'].items():
            # print("Visiting collection " + collection_key + " transform " + transform_key)
            # TODO after https://github.com/lardemua/AtlasCarCalibration/issues/54 this will be unnecessary

            parent = 'c' + collection_key + '_' + transform_key.split('-')[0]
            child = 'c' + collection_key + '_' + transform_key.split('-')[1]
            graphics['ros']['tf_broadcaster'].sendTransform(transform['trans'], transform['quat'], now, child, parent)

    for marker in graphics['ros']['markers_models'].markers:
        marker.header.stamp = now

    graphics['ros']['publisher_models'].publish(graphics['ros']['markers_models'])
    #     rospy.sleep(1)

    # # Publish Annotated images
    # for collection_key, collection in dataset_sensors['collections'].items():
    #     for sensor_key, sensor in dataset_sensors['sensors'].items():
    #
    #         if not collection['labels'][sensor_key]['detected']:  # not detected by sensor in collection
    #             continue
    #
    #         if sensor['msg_type'] == 'Image':
    #             if args['show_images']:
    #                 image = copy.deepcopy(collection['data'][sensor_key]['data'])
    #                 width = collection['data'][sensor_key]['width']
    #                 height = collection['data'][sensor_key]['height']
    #                 diagonal = math.sqrt(width ** 2 + height ** 2)
    #                 cm = dataset_graphics['chessboard']['colormap']
    #
    #                 # Draw projected points (as dots)
    #                 for idx, point in enumerate(collection['labels'][sensor_key]['idxs_projected']):
    #                     x = int(round(point['x']))
    #                     y = int(round(point['y']))
    #                     color = (cm[idx, 2] * 255, cm[idx, 1] * 255, cm[idx, 0] * 255)
    #                     cv2.line(image, (x, y), (x, y), color, int(6E-3 * diagonal))
    #
    #                 # Draw ground truth points (as squares)
    #                 for idx, point in enumerate(collection['labels'][sensor_key]['idxs']):
    #                     x = int(round(point['x']))
    #                     y = int(round(point['y']))
    #                     color = (cm[idx, 2] * 255, cm[idx, 1] * 255, cm[idx, 0] * 255)
    #                     utilities.drawSquare2D(image, x, y, int(8E-3 * diagonal), color=color, thickness=2)
    #
    #                 # Draw initial projected points (as crosses)
    #                 for idx, point in enumerate(collection['labels'][sensor_key]['idxs_initial']):
    #                     x = int(round(point['x']))
    #                     y = int(round(point['y']))
    #                     color = (cm[idx, 2] * 255, cm[idx, 1] * 255, cm[idx, 0] * 255)
    #                     utilities.drawCross2D(image, x, y, int(8E-3 * diagonal), color=color, thickness=1)
    #
    #                 msg = CvBridge().cv2_to_imgmsg(image, "bgr8")
    #                 msg.header.frame_id = sensor_key + '_optical'  # TODO should be automated
    #                 dataset_graphics['collections'][collection_key][sensor_key]['publisher'].publish(msg)
    #
    #                 # Publish camera info message
    #                 camera_info_msg = CameraInfo()
    #                 camera_info_msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/CameraInfo',
    #                                                                                       sensor['camera_info'])
    #                 dataset_graphics['collections'][collection_key][sensor_key]['publisher_camera_info'].publish(
    #                     camera_info_msg)
    #
    #         elif sensor['msg_type'] == 'LaserScan':
    #             pass
    #
    #         else:
    #             raise ValueError("Unknown sensor msg_type")

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
