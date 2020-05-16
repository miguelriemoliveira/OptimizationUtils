# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import pprint
from copy import deepcopy
import math
import numpy as np
from statistics import mean

import rospy
from geometry_msgs.msg import Point
from image_geometry import PinholeCameraModel
from rospy_message_converter import message_converter
from scipy.spatial import distance
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
import ros_numpy  # Added by Andre Aguiar (it that ok?) - i think this on have to be added to the requirements.txt

import OptimizationUtils.utilities as utilities


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

def distance_two_3D_points(p0, p1):
    return math.sqrt(((p0[0] - p1[0]) ** 2) + ((p0[1] - p1[1]) ** 2) + ((p0[2] - p1[2]) ** 2))


# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)
    else:
        # The segment is parallel to plane.
        return None


# ----------------------
# generic math functions


def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
            (v0[0] * v1[0]) +
            (v0[1] * v1[1]) +
            (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )


def computeResidualsAverage(residuals):
    for _, r in residuals.items():
        if not r['count'] == 0:
            r['average'] = r['total'] / r['count']
        else:
            r['average'] = np.nan


def objectiveFunction(data):
    """
    Computes the vector of residuals. There should be an error for each stamp, sensor and chessboard tuple.
    The computation of the error varies according with the modality of the sensor:
        - Reprojection error for camera to chessboard
        - Point to plane distance for 2D laser scanners
        - (...)
        :return: a vector of residuals
    """
    # print('Calling objective function.')

    # Get the data from the model
    dataset_sensors = data['dataset_sensors']
    dataset_chessboards = data['dataset_sensors']['chessboards']
    dataset_chessboard_points = data['dataset_chessboard_points']  # TODO should be integrated into chessboards
    args = data['args']
    if args['view_optimization'] or args['ros_visualization']:
        dataset_graphics = data['graphics']

    r = {}  # Initialize residuals dictionary.
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][sensor_key]['detected']:  # chess not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'Image':
                # Compute chessboard points in local sensor reference frame
                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                root_to_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                pts_in_root = np.dot(root_to_chessboard, dataset_chessboard_points['points'])

                sensor_to_root = np.linalg.inv(utilities.getAggregateTransform(sensor['chain'],
                                                                               collection['transforms']))
                pts_sensor = np.dot(sensor_to_root, pts_in_root)

                # K = np.ndarray((3, 3), buffer=np.array(sensor['camera_info']['K']), dtype=np.float)
                P = np.ndarray((3, 4), buffer=np.array(sensor['camera_info']['P']), dtype=np.float)
                # P = P[0:3, 0:3]
                # print('P = \n' + str(P))
                # D = np.ndarray((5, 1), buffer=np.array(sensor['camera_info']['D']), dtype=np.float)
                width = collection['data'][sensor_key]['width']
                height = collection['data'][sensor_key]['height']

                # pixs, valid_pixs, dists = utilities.projectToCamera(K, D, width, height, pts_sensor[0:3, :])
                # pixs, valid_pixs, dists = utilities.projectToCamera(P, D, width, height, pts_sensor[0:3, :])
                # pixs, valid_pixs, dists = utilities.projectWithoutDistortion(K, width, height, pts_sensor[0:3, :])
                # See issue #106
                pixs, valid_pixs, dists = utilities.projectWithoutDistortion(P, width, height, pts_sensor[0:3, :])

                pixs_ground_truth = collection['labels'][sensor_key]['idxs']
                array_gt = np.zeros(pixs.shape, dtype=np.float)  # transform to np array
                for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                    array_gt[0][idx] = pix_ground_truth['x']
                    array_gt[1][idx] = pix_ground_truth['y']

                # Compute the error as the average of the Euclidean distances between detected and projected pixels
                # for idx in range(0, dataset_chessboards['number_corners']):
                #     e1 = math.sqrt(
                #         (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                #     raw_residuals.append(e1)
                #     error_sum += e1

                idx = 0
                rname = collection_key + '_' + sensor_key + '_0'
                r[rname] = math.sqrt((pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)

                idx = dataset_chessboards['chess_num_x'] - 1
                rname = collection_key + '_' + sensor_key + '_1'
                r[rname] = math.sqrt((pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)

                idx = dataset_chessboards['number_corners'] - dataset_chessboards['chess_num_x']
                rname = collection_key + '_' + sensor_key + '_2'
                r[rname] = math.sqrt((pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)

                idx = dataset_chessboards['number_corners'] - 1
                rname = collection_key + '_' + sensor_key + '_3'
                r[rname] = math.sqrt((pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)

                # Required by the visualization function to publish annotated images
                idxs_projected = []
                for idx in range(0, pixs.shape[1]):
                    idxs_projected.append({'x': pixs[0][idx], 'y': pixs[1][idx]})
                collection['labels'][sensor_key]['idxs_projected'] = idxs_projected  # store projections

                if 'idxs_initial' not in collection['labels'][sensor_key]:  # store the first projections
                    collection['labels'][sensor_key]['idxs_initial'] = deepcopy(idxs_projected)

            elif sensor['msg_type'] == 'LaserScan':
                # Get laser points that belong to the chessboard
                idxs = collection['labels'][sensor_key]['idxs']
                rhos = [collection['data'][sensor_key]['ranges'][idx] for idx in idxs]
                thetas = [collection['data'][sensor_key]['angle_min'] +
                          collection['data'][sensor_key]['angle_increment'] * idx for idx in idxs]

                # Convert from polar to cartesian coordinates and create np array with xyz coords
                # TODO could be done only once
                pts_in_laser = np.zeros((4, len(rhos)), np.float32)
                for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                    pts_in_laser[0, idx] = rho * math.cos(theta)
                    pts_in_laser[1, idx] = rho * math.sin(theta)
                    pts_in_laser[2, idx] = 0
                    pts_in_laser[3, idx] = 1

                # Compute the coordinate of the laser points in the chessboard reference frame
                root_to_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                pts_in_root = np.dot(root_to_sensor, pts_in_laser)

                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                chessboard_to_root = np.linalg.inv(utilities.translationQuaternionToTransform(trans, quat))
                pts_in_chessboard = np.dot(chessboard_to_root, pts_in_root)

                # --- Residuals: longitudinal error for extrema
                pts_canvas_in_chessboard = dataset_chessboards['limit_points'][0:2, :].transpose()

                # compute minimum distance to inner_pts for right most edge (first in pts_in_chessboard list)
                extrema_right = np.reshape(pts_in_chessboard[0:2, 0], (2, 1))  # longitudinal -> ignore z values
                rname = collection_key + '_' + sensor_key + '_eright'
                r[rname] = float(
                    np.amin(distance.cdist(extrema_right.transpose(), pts_canvas_in_chessboard, 'euclidean')))

                # compute minimum distance to inner_pts for left most edge (last in pts_in_chessboard list)
                extrema_left = np.reshape(pts_in_chessboard[0:2, -1], (2, 1))  # longitudinal -> ignore z values
                rname = collection_key + '_' + sensor_key + '_eleft'
                r[rname] = float(
                    np.amin(distance.cdist(extrema_left.transpose(), pts_canvas_in_chessboard, 'euclidean')))

                # --- Residuals: Longitudinal distance for inner points
                pts_inner_in_chessboard = dataset_chessboards['inner_points'][0:2, :].transpose()
                edges2d_in_chessboard = pts_in_chessboard[0:2, collection['labels'][sensor_key]['edge_idxs']]  # this
                # is a longitudinal residual, so ignore z values.

                for idx in range(edges2d_in_chessboard.shape[1]):  # compute minimum distance to inner_pts for each edge
                    xa = np.reshape(edges2d_in_chessboard[:, idx], (2, 1)).transpose()  # need the reshape because this
                    # becomes a shape (2,) which the function cdist does not support.

                    rname = collection_key + '_' + sensor_key + '_inner_' + str(idx)
                    r[rname] = float(np.amin(distance.cdist(xa, pts_inner_in_chessboard, 'euclidean')))

                # --- Residuals: Beam direction distance from point to chessboard plan
                # For computing the intersection we need:
                # p0, p1: Define the line.
                # p_co, p_no: define the plane:
                # p_co Is a point on the plane (plane coordinate).
                # p_no Is a normal vector defining the plane direction (does not need to be normalized).

                # Compute p0 and p1: p1 will be all the lidar data points, i.e., pts_in_laser, p0 will be the origin
                # of the laser sensor. Compute the p0_in_laser (p0)
                p0_in_laser = np.array([[0], [0], [0], [1]], np.float)

                # Compute p_co. It can be any point in the chessboard plane. Lets transform the origin of the
                # chessboard to the laser reference frame
                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                root_to_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                laser_to_chessboard = np.dot(np.linalg.inv(root_to_sensor), root_to_chessboard)

                p_co_in_chessboard = np.array([[0], [0], [0], [1]], np.float)
                p_co_in_laser = np.dot(laser_to_chessboard, p_co_in_chessboard)

                # Compute p_no. First compute an aux point (p_caux) and then use the vector from p_co to p_caux.
                p_caux_in_chessboard = np.array([[0], [0], [1], [1]], np.float)  # along the zz axis (plane normal)
                p_caux_in_laser = np.dot(laser_to_chessboard, p_caux_in_chessboard)

                p_no_in_laser = np.array([[p_caux_in_laser[0] - p_co_in_laser[0]],
                                          [p_caux_in_laser[1] - p_co_in_laser[1]],
                                          [p_caux_in_laser[2] - p_co_in_laser[2]],
                                          [1]], np.float)  # plane normal

                if args['view_optimization']:
                    marker = [x for x in dataset_graphics['ros']['MarkersLaserBeams'].markers if
                              x.ns == str(collection_key) + '-' + str(sensor_key)][0]
                    marker.points = []
                    rviz_p0_in_laser = Point(p0_in_laser[0], p0_in_laser[1], p0_in_laser[2])

                for idx in range(0, pts_in_laser.shape[1]):  # for all points
                    rho = rhos[idx]
                    p1_in_laser = pts_in_laser[:, idx]
                    pt_intersection = isect_line_plane_v3(p0_in_laser, p1_in_laser, p_co_in_laser, p_no_in_laser)

                    if pt_intersection is None:
                        raise ValueError('Error: chessboard is almost parallel to the laser beam! Please delete the '
                                         'collections in question.')

                    rname = collection_key + '_' + sensor_key + '_beam_' + str(idx)
                    r[rname] = abs(distance_two_3D_points(p0_in_laser, pt_intersection) - rho)

                    if args['view_optimization']:
                        marker.points.append(deepcopy(rviz_p0_in_laser))
                        marker.points.append(Point(pt_intersection[0], pt_intersection[1], pt_intersection[2]))

            elif sensor['msg_type'] == 'PointCloud2':
                from rospy_message_converter import message_converter

                # Convert 3D cloud data on .json dictionary to ROS message type
                cloud_msg = message_converter.convert_dictionary_to_ros_message("sensor_msgs/PointCloud2",
                                                                                collection['data'][sensor_key])

                # Get LiDAR points that belong to the chessboard on LiDAR's reference frame
                idxs = collection['labels'][sensor_key]['idxs']
                pc = ros_numpy.numpify(cloud_msg)[idxs]
                points = np.zeros((pc.shape[0], 4))
                points[:, 0] = pc['x']
                points[:, 1] = pc['y']
                points[:, 2] = pc['z']
                points[:, 3] = 1

                # --- Residuals: Distance from 3D range sensor point to chessboard plan
                # For computing the intersection we need:
                # p0, p1: Define the line.
                # p_co, p_no: define the plane:
                # p_co Is a point on the plane (plane coordinate).
                # p_no Is a normal vector defining the plane direction (does not need to be normalized).

                # Compute the homogeneous transformation from the root base_link to the sensor's reference frame
                root_to_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])

                # Compute p_co. It can be any point in the chessboard plane. Lets transform the origin of the
                # chessboard to the 3D cloud reference frame
                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                root_to_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                lidar_to_chessboard = np.dot(np.linalg.inv(root_to_sensor), root_to_chessboard)

                # Origin of the chessboard (0, 0, 0, 1) homogenized to the 3D range sensor reference frame
                p_co_in_chessboard = np.array([[0], [0], [0], [1]], np.float)
                p_co_in_lidar = np.dot(lidar_to_chessboard, p_co_in_chessboard)

                # Compute p_no. First compute an aux point (p_caux) and then use the normal vector from p_co to p_caux.
                p_caux_in_chessboard = np.array([[0], [0], [1], [1]], np.float)  # along the zz axis (plane normal)
                p_caux_in_lidar = np.dot(lidar_to_chessboard, p_caux_in_chessboard)

                p_no_in_lidar = np.array([[p_caux_in_lidar[0] - p_co_in_lidar[0]],
                                          [p_caux_in_lidar[1] - p_co_in_lidar[1]],
                                          [p_caux_in_lidar[2] - p_co_in_lidar[2]],
                                          [1]], np.float)  # plane normal

                if args['ros_visualization']:
                    marker = [x for x in dataset_graphics['ros']['MarkersLaserBeams'].markers if
                              x.ns == str(collection_key) + '-' + str(sensor_key)][0]
                    marker.points = []
                    rviz_p0_in_laser = Point(0,0,0)

                # Define p0 - the origin of 3D range sensor reference frame - to compute the line that intercepts the
                # chessboard plane in the loop
                p0_in_lidar = np.array([[0], [0], [0], [1], np.float])
                for idx in range(0, len(points)):
                    # Compute the interception between the chessboard plane into the 3D sensor reference frame, and the
                    # line that goes through the sensor origin to the measured point in the chessboard plane
                    p1_in_lidar = points[idx, :]
                    pt_intersection = isect_line_plane_v3(p0_in_lidar, p1_in_lidar, p_co_in_lidar, p_no_in_lidar)

                    if pt_intersection is None:
                        raise ValueError('Error: chessboard is almost parallel to the lidar beam! Please delete the '
                                         'collections in question.')

                    # Compute the residual: distance from sensor origin from the interception minus the actual range
                    # measure
                    rname = collection_key + '_' + sensor_key + '_oe_' + str(idx)
                    rho = np.sqrt(p1_in_lidar[0] ** 2 + p1_in_lidar[1] ** 2 + p1_in_lidar[2] ** 2)
                    r[rname] = abs(distance_two_3D_points(p0_in_lidar, pt_intersection) - rho)

                    if args['ros_visualization']:
                        marker.points.append(deepcopy(rviz_p0_in_laser))
                        marker.points.append(Point(pt_intersection[0], pt_intersection[1], pt_intersection[2]))
            else:
                raise ValueError("Unknown sensor msg_type")

    # --- Normalization of residuals.
    rn = deepcopy(r)  # make a copy of the non normalized dictionary.

    # Message type normalization. Pixels and meters should be weighted based on an adhoc defined meter_to_pixel factor.
    meter_to_pixel_factor = 200  # trial and error, the best technique around :-)
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][sensor_key]['detected']:  # chess not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'LaserScan':
                pair_keys = [k for k in rn.keys() if collection_key == k.split('_')[0] and sensor_key in k]  #
                # compute the residual keys that belong to this collection-sensor pair.
                rn.update({k: rn[k] * meter_to_pixel_factor for k in pair_keys})  # update the normalized dictionary.

    # Intra and Inter collection-sensor pair normalization.
    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            if not collection['labels'][sensor_key]['detected']:  # chess not detected by sensor in collection
                continue

            pair_keys = [k for k in rn.keys() if collection_key == k.split('_')[0] and sensor_key in k]

            if sensor['msg_type'] == 'Image':  # Intra normalization: for cameras there is nothing to do, since all
                # measurements have the same importance. Inter normalization, divide by the number of pixels considered.
                rn.update({k: rn[k] / len(pair_keys) for k in pair_keys})

            elif sensor['msg_type'] == 'LaserScan':  # Intra normalization: longitudinal measurements (extrema (.25)
                # and inner (.25)] and orthogonal measurements (beam (0.5)). Inter normalization, consider the number of
                # measurements.
                extrema_keys = [k for k in pair_keys if '_eleft' in k or '_eright' in k]
                rn.update({k: 0.25 / len(extrema_keys) * rn[k] for k in extrema_keys})

                inner_keys = [k for k in pair_keys if '_inner' in k]
                rn.update({k: 0.25 / len(inner_keys) * rn[k] for k in inner_keys})

                beam_keys = [k for k in pair_keys if '_beam' in k]
                rn.update({k: 0.5 / len(beam_keys) * rn[k] for k in beam_keys})

    # for collection_key, collection in dataset_sensors['collections'].items():
    #     for sensor_key, sensor in dataset_sensors['sensors'].items():
    #         pair_keys = [k for k in rn.keys() if collection_key == k.split('_')[0] and sensor_key in k]
    #         extrema_keys = [k for k in pair_keys if 'eleft' in k or 'eright' in k]
    #         inner_keys = [k for k in pair_keys if 'inner' in k]
    #         beam_keys = [k for k in pair_keys if 'beam' in k]
    #         print('Collection ' + collection_key + ' ' + sensor_key + ' has ' + str(len(pair_keys)) +
    #               ' residuals: ' + str(len(extrema_keys)) + ' extrema, ' + str(len(inner_keys)) + ' inner, ' +
    #               str(len(beam_keys)) + ' beam.')
    #
    # per_col_sensor = {str(c): {str(s): {'avg': mean([r[k] for k in r.keys() if c == k.split('_')[0] and s in k]),
    #                                     'navg': mean([rn[k] for k in rn.keys() if c == k.split('_')[0] and s in k])}
    #                            for s in dataset_sensors['sensors']} for c in dataset_sensors['collections']}
    #
    per_sensor = {str(sensor_key): {'avg': mean([r[k] for k in r.keys() if sensor_key in k]),
                                    'navg': mean([rn[k] for k in rn.keys() if sensor_key in k])}
                  for sensor_key in dataset_sensors['sensors']}

    per_msg_type = {'Image': {'avg': mean([r[k] for k in r.keys() if 'camera' in k]),
                              'navg': mean([rn[k] for k in rn.keys() if 'camera' in k])},
                    # 'LaserScan': {'avg': mean([r[k] for k in r.keys() if 'laser' in k]),
                    #               'navg': mean([rn[k] for k in rn.keys() if 'laser' in k])}
                    }
    # report = {'0-per_col_sensor': per_col_sensor, '1-per_sensor': per_sensor, '2-per_msg_type': per_msg_type}
    report = {'1-per_sensor': per_sensor, '2-per_msg_type': per_msg_type}

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(report)

    return rn  # Return the residuals
