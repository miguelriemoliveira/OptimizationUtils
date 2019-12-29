# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
from copy import deepcopy
import math
import numpy as np

import rospy
from geometry_msgs.msg import Point
from image_geometry import PinholeCameraModel
from rospy_message_converter import message_converter
from scipy.spatial import distance
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import MarkerArray, Marker

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
    if args['view_optimization']:
        dataset_graphics = data['dataset_graphics']

    # Initialize residuals. Also create dictionaries of partial residuals for better debugging.
    # , collection_key + '_count': 0
    # for collection_key in dataset_sensors['collections']
    residuals = []
    residuals_per_collection = {collection_key: {'total': 0.0, 'count': 0}
                                for collection_key in dataset_sensors['collections']}
    residuals_per_sensor = {sensor_key: {'total': 0.0, 'count': 0} for sensor_key in dataset_sensors['sensors']}
    residuals_per_msg_type = {'Image': {'total': 0.0, 'count': 0}, 'LaserScan': {'total': 0.0, 'count': 0}}

    def incrementResidualsCount(collection_key, sensor_key, msg_type):
        residuals_per_collection[collection_key]['count'] += 1
        residuals_per_sensor[sensor_key]['count'] += 1
        residuals_per_msg_type[msg_type]['count'] += 1

    for collection_key, collection in dataset_sensors['collections'].items():
        for sensor_key, sensor in dataset_sensors['sensors'].items():
            local_residuals = []  # reset local residuals

            if not collection['labels'][sensor_key]['detected']:  # chess not detected by sensor in collection
                continue

            # print("Computing residuals for collection " + collection_key + ", sensor " + sensor_key)

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
                P = P[0:3, 0:3]
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
                #     local_residuals.append(e1)
                #     error_sum += e1

                idx = 0
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                local_residuals.append(e1)
                incrementResidualsCount(collection_key, sensor_key, 'Image')

                idx = dataset_chessboards['chess_num_x'] - 1
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                local_residuals.append(e1)
                incrementResidualsCount(collection_key, sensor_key, 'Image')

                idx = dataset_chessboards['number_corners'] - dataset_chessboards['chess_num_x']
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                local_residuals.append(e1)
                incrementResidualsCount(collection_key, sensor_key, 'Image')

                idx = dataset_chessboards['number_corners'] - 1
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                local_residuals.append(e1)
                incrementResidualsCount(collection_key, sensor_key, 'Image')

                # Update residuals for images
                residuals.extend(local_residuals)  # extend list of residuals
                residuals_per_collection[collection_key]['total'] += sum([abs(x) for x in local_residuals])
                residuals_per_sensor[sensor_key]['total'] += sum([abs(x) for x in local_residuals])
                residuals_per_msg_type['Image']['total'] += sum([abs(x) for x in local_residuals])

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
                pts_in_laser = np.zeros((4, len(rhos)), np.float32)
                for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                    pts_in_laser[0, idx] = rho * math.cos(theta)
                    pts_in_laser[1, idx] = rho * math.sin(theta)
                    pts_in_laser[2, idx] = 0
                    pts_in_laser[3, idx] = 1

                # Get transforms
                root_to_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                pts_in_root = np.dot(root_to_sensor, pts_in_laser)

                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                chessboard_to_root = np.linalg.inv(utilities.translationQuaternionToTransform(trans, quat))

                pts_in_chessboard = np.dot(chessboard_to_root, pts_in_root)

                # --- Residuals: longitudinal error for extrema
                # TODO verify if the extrema points are not outliers ...

                # Miguel's approach to longitudinal error for extrema points
                pts_canvas_in_chessboard = dataset_chessboards['limit_points'][0:2, :].transpose()

                # compute minimum distance to inner_pts for right most edge (first in pts_in_chessboard list)
                extrema_right = np.reshape(pts_in_chessboard[0:2, 0], (2, 1))  # longitudinal -> ignore z values
                min_distance_right = np.amin(distance.cdist(extrema_right.transpose(),
                                                            pts_canvas_in_chessboard, 'euclidean'))
                local_residuals.append(min_distance_right)
                incrementResidualsCount(collection_key, sensor_key, 'LaserScan')

                # compute minimum distance to inner_pts for left most edge (last in pts_in_chessboard list)
                extrema_left = np.reshape(pts_in_chessboard[0:2, -1], (2, 1))  # longitudinal -> ignore z values
                min_distance_left = np.amin(distance.cdist(extrema_left.transpose(),
                                                           pts_canvas_in_chessboard, 'euclidean'))
                local_residuals.append(min_distance_left)
                incrementResidualsCount(collection_key, sensor_key, 'LaserScan')

                # Afonso's way
                # dists = np.zeros((1, 2), np.float)
                # idxs_min = np.zeros((1, 2), np.int)
                #
                # counter = 0
                # for idx in [0, -1]:
                #     pt_chessboard = pts_in_chessboard[:, idx]
                #     planar_pt_chessboard = pt_chessboard[0:2]
                #     pt = np.zeros((2, 1), dtype=np.float)
                #     pt[0, 0] = planar_pt_chessboard[0]
                #     pt[1, 0] = planar_pt_chessboard[1]
                #     planar_l_chess_pts = dataset_chessboards['limit_points'][0:2, :]
                #     vals = distance.cdist(pt.transpose(), planar_l_chess_pts.transpose(), 'euclidean')
                #     minimum = np.amin(vals)
                #     dists[0, counter] = minimum  # longitudinal distance to the chessboard limits
                #     for i in range(0, len(planar_l_chess_pts[0])):
                #         if vals[0, i] == minimum:
                #             idxs_min[0, counter] = i
                #
                #     counter += 1
                #
                # local_residuals.append(dists[0, 0])
                # incrementResidualsCount(collection_key, sensor_key, 'LaserScan')
                # local_residuals.append(dists[0, 1])
                # incrementResidualsCount(collection_key, sensor_key, 'LaserScan')

                # --- Residuals: Longitudinal distance for inner points

                # Miguel's approach to longitudinal distance inner points
                pts_inner_in_chessboard = dataset_chessboards['inner_points'][0:2, :].transpose()
                edges2d_in_chessboard = pts_in_chessboard[0:2, collection['labels'][sensor_key]['edge_idxs']]  # this
                # is a longitudinal residual, so ignore z values.

                for i in range(edges2d_in_chessboard.shape[1]):  # compute minimum distance to inner_pts for each edge
                    xa = np.reshape(edges2d_in_chessboard[:, i], (2, 1)).transpose()  # need the reshape because this
                    # becomes a shape (2,) which the function cdist does not support.

                    min_distance = np.amin(distance.cdist(xa, pts_inner_in_chessboard, 'euclidean'))
                    local_residuals.append(0.2 * min_distance)  # TODO check this ad hoc weighing of the residual
                    incrementResidualsCount(collection_key, sensor_key, 'LaserScan')

                # Afonso's way
                # edges = 0
                # for i in range(0, len(idxs) - 1):
                #     if (idxs[i + 1] - idxs[i]) != 1:
                #         edges += 1
                #
                #
                # # edges = len(collection['labels'][sensor_key]['edge_idxs']
                #
                # dists_inner_1 = np.zeros((1, edges), np.float)
                # dists_inner_2 = np.zeros((1, edges), np.float)
                # idxs_min_1 = np.zeros((1, edges), np.int)
                # idxs_min_2 = np.zeros((1, edges), np.int)
                # counter = 0
                #
                # for i in range(0, len(idxs) - 1):
                #     if (idxs[i + 1] - idxs[i]) != 1:
                #         # Compute longitudinal error for inner
                #         pt_chessboard_1 = pts_in_chessboard[:, i]
                #         pt_chessboard_2 = pts_in_chessboard[:, i + 1]
                #         planar_pt_chessboard_1 = pt_chessboard_1[0:2]
                #         planar_pt_chessboard_2 = pt_chessboard_2[0:2]
                #         pt_1 = np.zeros((2, 1), dtype=np.float)
                #         pt_1[0, 0] = planar_pt_chessboard_1[0]
                #         pt_1[1, 0] = planar_pt_chessboard_1[1]
                #         pt_2 = np.zeros((2, 1), dtype=np.float)
                #         pt_2[0, 0] = planar_pt_chessboard_2[0]
                #         pt_2[1, 0] = planar_pt_chessboard_2[1]
                #         planar_i_chess_pts = dataset_chessboards['inner_points'][0:2, :]
                #         vals_1 = distance.cdist(pt_1.transpose(), planar_i_chess_pts.transpose(), 'euclidean')
                #         vals_2 = distance.cdist(pt_2.transpose(), planar_i_chess_pts.transpose(), 'euclidean')
                #         minimum_1 = np.amin(vals_1)
                #         minimum_2 = np.amin(vals_2)
                #         dists_inner_1[0, counter] = minimum_1
                #         dists_inner_2[0, counter] = minimum_2
                #         for ii in range(0, len(planar_i_chess_pts[0])):
                #             if vals_1[0, ii] == minimum_1:
                #                 idxs_min_1[0, counter] = ii
                #             if vals_2[0, ii] == minimum_2:
                #                 idxs_min_2[0, counter] = ii
                #
                #         counter += 1
                # for c in range(0, counter):
                #     local_residuals.append(dists_inner_1[0, c])
                #     incrementResidualsCount(collection_key, sensor_key, 'LaserScan')
                #     local_residuals.append(dists_inner_2[0, c])
                #     incrementResidualsCount(collection_key, sensor_key, 'LaserScan')

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

                counter = 0
                for idx in range(0, pts_in_laser.shape[1]):  # for all points
                    rho = rhos[idx]
                    p1_in_laser = pts_in_laser[:, idx]
                    pt_intersection = isect_line_plane_v3(p0_in_laser, p1_in_laser, p_co_in_laser, p_no_in_laser)

                    if pt_intersection is None:
                        raise ValueError('Error: chessboard is almost parallel to the laser beam! Please delete the '
                                         'collections in question.')

                    computed_rho = distance_two_3D_points(p0_in_laser, pt_intersection)
                    local_residuals.append(abs(computed_rho - rho))  # abs is ok, check #109.
                    incrementResidualsCount(collection_key, sensor_key, 'LaserScan')

                    if args['view_optimization']:
                        marker.points.append(deepcopy(rviz_p0_in_laser))
                        marker.points.append(Point(pt_intersection[0], pt_intersection[1], pt_intersection[2]))

                    counter += 1

                # --------------------------------------------------------------------

                # Compute orthogonal error (for extremas only)
                # oe = np.zeros((1, 2), np.float)
                # counter = 0
                # for idx in [0, -1]: # for extremas only
                #     pt_chessboard = pts_in_chessboard[:, idx]
                #     oe[0, counter] = np.absolute( pt_chessboard[2])  # orthogonal distance to the chessboard limit points
                #     residuals.append(oe[0, counter])
                #     counter += 1

                # Compute orthogonal error (all the points)
                # oe = np.zeros((1, pts_in_chessboard.shape[1]), np.float)
                # counter = 0
                # for idx in range(0, pts_in_chessboard.shape[1]):  # for all points
                #     pt_chessboard = pts_in_chessboard[:, idx]
                #     oe[0, counter] = np.absolute(pt_chessboard[2])  # orthogonal distance to the chessboard limit points
                #     # residuals.append(oe[0, counter])
                #     local_residuals.append(oe[0, counter])
                #     incrementResidualsCount(collection_key, sensor_key, 'LaserScan')
                #     counter += 1

                # UPDATE GLOBAL RESIDUALS
                # local_residuals = [x * 10 for x in local_residuals]
                residuals.extend(local_residuals)  # extend list of residuals
                residuals_per_collection[collection_key]['total'] += sum([abs(x) for x in local_residuals])
                residuals_per_sensor[sensor_key]['total'] += sum([abs(x) for x in local_residuals])
                residuals_per_msg_type['LaserScan']['total'] += sum([abs(x) for x in local_residuals])

            else:
                raise ValueError("Unknown sensor msg_type")

    # Compute average of residuals
    # print('Residuals:\n ' + str(residuals))
    computeResidualsAverage(residuals_per_sensor)
    computeResidualsAverage(residuals_per_collection)
    computeResidualsAverage(residuals_per_msg_type)
    print('Avg residuals per collection:\n ' + str({key: residuals_per_collection[key]['average']
                                                    for key in residuals_per_collection}))
    print('Avg residuals per sensor:\n ' + str({key: residuals_per_sensor[key]['average']
                                                for key in residuals_per_sensor}))
    print('Avg residuals per msg_type:\n ' + str({key: residuals_per_msg_type[key]['average']
                                                  for key in residuals_per_msg_type}))
    print('Total error:\n ' + str(sum(residuals)))

    # createJSONFile('/tmp/data_collected_results.json', dataset_sensors)
    return residuals  # Return the residuals
