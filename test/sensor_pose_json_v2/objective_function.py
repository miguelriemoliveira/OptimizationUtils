# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
from copy import deepcopy
import math
import numpy as np
from scipy.spatial import distance

import OptimizationUtils.utilities as utilities


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

def distance_two_3D_points(p0, p1):
    distance = math.sqrt(((p0[0] - p1[0]) ** 2) + ((p0[1] - p1[1]) ** 2) + ((p0[2] - p1[2]) ** 2))
    return distance


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
    dataset_chessboard_points = data['dataset_chessboard_points']
    residuals = []  # a list of errors returned
    residuals_per_collections = {}
    residuals_per_sensor = {}

    for collection_key, collection in dataset_sensors['collections'].items():
        residuals_per_collections[collection_key] = 0.0  # initialize error per collection

        for sensor_key, sensor in dataset_sensors['sensors'].items():
            residuals_per_sensor[sensor_key] = 0.0  # initialize error per sensor
            sensor_residuals = []  # reset sensor residuals
            sum_error = 0
            num_detections = 0

            if not collection['labels'][sensor_key]['detected']:  # chess not detected by sensor in collection
                continue

            # print("Computing residuals for collection " + collection_key + ", sensor " + sensor_key)

            if sensor['msg_type'] == 'Image':

                # Compute chessboard points in local sensor reference frame
                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                root_to_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                pts_root = np.dot(root_to_chessboard, dataset_chessboard_points['points'])

                sensor_to_root = np.linalg.inv(utilities.getAggregateTransform(sensor['chain'],
                                                                               collection['transforms']))
                pts_sensor = np.dot(sensor_to_root, pts_root)

                K = np.ndarray((3, 3), buffer=np.array(sensor['camera_info']['K']), dtype=np.float)
                D = np.ndarray((5, 1), buffer=np.array(sensor['camera_info']['D']), dtype=np.float)
                width = collection['data'][sensor_key]['width']
                height = collection['data'][sensor_key]['height']

                pixs, valid_pixs, dists = utilities.projectToCamera(K, D, width, height, pts_sensor[0:3, :])
                # pixs, valid_pixs, dists = utilities.projectWithoutDistortion(K, width, height, pts_sensor[0:3, :])

                pixs_ground_truth = collection['labels'][sensor_key]['idxs']
                array_gt = np.zeros(pixs.shape, dtype=np.float)  # transform to np array
                for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                    array_gt[0][idx] = pix_ground_truth['x']
                    array_gt[1][idx] = pix_ground_truth['y']

                # Compute the error as the average of the Euclidean distances between detected and projected pixels
                # for idx in range(0, dataset_chessboards['number_corners']):
                #     e1 = math.sqrt(
                #         (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                #     sensor_residuals.append(e1)
                #     error_sum += e1

                idx = 0
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                sensor_residuals.append(e1)

                idx = dataset_chessboards['chess_num_x'] - 1
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                sensor_residuals.append(e1)

                idx = dataset_chessboards['number_corners'] - dataset_chessboards['chess_num_x']
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                sensor_residuals.append(e1)

                idx = dataset_chessboards['number_corners'] - 1
                e1 = math.sqrt(
                    (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                # e1 = e1 / 100
                sensor_residuals.append(e1)

                # UPDATE GLOBAL RESIDUALS
                residuals.extend(sensor_residuals)  # extend list of residuals
                residuals_per_collections[collection_key] += sum(sensor_residuals)
                residuals_per_sensor[sensor_key] += sum(sensor_residuals)

                # Required by the visualization function to publish annotated images
                idxs_projected = []
                for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                    idxs_projected.append({'x': pixs[0][idx], 'y': pixs[1][idx]})

                collection['labels'][sensor_key]['idxs_projected'] = idxs_projected

                if 'idxs_initial' not in collection['labels'][sensor_key]:  # store the first projections
                    collection['labels'][sensor_key]['idxs_initial'] = deepcopy(idxs_projected)

            elif sensor['msg_type'] == 'LaserScan':

                # Get laser points that belong to the chessboard
                idxs = collection['labels'][sensor_key]['idxs']
                rhos = [collection['data'][sensor_key]['ranges'][idx] for idx in idxs]
                thetas = [collection['data'][sensor_key]['angle_min'] +
                          collection['data'][sensor_key]['angle_increment'] * idx for idx in idxs]

                # Convert from polar to cartesian coordinates and create np array with xyz coords
                pts_laser = np.zeros((3, len(rhos)), np.float32)
                for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                    pts_laser[0, idx] = rho * math.cos(theta)
                    pts_laser[1, idx] = rho * math.sin(theta)

                # homogenize
                pts_laser = np.vstack((pts_laser, np.ones((1, pts_laser.shape[1]), dtype=np.float)))

                # Get transforms
                root_T_sensor = utilities.getAggregateTransform(sensor['chain'], collection['transforms'])
                pts_root = np.dot(root_T_sensor, pts_laser)

                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                chessboard_T_root = np.linalg.inv(utilities.translationQuaternionToTransform(trans, quat))

                pts_chessboard = np.dot(chessboard_T_root, pts_root)

                # Compute longitudinal and orthogonal error

                dists = np.zeros((1, 2), np.float)
                idxs_min = np.zeros((1, 2), np.int)

                # TODO verify if the extrema points are not outliers ...

                # Compute longitudinal error for extremas
                counter = 0
                for idx in [0, -1]:
                    pt_chessboard = pts_chessboard[:, idx]
                    planar_pt_chessboard = pt_chessboard[0:2]
                    pt = np.zeros((2, 1), dtype=np.float)
                    pt[0, 0] = planar_pt_chessboard[0]
                    pt[1, 0] = planar_pt_chessboard[1]
                    planar_l_chess_pts = dataset_chessboards['limit_points'][0:2, :]
                    vals = distance.cdist(pt.transpose(), planar_l_chess_pts.transpose(), 'euclidean')
                    minimum = np.amin(vals)
                    dists[0, counter] = minimum  # longitudinal distance to the chessboard limits
                    for i in range(0, len(planar_l_chess_pts[0])):
                        if vals[0, i] == minimum:
                            idxs_min[0, counter] = i

                    counter += 1

                sensor_residuals.append(dists[0, 0])
                sensor_residuals.append(dists[0, 1])

                # ---------------------------------LONGITUDINAL DISTANCE FOR INNER POINTS -------------------------
                edges = 0
                for i in range(0, len(idxs) - 1):
                    if (idxs[i + 1] - idxs[i]) != 1:
                        edges += 1

                dists_inner_1 = np.zeros((1, edges), np.float)
                dists_inner_2 = np.zeros((1, edges), np.float)
                idxs_min_1 = np.zeros((1, edges), np.int)
                idxs_min_2 = np.zeros((1, edges), np.int)
                counter = 0

                for i in range(0, len(idxs) - 1):
                    if (idxs[i + 1] - idxs[i]) != 1:
                        # Compute longitudinal error for inner
                        pt_chessboard_1 = pts_chessboard[:, i]
                        pt_chessboard_2 = pts_chessboard[:, i + 1]
                        planar_pt_chessboard_1 = pt_chessboard_1[0:2]
                        planar_pt_chessboard_2 = pt_chessboard_2[0:2]
                        pt_1 = np.zeros((2, 1), dtype=np.float)
                        pt_1[0, 0] = planar_pt_chessboard_1[0]
                        pt_1[1, 0] = planar_pt_chessboard_1[1]
                        pt_2 = np.zeros((2, 1), dtype=np.float)
                        pt_2[0, 0] = planar_pt_chessboard_2[0]
                        pt_2[1, 0] = planar_pt_chessboard_2[1]
                        planar_i_chess_pts = dataset_chessboards['inner_points'][0:2, :]
                        vals_1 = distance.cdist(pt_1.transpose(), planar_i_chess_pts.transpose(), 'euclidean')
                        vals_2 = distance.cdist(pt_2.transpose(), planar_i_chess_pts.transpose(), 'euclidean')
                        minimum_1 = np.amin(vals_1)
                        minimum_2 = np.amin(vals_2)
                        dists_inner_1[0, counter] = minimum_1
                        dists_inner_2[0, counter] = minimum_2
                        for ii in range(0, len(planar_i_chess_pts[0])):
                            if vals_1[0, ii] == minimum_1:
                                idxs_min_1[0, counter] = ii
                            if vals_2[0, ii] == minimum_2:
                                idxs_min_2[0, counter] = ii

                        counter += 1
                for c in range(0, counter):
                    sensor_residuals.append(dists_inner_1[0, c])
                    sensor_residuals.append(dists_inner_2[0, c])

                # --------------------------------------------------------------------

                # --------------------------------- BEAM DIRECTION DISTANCE FROM POINT TO CHESSBOARD PLAN  ------------
                # Compute chessboard points in local sensor reference frame
                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                root_to_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                origin = np.zeros((3, 1), np.int)
                origin = np.vstack((origin, np.ones((1, origin.shape[1]), dtype=np.float)))
                laser_origin = np.dot(root_T_sensor, origin)
                beam_distance_error = np.zeros((1, pts_chessboard.shape[1]), np.float)
                num_x = dataset_chessboards['chess_num_x']
                num_y = dataset_chessboards['chess_num_y']
                point_chessboard = np.zeros((3, 1), np.float32)
                point_chessboard[0] = dataset_chessboard_points['points'][0][num_x + 3]
                point_chessboard[1] = dataset_chessboard_points['points'][1][num_x + 3]
                point_chessboard = np.vstack(
                    (point_chessboard, np.ones((1, point_chessboard.shape[1]), dtype=np.float)))
                point_chessboard_root = np.dot(root_T_sensor, point_chessboard)
                normal_vector = np.zeros((3, 1), np.float32)
                normal_vector[0] = root_to_chessboard[0][2]
                normal_vector[1] = root_to_chessboard[1][2]
                normal_vector[2] = root_to_chessboard[2][2]
                counter = 0
                for idx in range(0, pts_chessboard.shape[1]):  # for all points
                    pt_root = pts_root[:, idx]
                    interseption_point = isect_line_plane_v3(laser_origin, pt_root, point_chessboard_root,
                                                             normal_vector)
                    if interseption_point == None:
                        print("Error: chessboard is almost parallel to the laser beam! Please delete the collections"
                              " in question.")
                        exit(0)
                    else:
                        beam_distance_error[0, counter] = distance_two_3D_points(pt_root, interseption_point)
                        sensor_residuals.append(beam_distance_error[0, counter])
                    counter += 1
                # print("beam distance error vector:\n")
                # print(beam_distance_error)
                # --------------------------------------------------------------------

                # Compute orthogonal error (for extremas only)
                # oe = np.zeros((1, 2), np.float)
                # counter = 0
                # for idx in [0, -1]: # for extremas only
                #     pt_chessboard = pts_chessboard[:, idx]
                #     oe[0, counter] = np.absolute( pt_chessboard[2])  # orthogonal distance to the chessboard limit points
                #     residuals.append(oe[0, counter])
                #     counter += 1

                # Compute orthogonal error (all the points)
                # oe = np.zeros((1, pts_chessboard.shape[1]), np.float)
                # counter = 0
                # for idx in range(0, pts_chessboard.shape[1]):  # for all points
                #     pt_chessboard = pts_chessboard[:, idx]
                #     oe[0, counter] = np.absolute(pt_chessboard[2])  # orthogonal distance to the chessboard limit points
                #     residuals.append(oe[0, counter])
                #     counter += 1

                # UPDATE GLOBAL RESIDUALS
                residuals.extend(sensor_residuals)  # extend list of residuals
                residuals_per_collections[collection_key] += sum(sensor_residuals)
                residuals_per_sensor[sensor_key] += sum(sensor_residuals)

                # Compute average for printing
                num_detections += 1
                # c_error = (dists[0, 0] + dists[0, 1] + oe[0, 0] + oe[0, 1]) / 4
                c_error = (dists[0, 0] + dists[0, 1]) / 2
                sum_error += c_error

                # Store for visualization
                collection['labels'][sensor_key]['pts_root'] = pts_root

                # root_to_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                # minimum_associations = chessboard_evaluation_points[:,idxs_min[0]]
                # # print('chessboard_evaluation_points_in_root = ' + str(minimum_associations))
                # minimum_associations_in_root = np.dot(root_to_chessboard, minimum_associations)
                #
                # if 'minimum_associations' in collection['labels'][sensor_key]:
                #     utilities.drawPoints3D(ax, None, minimum_associations_in_root, color=[.8, .7, 0],
                #                        marker_size=3.5, line_width=2.2,
                #                        marker='^',
                #                        mfc=[.8,.7,0],
                #                        text=None,
                #                        text_color=sensor['color'],
                #                        sensor_color=sensor['color'],
                #                        handles=collection['labels'][sensor_key]['minimum_associations'])
                # else:
                #     collection['labels'][sensor_key]['minimum_associations'] = utilities.drawPoints3D(ax, None, minimum_associations_in_root, color=[.8, .7, 0],
                #                        marker_size=3.5, line_width=2.2,
                #                        marker='*',
                #                        mfc=[.8,.7,0],
                #                        text=None,
                #                        text_color=sensor['color'],
                #                        sensor_color=sensor['color'],
                #                        handles=None)

                # if sensor_key == 'left_laser':
                #     print('Collection ' + collection_key + ', sensor ' + sensor_key + ' tranforms=' + str(
                #         collection['transforms']['car_center-left_laser']))

                # if sensor_key == 'left_laser':
                #     print('Collection ' + collection_key + ', sensor ' + sensor_key + ' pts_root=\n' + str(
                #         collection['labels'][sensor_key]['pts_root'][:,0::15]))

                if not 'pts_root_initial' in collection['labels'][sensor_key]:  # store the first projections
                    collection['labels'][sensor_key]['pts_root_initial'] = deepcopy(pts_root)

            else:
                raise ValueError("Unknown sensor msg_type")
            #
            # print('\n\nerror for sensor ' + sensor_key + ' in collection ' + collection_key + ' is ' + str(residuals))

        # if num_detections == 0:
        #     continue
        # else:
        #     print('avg error for sensor ' + sensor_key + ' is ' + str(sum_error / num_detections))

    # Return the residuals
    # createJSONFile('/tmp/data_collected_results.json', dataset_sensors)
    return residuals
