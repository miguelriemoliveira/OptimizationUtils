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

def objectiveFunction(data):
    """
    Computes the vector of errors. There should be an error for each stamp, sensor and chessboard tuple.
    The computation of the error varies according with the modality of the sensor:
        - Reprojection error for camera to chessboard
        - Point to plane distance for 2D laser scanners
        - (...)
        :return: a vector of residuals
    """
    print('Calling objective function.')

    # Get the data from the model
    dataset_sensors = data['dataset_sensors']
    dataset_chessboards = data['dataset_chessboards']
    dataset_chessboard_points = data['dataset_chessboard_points']
    errors = []

    for sensor_key, sensor in dataset_sensors['sensors'].items():
        sum_error = 0
        num_detections = 0
        for collection_key, collection in dataset_sensors['collections'].items():
            if not collection['labels'][sensor_key]['detected']:  # chess not detected by sensor in collection
                continue

            if sensor['msg_type'] == 'Image':

                # Compute chessboard points in local sensor reference frame
                trans = dataset_chessboards['collections'][collection_key]['trans']
                quat = dataset_chessboards['collections'][collection_key]['quat']
                root_T_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                pts_root = np.dot(root_T_chessboard, dataset_chessboard_points['points'])

                sensor_T_root = np.linalg.inv(
                    utilities.getAggregateTransform(sensor['chain'], collection['transforms']))
                pts_sensor = np.dot(sensor_T_root, pts_root)

                K = np.ndarray((3, 3), buffer=np.array(sensor['camera_info']['K']), dtype=np.float)
                D = np.ndarray((5, 1), buffer=np.array(sensor['camera_info']['D']), dtype=np.float)
                width = collection['data'][sensor_key]['width']
                height = collection['data'][sensor_key]['height']

                pixs, valid_pixs, dists = utilities.projectToCamera(K, D, width, height, pts_sensor[0:3, :])
                # print('pixs ' + str(pixs.shape) + ' =' + str(pixs))

                pixs_ground_truth = collection['labels'][sensor_key]['idxs']
                array_gt = np.zeros(pixs.shape, dtype=np.float)  # transform to np array
                for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                    array_gt[0][idx] = pix_ground_truth['x']
                    array_gt[1][idx] = pix_ground_truth['y']
                # print('pixs ground truth ' + str(array_gt.shape) + ' =' + str(array_gt))

                # Compute the error as the average of the Euclidean distances between detected and projected
                error_sum = 0
                error_vector = []
                # for idx in range(0, dataset_chessboards['number_corners']):
                #     # for idx in range(0, 1):
                #     e1 = math.sqrt(
                #         (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                #     error_vector.append(e1)
                #     error_sum += e1

                idx = 0
                e1 = math.sqrt(
                        (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                error_vector.append(e1)

                idx = dataset_chessboards['chess_num_x']-1
                e1 = math.sqrt(
                        (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                error_vector.append(e1)

                idx = dataset_chessboards['number_corners'] - dataset_chessboards['chess_num_x']
                e1 = math.sqrt(
                        (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                error_vector.append(e1)

                idx = dataset_chessboards['chess_num_x']-1
                e1 = math.sqrt(
                        (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                error_vector.append(e1)

                # for idx in range(dataset_chessboards['chess_num_x']-1, dataset_chessboards['chess_num_x']):
                #     e2 = math.sqrt(
                #         (pixs[0, idx] - array_gt[0, idx]) ** 2 + (pixs[1, idx] - array_gt[1, idx]) ** 2)
                    # error_sum += e2


                # error = error_sum / (args['chess_num_x'] * args['chess_num_y'])
                # error = error_sum
                # # error = max(e1, e2)
                # print('e1 = ' + str(e1))
                # print('e2 = ' + str(e2))
                # print('error_sum = ' + str(error_sum))
                # print('error = ' + str(error))
                # errors.append(error)
                errors.extend(error_vector)

                # For collecting individual distances and to store projected pixels into dataset_sensors dict for
                # drawing in visualization function

                # error_x = []
                # error_y = []
                # for idx in range(0, args['chess_num_x'] * args['chess_num_y']):
                #     error_x.append(pixs[0, idx] - array_gt[0, idx])
                #     error_y.append(pixs[1, idx] - array_gt[1, idx])
                #
                # sum_error += error
                # num_detections += 1
                # collection['labels'][sensor_key]['errors'] = {'x': error_x, 'y': error_y}

                idxs_projected = []
                for idx, pix_ground_truth in enumerate(pixs_ground_truth):
                    idxs_projected.append({'x': pixs[0][idx], 'y': pixs[1][idx]})

                collection['labels'][sensor_key]['idxs_projected'] = idxs_projected

                if not 'idxs_initial' in collection['labels'][sensor_key]:  # store the first projections
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
                # print('pts_chessboard =\n' + str(pts_chessboard))

                # Compute longitudinal error

                # dists = np.zeros((1, pts_chessboard.shape[1]), np.float)
                dists = np.zeros((1, 2), np.float)
                idxs_min = np.zeros((1, 2), np.int)
                # for idx in range(pts_chessboard.shape[1]):
                counter = 0

                # print('chessboard_evaluation_points = \n' + str(chessboard_evaluation_points))

                for idx in [0, -1]:
                    # for idx in [0]:
                    pt_chessboard = pts_chessboard[0:2, idx]
                    # if idx == 0:
                    #     print('extrema_right = ' + str(pt_chessboard))
                    # else:
                    #     print('extrema_left = ' + str(pt_chessboard))

                    pt = np.zeros((2, 1), dtype=np.float)
                    pt[0, 0] = pt_chessboard[0]
                    pt[1, 0] = pt_chessboard[1]
                    # pt[2, 0] = pt_chessboard[2]
                    # # dists[0, idx] = directed_hausdorff(pt_chessboard, chessboard_evaluation_points[0:3, :])[0]
                    # # dists[0, idx] = directed_hausdorff(pt, chessboard_evaluation_points[0:3, :])[0]
                    # dists[0, idx] = directed_hausdorff(chessboard_evaluation_points[0:3, :], pt)[0]
                    # print('using hausdorf got ' + str(dists[0,idx]))

                    vals = distance.cdist(pt.transpose(), dataset_chessboards['evaluation_points'][0:2, :].transpose())
                    # print(vals)
                    # print(idxs1)
                    # print(idxs2)

                    # dists[0,counter] = np.power(vals.min(axis=1), 7)/10
                    # dists[0,counter] = np.exp(vals.min(axis=1))/100
                    dists[0, counter] = vals.min(axis=1)
                    idxs_min[0, counter] = vals.argmin(axis=1)

                    # dists[0, counter] = np.average(vals)
                    # print(dists[0,counter])

                    # print('using cdist got ' + str(dists[0,idx]))
                    # distances = np.linalg.norm(pt_chessboard - chessboard_evaluation_points[0:3, idx_eval])

                    # dists[0, idx] = sys.maxsize
                    # for idx_eval in range(chessboard_evaluation_points.shape[1]):
                    #     dist = np.linalg.norm(pt_chessboard - chessboard_evaluation_points[0:3, idx_eval])
                    #     if dist < dists[0, idx]:
                    #         dists[0, idx] = dist
                    #
                    #
                    # print('using for got ' + str(dists[0,idx]))
                    counter += 1

                dists = dists[0]
                # print('dists = ' + str(dists))
                # print('idxs_min = ' + str(idxs_min))
                # error_longitudinal = np.average(dists)
                # error_longitudinal = np.average(dists) * 100
                # error_longitudinal = np.sum(dists) * 100
                error_longitudinal = np.max(dists)

                # Compute the longitudinal radius error

                # dists = np.zeros((1, 2), np.float)
                distances = []

                # pts_extrema = pts_chessboard[0:2, [0,-1]]
                # print(pts_extrema)

                # Compute orthogonal error

                collection['labels'][sensor_key]['errors'] = pts_chessboard[2, :].tolist()

                # error_orthogonal = np.average(np.absolute(pts_chessboard[2, :]))
                error_orthogonal = np.absolute(pts_chessboard[2, :])

                # sum_error += error_orthogonal
                num_detections += 1

                # TODO error in meters? Seems small when compared with pixels ...
                # error = error_longitudinal + error_orthogonal
                # error = error_longitudinal
                error = error_orthogonal
                # errors.append(error)

                errors.append(error_orthogonal[0])
                errors.append(error_orthogonal[-1])

                # Store for visualization
                collection['labels'][sensor_key]['pts_root'] = pts_root

                # root_T_chessboard = utilities.translationQuaternionToTransform(trans, quat)
                # minimum_associations = chessboard_evaluation_points[:,idxs_min[0]]
                # # print('chessboard_evaluation_points_in_root = ' + str(minimum_associations))
                # minimum_associations_in_root = np.dot(root_T_chessboard, minimum_associations)
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

            # print('error for sensor ' + sensor_key + ' in collection ' + collection_key + ' is ' + str(error))
        if num_detections == 0:
            continue
        else:
            print('avg error for sensor ' + sensor_key + ' is ' + str(sum_error / num_detections))

    # Return the errors
    # createJSONFile('/tmp/data_collected_results.json', dataset_sensors)
    return errors
