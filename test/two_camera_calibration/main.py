#!/usr/bin/env python2
"""
This code shows how the optimizer class may be used for changing the colors of a set of images so that the average
color in all images is very similar. This is often called color correction.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import argparse  # to read command line arguments
import math

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import cv2
import KeyPressManager.KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities


class Calibration:

    def __init__(self):
        self.right_cam_rotation_vector = [0.0, 0.0, 0.0]
        # np.zeros(3, np.float32)
        self.right_cam_translation = [0.0, 0.0, 0.0]
        self.right_cam_image_points = []


def generate_chessboard(size, dimensions=(9, 6)):
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp = objp * size
    return objp


def find_cam_chess_realpoints(fname, left_or_right):
    objpoints_left = []
    imgpoints_left = []
    objpoints_right = []
    imgpoints_right = []
    print(fname)
    img = cv2.imread(fname)
    if img is None:
        raise ValueError('Could not read image from ' + str(fname))
    # print(img.shape)
    # cv2.imshow('gui', img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        if left_or_right == 0:
            objpoints_left.append(pts_chessboard)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2)
            retval, rvec, tvec = cv2.solvePnP(objpoints_left[0], imgpoints_left[0], k_left,
                                              dist_left)  # calculating the rotation and translation vectors from left camera to chess board
        elif left_or_right == 1:
            objpoints_right.append(pts_chessboard)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(corners2)
            retval, rvec, tvec = cv2.solvePnP(objpoints_right[0], imgpoints_right[0], k_right,
                                              dist_right)  # calculating the rotation and translation vectors from left camera to chess board

    return rvec, tvec


if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-d", "--dataset_path", help="Path to the dataset", type=str, required=True)
    args = vars(ap.parse_args())

    print(args)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    calibration = Calibration()

    # Chessboard dimensions
    dimensions = (6, 9)
    size_board = 100

    # K matrix and distortion coefficients from cameras
    k_left = np.array([[1149.369, 0.0, 471.693], [0.0, 1153.728, 396.955], [0.0, 0.0, 1.0]])
    dist_left = np.array([-1.65512977e-01, -2.08184195e-01, -2.17490237e-03, -5.04628479e-04, 1.18772434e+00])

    k_right = np.array([[1135.560, 0.0, 490.807], [0.0, 1136.240, 412.468], [0.0, 0.0, 1.0]])
    dist_right = np.array([-2.06069540e-01, -1.27768958e-01, 2.22591520e-03, 1.60327811e-03, 2.08236968e+00])

    # Image used
    if not args['dataset_path'][-1] == '/':  # make sure the path is correct
        args['dataset_path'] += '/'
    name_image_left = args['dataset_path'] + 'top_left_camera_10.jpg'
    name_image_right = args['dataset_path'] + 'top_right_camera_10.jpg'

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating chessboard points
    pts_chessboard = generate_chessboard(0.101, (9, 6))

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('calibration', calibration)

    print("calibration model\n")
    print(calibration.right_cam_rotation_vector[0])
    print(opt.data_models['calibration'].right_cam_rotation_vector[0])

    # Create specialized getter and setter functions
    def setter(calibration, value, idx):
        if idx == 0:
            print("value\n")
            print(value)
            calibration.right_cam_rotation_vector[0] = value
        elif idx == 1:
            calibration.right_cam_rotation_vector[1] = value
        elif idx == 2:
            calibration.right_cam_rotation_vector[2] = value
        elif idx == 3:
            calibration.right_cam_translation[0] = value
        elif idx == 4:
            calibration.right_cam_translation[1] = value
        elif idx == 5:
            calibration.right_cam_translation[2] = value
        else:
            raise ValueError('Unknown i value: ' + str(idx))


    def getter(calibration, idx):
        if idx == 0:
            print("getter\n")
            print(calibration.right_cam_rotation_vector)
            return [calibration.right_cam_rotation_vector[0]]
        elif idx == 1:
            return [calibration.right_cam_rotation_vector[1]]
        elif idx == 2:
            return [calibration.right_cam_rotation_vector[2]]
        elif idx == 3:
            return [calibration.right_cam_translation[0]]
        elif idx == 4:
            return [calibration.right_cam_translation[1]]
        elif idx == 5:
            return [calibration.right_cam_translation[2]]
        else:
            raise ValueError('Unknown i value: ' + str(idx))

    parameter_names = ['r1', 'r2', 'r3', 'tx', 'ty', 'tz']
    for idx in range(0, 6):
        opt.pushParamScalar(group_name=parameter_names[idx], data_key='calibration', getter=partial(getter, idx=idx),
                            setter=partial(setter, idx=idx))

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        calibration = model['calibration']
        print("calibrations")
        print(calibration.right_cam_translation)

        right_cam_rotation_vector = np.zeros((1, 3), np.float32)
        right_cam_translation = np.zeros((3), np.float32)
        for i in range(0, 3):
            right_cam_rotation_vector[0, i] = calibration.right_cam_rotation_vector[i][0]
            print(calibration.right_cam_translation[i][0])
            right_cam_translation[i] = calibration.right_cam_translation[i][0]
        # Get T from left camera to chess
        rvec_cam_left, tvec_cam_left = find_cam_chess_realpoints(name_image_left, 0)
        left_cam_T_chess = np.zeros((4, 4), np.float32)
        left_cam_T_chess[0:3, 0:3], _ = cv2.Rodrigues(rvec_cam_left)
        left_cam_T_chess[0:3, 3] = tvec_cam_left.T
        left_cam_T_chess[3, :] = [0, 0, 0, 1]  # homogenize

        # Get T from left cam to right cam (based on the params being oiptimized)
        right_cam_T_left_cam = np.zeros((4, 4), np.float32)

        right_cam_T_left_cam[0:3, 0:3], _ = cv2.Rodrigues(right_cam_rotation_vector)
        right_cam_T_left_cam[0:3, 3] = right_cam_translation.T
        right_cam_T_left_cam[3, :] = [0, 0, 0, 1]  # homogenize

        # Get aggregate T from cam_right to chess (optimized)
        right_cam_T_chess_opt = np.matmul(right_cam_T_left_cam, left_cam_T_chess)

        # Get T from right camera to chess (ground truth)
        rvec_cam_right, tvec_cam_right = find_cam_chess_realpoints(name_image_right, 1)
        right_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
        right_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_right)[0]
        right_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_right.T

        # Draw projection of (optimized) 3D points
        r_cam2tochess_vector, _ = cv2.Rodrigues(right_cam_T_chess_opt[0:3, 0:3])
        # t_cam2tochess = np.zeros((3, 1))
        t_cam2tochess = right_cam_T_chess_opt[0:3, 3]
        imgpoints_right_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess, k_right,
                                                     dist_right)

        calibration.right_cam_image_points = imgpoints_right_optimize

        imgpoints_right_real = cv2.projectPoints(pts_chessboard, rvec_cam_right, tvec_cam_right, k_right, dist_right)

        error = []
        for a in range(dimensions[0] * dimensions[1]):
            error.append(math.sqrt((imgpoints_right_optimize[0][a][0][0] - imgpoints_right_real[0][a][0][0]) ** 2 + (
                    imgpoints_right_optimize[0][a][0][1] - imgpoints_right_real[0][a][0][1]) ** 2))


        print("avg error: " + str(np.mean(np.array(error))))
        return error


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for a in range(0, dimensions[0] * dimensions[1]):
        opt.pushResidual(name='r' + str(a), params=parameter_names)

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()
    opt.printSparseMatrix()

    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------

    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # fig = plt.figure()
    # ax = fig.gca()
    #
    # ax.set_xlabel('X'), ax.set_ylabel('Y'),
    # ax.set_xticklabels([]), ax.set_yticklabels([])
    # ax.set_xlim(-math.pi/2, math.pi/2), ax.set_ylim(-5, 5)
    #
    # # Draw cosine fucntion
    # f = np.cos(x)
    # ax.plot(x, f, label="cosine")
    # legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    #
    # y = 0 + \
    #     np.multiply(0, np.power(x, 1)) + \
    #     np.multiply(0, np.power(x, 2)) + \
    #     np.multiply(0, np.power(x, 3)) + \
    #     np.multiply(0, np.power(x, 4))
    #
    # handle_plot = ax.plot(x, y, label="calibration")
    # print(type(handle_plot))
    # print((handle_plot))
    #
    # wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    # if wm.waitForKey(0., verbose=False):
    #     exit(0)

    # handles_out = {}
    # handles_out['point'] = ax.plot([pt_origin[0, 0], pt_origin[0, 0]], [pt_origin[1, 0], pt_origin[1, 0]],
    #                                [pt_origin[2, 0], pt_origin[2, 0]], 'k.')[0]
    # handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black',
    #                               fontsize=fontsize)
    # else:
    #     handles['point'].set_xdata([pt_origin[0, 0], pt_origin[0, 0]])
    #     handles['point'].set_ydata([pt_origin[1, 0], pt_origin[1, 0]])
    #     handles['point'].set_3d_properties(zs=[pt_origin[2, 0], pt_origin[2, 0]])
    #
    #     handles['text'].set_position((pt_origin[0, 0], pt_origin[1, 0]))
    #     handles['text'].set_3d_properties(z=pt_origin[2, 0], zdir='x')

    def visualizationFunction(model):
        pass
    #
    #     calibration = model['calibration']
    #
    #     img1 = cv2.imread(name_image_right)
    #     img = cv2.drawChessboardCorners(img1, (9, 6), calibration.right_cam_image_points[0], True)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(20)
    #
    #     # y = calibration.param0[0] + \
    #     #     np.multiply(calibration.param1[0], np.power(x, 1)) + \
    #     #     np.multiply(calibration.param2[0], np.power(x, 2)) + \
    #     #     np.multiply(calibration.params_3_and_4[0][0], np.power(x, 3)) + \
    #     #     np.multiply(calibration.params_3_and_4[1][0], np.power(x, 4))
    #     #
    #     # handle_plot[0].set_ydata(y)
    # #
    #     wm = KeyPressManager.KeyPressManager.WindowManager(img)
    #     if wm.waitForKey(0.01, verbose=False):
    #         exit(0)
    # #
    opt.setVisualizationFunction(visualizationFunction, True)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # opt.fromXToData()
    # opt.callObjectiveFunction()
    # wm = KeyPressManager.KeyPressManager.WindowManager()
    # if wm.waitForKey():
    #     exit(0)
    #
    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-4, 'diff_step': 1e-4})

    wm = KeyPressManager.KeyPressManager.WindowManager()
    if wm.waitForKey():
        exit(0)
