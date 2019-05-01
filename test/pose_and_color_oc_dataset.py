#!/usr/bin/env python
"""
This example shows an optimizer working with a set of n cameras, changing their pose so that the reprojection error is
minimized.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import argparse  # to read command line arguments
from copy import deepcopy
from itertools import combinations

import numpy as np
import cv2
from functools import partial
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import KeyPressManager.KeyPressManager
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OCDatasetLoader.OCArucoDetector as OCArucoDetector
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.utilities as utilities

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()

    # Dataset loader arguments
    ap.add_argument("-p", "--path_to_images", help="path to the folder that contains the OC dataset", required=True)
    ap = OCDatasetLoader.addArguments(ap) # Dataset loader arguments
    ap = OptimizationUtils.addArguments(ap) # OptimizationUtils arguments
    args = vars(ap.parse_args())

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    dataset_loader = OCDatasetLoader.Loader(args)
    dataset_cameras = dataset_loader.loadDataset()
    print("dataset_cameras contains " + str(len(dataset_cameras.cameras)) + " cameras")

    aruco_detector = OCArucoDetector.ArucoDetector(args)
    dataset_arucos, dataset_cameras = aruco_detector.detect(dataset_cameras)

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('data_cameras', dataset_cameras)
    opt.addModelData('data_arucos', dataset_arucos)


    # ------------  Cameras -----------------
    # Each camera will have a position (tx,ty,tz) and a rotation (r1,r2,r3)
    # thus, the getter should return a list of size 6

    def getterCameraTranslation(data, cam_idx):
        return data.cameras[cam_idx].rgb.matrix[0:3, 3]


    def setterCameraTranslation(data, value, cam_idx):
        data.cameras[cam_idx].rgb.matrix[0:3, 3] = value


    def getterCameraRotation(data, cam_idx):
        matrix = data.cameras[cam_idx].rgb.matrix[0:3, 0:3]
        return utilities.matrixToRodrigues(matrix)


    def setterCameraRotation(data, value, cam_idx):
        matrix = utilities.rodriguesToMatrix(value)
        data.cameras[cam_idx].rgb.matrix[0:3, 0:3] = matrix


    # Add parameters related to the cameras
    for cam_idx, camera in enumerate(dataset_cameras.cameras):
        # Add the translation
        opt.pushParamVector3(group_name='C' + camera.name + '_t', data_key='data_cameras',
                             getter=partial(getterCameraTranslation, cam_idx=cam_idx),
                             setter=partial(setterCameraTranslation, cam_idx=cam_idx),
                             sufix=['x', 'y', 'z'])

        # Add the rotation
        opt.pushParamVector3(group_name='C' + camera.name + '_r', data_key='data_cameras',
                             getter=partial(getterCameraRotation, cam_idx=cam_idx),
                             setter=partial(setterCameraRotation, cam_idx=cam_idx),
                             sufix=['1', '2', '3'])


    # ------------  Arucos -----------------
    # Each aruco will only have the position (tx,ty,tz)
    # thus, the getter should return a list of size 3
    def getterArucoTranslation(data, aruco_id):
        return data.arucos[aruco_id][0:3, 3]


    def setterArucoTranslation(data, value, aruco_id):
        data.arucos[aruco_id][0:3, 3] = value


    # Add parameters related to the arucos
    for aruco_id, aruco in dataset_arucos.arucos.items():
        opt.pushParamVector3(group_name='A' + str(aruco_id), data_key='data_arucos',
                             getter=partial(getterArucoTranslation, aruco_id=aruco_id),
                             setter=partial(setterArucoTranslation, aruco_id=aruco_id),
                             sufix=['_tx', '_ty', '_tz'])

    opt.printParameters()

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------

    first_time = True


    def objectiveFunction(data):
        """
        :param data: points to the camera and aruco dataset
        :return: a vector of residuals
        """
        # Get the data
        data_cameras = data['data_cameras']
        data_arucos = data['data_arucos']

        # print("data_cameras\n" + str(data_cameras.cameras[0].rgb.matrix))
        # print("data_arucos" + str(data_arucos))

        errors = []
        # Cycle all cameras in the dataset
        for camera in data_cameras.cameras:
            # print("Cam " + str(camera.name))
            for aruco_id, aruco_detection in camera.rgb.aruco_detections.items():
                # print("Aruco " + str(aruco_id))
                # print("Pixel center coords (ground truth) = " + str(aruco_detection.center))  # ground truth

                # Find current position of aruco
                world_T_camera = np.linalg.inv(camera.rgb.matrix)

                # Extract the translation from the transform matrix and create a np array with a 4,1 point coordinate
                aruco_origin_world = data_arucos.arucos[aruco_id][0:4, 3]
                # print("aruco_origin_world = " + str(aruco_origin_world))

                aruco_origin_camera = np.dot(world_T_camera, aruco_origin_world)
                # print("aruco_origin_camera = " + str(aruco_origin_camera))

                pixs, valid_pixs, dists = utilities.projectToCamera(np.array(camera.rgb.camera_info.K).reshape((3, 3)),
                                                                    camera.rgb.camera_info.D,
                                                                    camera.rgb.camera_info.width,
                                                                    camera.rgb.camera_info.height,
                                                                    np.array(aruco_origin_camera,
                                                                            dtype=np.float).reshape((4, 1)))
                aruco_detection.projected = (pixs[0][0], pixs[1][0])
                global first_time
                if first_time:
                    aruco_detection.first_projection = aruco_detection.projected

                # print(aruco_detection.projected)
                error = euclidean(aruco_detection.center, aruco_detection.projected)
                # print("error = " + str(error))
                errors.append(error)

        first_time = False
        # Return the errors
        return errors


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for camera in dataset_cameras.cameras:
        for aruco_id, aruco_detection in camera.rgb.aruco_detections.items():
            params = opt.getParamsContainingPattern('C' + str(camera.name))
            params.extend(opt.getParamsContainingPattern('A' + str(aruco_id)))
            opt.pushResidual(name='C' + camera.name + 'A' + str(aruco_id), params=params)

    print('residuals = ' + str(opt.residuals))

    # ---------------------------------------
    # --- Compute the SPARSE MATRIX
    # ---------------------------------------
    opt.computeSparseMatrix()

    # ---------------------------------------
    # --- SETUP THE VISUALIZATION FUNCTION
    # ---------------------------------------
    # position the windows in the proper place
    for i, camera in enumerate(dataset_cameras.cameras):
        cv2.namedWindow('Cam ' + camera.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow('Cam ' + camera.name, 300 * i, 50)
        cv2.imshow('Cam ' + camera.name, camera.rgb.image)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    limit = 1.5
    ax.set_xlim3d(-limit, limit), ax.set_ylim3d(-limit, limit), ax.set_zlim3d(-limit, limit)
    ax.view_init(elev=122, azim=-87)

    # Draw world axis
    world_T_world = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float)
    utilities.drawAxis3D(ax, world_T_world, "world", axis_scale=0.7, line_width=3)

    # Draw cameras
    for camera in dataset_cameras.cameras:
        camera.handle_frame = utilities.drawAxis3D(ax, camera.rgb.matrix, "C" + camera.name, axis_scale=0.3,
                                                   line_width=2)
        # print("camera " + camera.name + " " + str(camera.handle_frame))

    # Draw Arucos
    dataset_arucos.handles = {}
    for aruco_id, transform in dataset_arucos.arucos.items():
        dataset_arucos.handles[aruco_id] = utilities.drawAxis3DOrigin(ax, transform, 'A' + str(aruco_id),
                                                                      line_width=1.0,
                                                                      fontsize=8,
                                                                      handles=None)
        print("aruco " + str(aruco_id) + "= " + str(dataset_arucos.handles[aruco_id]))

    wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    if wm.waitForKey(time_to_wait=None, verbose=True):
        exit(0)

    # Obtaining the 3D measurements for each camera
    # ---------------------------------------
    # --- Compute the list of points that will be used by the cost function
    # ---------------------------------------
    cam_pairs = []
    for cam_a, cam_b in combinations(dataset_cameras.cameras, 2):
        print(cam_a.name + ' with ' + cam_b.name)

        # get a list of 3D points by concatenating the 3D measurements of cam_a and cam_b
        print(cam_a.depth.vertices[:, 0::args['skip_vertices']])
        # exit(0)

        print('depth matrix=\n' + str(cam_a.depth.matrix))
        pts3D_in_map_a = np.dot(cam_a.depth.matrix, cam_a.depth.vertices[:, 0::args['skip_vertices']])
        print('pts3D_in_map_a=\n' + str(pts3D_in_map_a))
        pts3D_in_map_b = np.dot(cam_b.depth.matrix, cam_b.depth.vertices[:, 0::args['skip_vertices']])
        pts3D_in_map = np.concatenate([pts3D_in_map_a, pts3D_in_map_b], axis=1)

        # project 3D points to cam_a
        pts3D_in_cam_a = np.dot(np.linalg.inv(cam_a.rgb.matrix), pts3D_in_map)
        pts2D_a, pts_valid_a, pts_range_a = utilities.projectToCamera(
            np.array(cam_a.rgb.camera_info.K).reshape((3, 3)),
            cam_a.rgb.camera_info.D,
            cam_a.rgb.camera_info.width,
            cam_a.rgb.camera_info.height,
            pts3D_in_cam_a)

        pts2D_a = np.where(pts_valid_a, pts2D_a, 0)
        range_meas_a = cam_a.rgb.range_dense[pts2D_a[1, :], pts2D_a[0, :]]
        z_valid_a = abs(pts_range_a - range_meas_a) < args['z_inconsistency_threshold']

        # project 3D points to cam_b
        pts3D_in_cam_b = np.dot(np.linalg.inv(cam_b.rgb.matrix), pts3D_in_map)
        pts2D_b, pts_valid_b, pts_range_b = utilities.projectToCamera(
            np.array(cam_b.rgb.camera_info.K).reshape((3, 3)),
            cam_b.rgb.camera_info.D,
            cam_b.rgb.camera_info.width,
            cam_b.rgb.camera_info.height,
            pts3D_in_cam_b)

        pts2D_b = np.where(pts_valid_b, pts2D_b, 0)
        range_meas_b = cam_b.rgb.range_dense[pts2D_b[1, :], pts2D_b[0, :]]
        z_valid_b = abs(pts_range_b - range_meas_b) < args['z_inconsistency_threshold']

        # Compute masks for the valid projections
        mask = np.logical_and(pts_valid_a, pts_valid_b)
        z_mask = np.logical_and(z_valid_a, z_valid_b)
        final_mask = np.logical_and(mask, z_mask)

        if args['view_projected_vertices']:
            print("pts2d_a has " + str(np.count_nonzero(pts_valid_a)) + ' valid projections')
            print("pts2d_b has " + str(np.count_nonzero(pts_valid_b)) + ' valid projections')
            print("there are " + str(np.count_nonzero(mask)) + ' valid projection pairs')
            cam_a_image = deepcopy(cam_a.rgb.image)
            cam_b_image = deepcopy(cam_b.rgb.image)
            for i, val in enumerate(mask):
                if pts_valid_a[i] == True:
                    x0 = pts2D_a[0, i]
                    y0 = pts2D_a[1, i]
                    cv2.line(cam_a_image, (x0, y0), (x0, y0), color=(80, 80, 80), thickness=2)

                if pts_valid_b[i] == True:
                    x0 = pts2D_b[0, i]
                    y0 = pts2D_b[1, i]
                    cv2.line(cam_b_image, (x0, y0), (x0, y0), color=(80, 80, 80), thickness=2)

                if val == True:
                    x0 = pts2D_a[0, i]
                    y0 = pts2D_a[1, i]
                    cv2.line(cam_a_image, (x0, y0), (x0, y0), color=(0, 0, 210), thickness=2)

                    x0 = pts2D_b[0, i]
                    y0 = pts2D_b[1, i]
                    cv2.line(cam_b_image, (x0, y0), (x0, y0), color=(0, 0, 210), thickness=2)

                if z_mask[i] == True:
                    x0 = pts2D_a[0, i]
                    y0 = pts2D_a[1, i]
                    cv2.line(cam_a_image, (x0, y0), (x0, y0), color=(0, 210, 0), thickness=2)

                    x0 = pts2D_b[0, i]
                    y0 = pts2D_b[1, i]
                    cv2.line(cam_b_image, (x0, y0), (x0, y0), color=(0, 210, 0), thickness=2)

            cv2.namedWindow('cam_a', cv2.WINDOW_NORMAL)
            cv2.imshow('cam_a', cam_a_image)
            cv2.namedWindow('cam_b', cv2.WINDOW_NORMAL)
            cv2.imshow('cam_b', cam_b_image)

            if wm.waitForKey(time_to_wait=None, verbose=False):
                exit(0)

    # exit(0)


    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    def visualizationFunction(data):
        # Get the data
        data_cameras = data['data_cameras']
        data_arucos = data['data_arucos']

        # print("data_cameras\n" + str(data_cameras.cameras[0].rgb.matrix))

        for i, camera in enumerate(data_cameras.cameras):
            image = deepcopy(camera.rgb.image)
            # print("Cam " + str(camera.name))
            for aruco_id, aruco_detection in camera.rgb.aruco_detections.items():
                # print("Aruco " + str(aruco_id))
                # print("Pixel center coords (ground truth) = " + str(aruco_detection.center))  # ground truth

                utilities.drawSquare2D(image, aruco_detection.center[0], aruco_detection.center[1], 10,
                                       color=(0, 0, 255), thickness=2)

                # cv2.line(image, aruco_detection.center, aruco_detection.center, (0, 0, 255), 10)
                # print("Pixel center projected = " + str(aruco_detection.projected))  # ground truth

                if 0 < aruco_detection.projected[0] < camera.rgb.camera_info.width \
                        and 0 < aruco_detection.projected[1] < camera.rgb.camera_info.height:
                    cv2.line(image, aruco_detection.projected, aruco_detection.projected, (255, 0, 0), 10)

                # TODO: improve drawing first detection code
                if 0 < aruco_detection.first_projection[0] < camera.rgb.camera_info.width \
                        and 0 < aruco_detection.first_projection[1] < camera.rgb.camera_info.height:
                    cv2.line(image, aruco_detection.first_projection, aruco_detection.first_projection, (0, 255, 0), 10)

            cv2.imshow('Cam ' + camera.name, image)

        # Draw camera's axes
        for camera in data_cameras.cameras:
            utilities.drawAxis3D(ax=ax, transform=camera.rgb.matrix, text="C" + camera.name, axis_scale=0.3,
                                 line_width=2,
                                 handles=camera.handle_frame)

        # Draw Arucos
        for aruco_id, transform in data_arucos.arucos.items():
            utilities.drawAxis3DOrigin(ax, transform, 'A' + str(aruco_id), line_width=1.0,
                                       handles=data_arucos.handles[aruco_id])

        wm = KeyPressManager.KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)


    opt.setVisualizationFunction(visualizationFunction, args['view_optimization'], niterations=10)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    opt.x = opt.addNoiseToX(noise=0.01)
    opt.fromXToData()
    # opt.callObjectiveFunction()

    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization()

    print('ola')
    wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    if wm.waitForKey(time_to_wait=None, verbose=True):
        exit(0)

    print('ola')
