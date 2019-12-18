#!/usr/bin/env python
"""

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
import KeyPressManager.KeyPressManager as KeyPressManager
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OCDatasetLoader.OCArucoDetector as OCArucoDetector
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.utilities as utilities

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------
# For some awkward reason the local point clouds (ply files) are stored in opengl coordinates.
# This matrix puts the coordinate frames back in opencv fashion
opencv2opengl = np.zeros((4, 4))
opencv2opengl[0, :] = [1, 0, 0, 0]
opencv2opengl[1, :] = [0, 0, 1, 0]
opencv2opengl[2, :] = [0, -1, 0, 0]
opencv2opengl[3, :] = [0, 0, 0, 1]

# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap = OCDatasetLoader.addArguments(ap)  # Dataset loader arguments
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    args = vars(ap.parse_args())

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    dataset_loader = OCDatasetLoader.Loader(args)
    dataset_cameras = dataset_loader.loadDataset()
    aruco_detector = OCArucoDetector.ArucoDetector(args)
    dataset_arucos, dataset_cameras = aruco_detector.detect(dataset_cameras)
    print("dataset_cameras contains " + str(len(dataset_cameras.cameras)) + " cameras")

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('data_cameras', dataset_cameras)
    opt.addDataModel('data_arucos', dataset_arucos)


    # ------------  Cameras -----------------
    # Each camera will have a position (tx,ty,tz) and a rotation (r1,r2,r3)
    # thus, the getter should return a list of size 6

    # Add parameters related to the cameras
    for camera_idx, camera in enumerate(dataset_cameras.cameras):
        # Add the translation
        opt.pushParamV3(group_name='C' + camera.name + '_t', data_key='data_cameras',
                        getter=partial(
                            lambda data, cam_idx: data.cameras[cam_idx].rgb.getTranslation(), cam_idx=camera_idx),
                        setter=partial(
                            lambda data, value, cam_idx: data.cameras[cam_idx].rgb.setTranslation(value),
                            cam_idx=camera_idx),
                        suffix=['x', 'y', 'z'])

        # Add the rotation
        opt.pushParamV3(group_name='C' + camera.name + '_r', data_key='data_cameras',
                        getter=partial(
                            lambda data, cam_idx: data.cameras[cam_idx].rgb.getRodrigues(), cam_idx=camera_idx),
                        setter=partial(
                            lambda data, value, cam_idx: data.cameras[cam_idx].rgb.setRodrigues(value),
                            cam_idx=camera_idx),
                        suffix=['1', '2', '3'])


    # ------------  Arucos -----------------
    # Each aruco will only have the position (tx,ty,tz)
    # thus, the getter should return a list of size 3
    def getterArucoTranslation(data, aruco_id):
        return data.arucos[aruco_id].matrix[0:3, 3]


    def setterArucoTranslation(data, value, aruco_id):
        data.arucos[aruco_id].matrix[0:3, 3] = value


    # Add parameters related to the arucos
    for aruco_id, aruco in dataset_arucos.arucos.items():
        opt.pushParamV3(group_name='A' + str(aruco_id), data_key='data_arucos',
                        getter=partial(getterArucoTranslation, aruco_id=aruco_id),
                        setter=partial(setterArucoTranslation, aruco_id=aruco_id),
                        suffix=['_tx', '_ty', '_tz'])

    opt.printParameters()

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    first_time = True


    def objectiveFunction(data):
        """
        Computes the vector of errors. Each error is associated with a camera, ans is computed from the Euclidean distance
        between the projected coordinates of aruco centers and the coordinates given by the detection of the aruco in the image.
        :param data: points to the camera and aruco dataset
        :return: a vector of residuals
        """
        # Get the data
        data_cameras = data['data_cameras']
        data_arucos = data['data_arucos']

        errors = []

        # # Cycle all cameras in the dataset
        # for _camera in data_cameras.cameras:
        #     # print("Cam " + str(camera.name))
        #     for _aruco_id, _aruco_detection in _camera.rgb.aruco_detections.items():
        #         # print("Aruco " + str(aruco_id))
        #         # print("Pixel center coords (ground truth) = " + str(aruco_detection.center))  # ground truth
        #
        #         # Find current position of aruco
        #         world_T_camera = np.linalg.inv(_camera.rgb.matrix)
        #         # print('world_to_camera = ' + str(world_T_camera))
        #
        #         # Extract the translation from the transform matrix and create a np array with a 4,1 point coordinate
        #         aruco_origin_world = data_arucos.arucos[_aruco_id].getTranslation(homogeneous=True)
        #         # print("aruco_origin_world = " + str(aruco_origin_world))
        #
        #         aruco_origin_camera = np.dot(world_T_camera, aruco_origin_world)
        #         # print("aruco_origin_camera = " + str(aruco_origin_camera))
        #
        #         pixs, valid_pixs, dists = utilities.projectToCamera(np.array(_camera.rgb.camera_info.K).reshape((3, 3)),
        #                                                             _camera.rgb.camera_info.D,
        #                                                             _camera.rgb.camera_info.width,
        #                                                             _camera.rgb.camera_info.height,
        #                                                             np.array(aruco_origin_camera,
        #                                                                      dtype=np.float).reshape((4, 1)))
        #         _aruco_detection.projected = (pixs[0][0], pixs[1][0])
        #
        #         global first_time
        #         if first_time:
        #             _aruco_detection.first_projection = _aruco_detection.projected
        #
        #         # print("aruco " + str(aruco_id) + " = " + str(aruco_detection.center))
        #         error = euclidean(_aruco_detection.center, _aruco_detection.projected)
        #         # print("error = " + str(error))
        #         # if error > 150:
        #         #     print(camera.name + 'is an outlier')
        #         errors.append(error)
        #
        # first_time = False
        # # Return the errors
        # return errors

        cam_pairs = []
        for cam_a, cam_b in combinations(dataset_cameras.cameras, 2):
            print(cam_a.name + ' with ' + cam_b.name)

            # ---------------------------------------------------------------------------------------
            # STEP 1: Get a list of 3D points by concatenating the 3D measurements of cam_a and cam_b
            # ---------------------------------------------------------------------------------------
            pts3D_in_map = np.concatenate([np.dot(opencv2opengl, cam_a.depth.vertices[:, 0::args['skip_vertices']]),
                                           np.dot(opencv2opengl, cam_b.depth.vertices[:, 0::args['skip_vertices']])],
                                          axis=1)

            # ---------------------------------------------------------------------------------------
            # STEP 2: project 3D points to cam_a and cam_b
            # ---------------------------------------------------------------------------------------
            for cam in [cam_a, cam_b]:
                pts3D_in_cam = cam.rgb.transformToCamera(pts3D_in_map)
                cam.pts2D, cam.pts_valid, cam.pts_range = cam_a.rgb.projectToCamera(pts3D_in_cam)
                cam.pts2D = np.where(cam.pts_valid, cam.pts2D, 0)  # set to 0 those not marked by valid mask
                cam.range_meas = cam_a.rgb.range_dense[(cam.pts2D[1, :]).astype(np.int), (cam.pts2D[0, :]).astype(int)]
                cam.z_valid = abs(cam.pts_range - cam.range_meas) < args['z_inconsistency_threshold']

            # ---------------------------------------------------------------------------------------
            # STEP 3: Verify valid projections on both cameras
            # ---------------------------------------------------------------------------------------
            mask = np.logical_and(cam_a.pts_valid, cam_b.pts_valid)
            z_mask = np.logical_and(cam_a.z_valid, cam_b.z_valid)
            final_mask = np.logical_and(mask, z_mask)

            if args['view_projected_vertices']:

                print("There are " + str(np.count_nonzero(mask)) + ' valid projection pairs')

                for cam, name in zip([cam_a, cam_b], ['cam_a', 'cam_b']):

                    print('pts2d ' + name + ' has ' + str(np.count_nonzero(cam.pts_valid)) + ' valid projections')
                    cam_image = deepcopy(cam.rgb.image)
                    for i, val in enumerate(mask):
                        x0 = int(cam.pts2D[0, i])
                        y0 = int(cam.pts2D[1, i])

                        if cam.pts_valid[i]:
                            cv2.line(cam_image, (x0, y0), (x0, y0), color=(80, 80, 80), thickness=3)

                        if val:
                            cv2.line(cam_image, (x0, y0), (x0, y0), color=(0, 0, 210), thickness=2)

                        if final_mask[i]:
                            cv2.line(cam_image, (x0, y0), (x0, y0), color=(0, 210, 0), thickness=2)

                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.imshow(name, cam_image)

                wm = KeyPressManager.WindowManager()
                if wm.waitForKey(time_to_wait=None, verbose=False):
                    exit(0)

            # Compute the error with the valid projections
            colors_a = cam_a.rgb.image_changed[
                (cam_a.pts2D[1, cam_a.valid_mask]).astype(np.int), (cam_a.pts2D[0, cam_a.valid_mask]).astype(np.int)]
            colors_b = cam_b.rgb.image_changed[
                (cam_b.pts2D[1, cam_b.valid_mask]).astype(np.int), (cam_b.pts2D[0, cam_b.valid_mask]).astype(np.int)]
            error = np.linalg.norm(colors_a.astype(np.float) - colors_b.astype(np.float), ord=2, axis=1)
            # utilities.printNumPyArray({'colors_a': colors_a, 'colors_b': colors_b, 'error': error})

            utilities.drawProjectionErrors(cam_a.rgb.image_changed, cam_a.pts2D[:, cam_a.valid_mask], cam_b.rgb.image_changed,
                                           cam_b.pts2D_b[:, cam_b.valid_mask],
                                           error, cam_a.name + '_' + cam_b.name, skip=10)

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
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    if args['view_optimization']:
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
        for aruco_id, aruco in dataset_arucos.arucos.items():
            dataset_arucos.handles[aruco_id] = utilities.drawAxis3DOrigin(ax, aruco.matrix, 'A' + str(aruco_id),
                                                                          line_width=1.0,
                                                                          fontsize=8,
                                                                          handles=None)
            # print("aruco " + str(aruco_id) + "= " + str(dataset_arucos.handles[aruco_id]))

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(time_to_wait=0.01, verbose=True):
            exit(0)


    # ---------------------------------------
    # --- DEFINE THE VISUALIZATION FUNCTION
    # ---------------------------------------
    def visualizationFunction(data):
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text
        # Get the data
        data_cameras = data['data_cameras']
        data_arucos = data['data_arucos']

        # print("data_cameras\n" + str(data_cameras.cameras[0].rgb.matrix))

        for i, _camera in enumerate(data_cameras.cameras):
            image = deepcopy(_camera.rgb.image)
            # print("Cam " + str(camera.name))
            for _aruco_id, _aruco_detection in _camera.rgb.aruco_detections.items():
                # print("Aruco " + str(aruco_id))
                # print("Pixel center coords (ground truth) = " + str(aruco_detection.center))  # ground truth

                utilities.drawSquare2D(image, _aruco_detection.center[0], _aruco_detection.center[1], 10,
                                       color=(0, 0, 255), thickness=2)

                cv2.putText(image, "Id:" + str(_aruco_id), _aruco_detection.center, font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                # cv2.line(image, aruco_detection.center, aruco_detection.center, (0, 0, 255), 10)
                # print("Pixel center projected = " + str(aruco_detection.projected))  # ground truth

                if 0 < _aruco_detection.projected[0] < _camera.rgb.camera_info.width \
                        and 0 < _aruco_detection.projected[1] < _camera.rgb.camera_info.height:
                    x = int(_aruco_detection.projected[0])
                    y = int(_aruco_detection.projected[1])
                    # cv2.line(image, aruco_detection.projected, aruco_detection.projected, (255, 0, 0), 10)
                    cv2.line(image, (x, y), (x, y), (255, 0, 0), 10)

                # TODO: debug drawing first detection code
                if 0 < _aruco_detection.first_projection[0] < _camera.rgb.camera_info.width \
                        and 0 < _aruco_detection.first_projection[1] < _camera.rgb.camera_info.height:
                    x = int(_aruco_detection.first_projection[0])
                    y = int(_aruco_detection.first_projection[1])
                    # cv2.line(image, aruco_detection.first_projection, aruco_detection.first_projection, (0, 255, 0), 10)
                    cv2.line(image, (x, y), (x, y), (0, 255, 0), 10)

            cv2.imshow('Cam ' + _camera.name, image)

        # Draw camera's axes
        for _camera in data_cameras.cameras:
            utilities.drawAxis3D(ax=ax, transform=_camera.rgb.matrix, text="C" + _camera.name, axis_scale=0.3,
                                 line_width=2,
                                 handles=_camera.handle_frame)

        # Draw Arucos
        for _aruco_id, _aruco in data_arucos.arucos.items():
            utilities.drawAxis3DOrigin(ax, _aruco.matrix, 'A' + str(_aruco_id), line_width=1.0,
                                       handles=data_arucos.handles[_aruco_id])

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(time_to_wait=0.01, verbose=True):
            exit(0)


    # def visualizationFunction(data):
    #     # Get the data
    #     data_cameras = data['data_cameras']
    #     data_arucos = data['data_arucos']
    #
    #     # print("data_cameras\n" + str(data_cameras.cameras[0].rgb.matrix))
    #
    #     for i, camera in enumerate(data_cameras.cameras):
    #         image = deepcopy(camera.rgb.image)
    #         # print("Cam " + str(camera.name))
    #         for aruco_id, aruco_detection in camera.rgb.aruco_detections.items():
    #             # print("Aruco " + str(aruco_id))
    #             # print("Pixel center coords (ground truth) = " + str(aruco_detection.center))  # ground truth
    #
    #             utilities.drawSquare2D(image, aruco_detection.center[0], aruco_detection.center[1], 10,
    #                                    color=(0, 0, 255), thickness=2)
    #
    #             # cv2.line(image, aruco_detection.center, aruco_detection.center, (0, 0, 255), 10)
    #             # print("Pixel center projected = " + str(aruco_detection.projected))  # ground truth
    #
    #             if 0 < aruco_detection.projected[0] < camera.rgb.camera_info.width \
    #                     and 0 < aruco_detection.projected[1] < camera.rgb.camera_info.height:
    #                 cv2.line(image, aruco_detection.projected, aruco_detection.projected, (255, 0, 0), 10)
    #
    #             # TODO: improve drawing first detection code
    #             if 0 < aruco_detection.first_projection[0] < camera.rgb.camera_info.width \
    #                     and 0 < aruco_detection.first_projection[1] < camera.rgb.camera_info.height:
    #                 cv2.line(image, aruco_detection.first_projection, aruco_detection.first_projection, (0, 255, 0), 10)
    #
    #         cv2.imshow('Cam ' + camera.name, image)
    #
    #     # Draw camera's axes
    #     for camera in data_cameras.cameras:
    #         utilities.drawAxis3D(ax=ax, transform=camera.rgb.matrix, text="C" + camera.name, axis_scale=0.3,
    #                              line_width=2,
    #                              handles=camera.handle_frame)
    #
    #     # Draw Arucos
    #     for aruco_id, transform in data_arucos.arucos.items():
    #         utilities.drawAxis3DOrigin(ax, transform, 'A' + str(aruco_id), line_width=1.0,
    #                                    handles=data_arucos.handles[aruco_id])
    #
    #     wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    #     if wm.waitForKey(0.01, verbose=False):
    #         exit(0)

    opt.setVisualizationFunction(visualizationFunction, args['view_optimization'], niterations=10)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # opt.x = opt.addNoiseToX(noise=0.01)
    # opt.fromXToData()
    # opt.callObjectiveFunction()

    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5, 'diff_step': 1e-4})

    wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    if wm.waitForKey(time_to_wait=None, verbose=True):
        exit(0)
