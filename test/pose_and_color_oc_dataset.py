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
import numpy as np
import cv2
from functools import partial
from numpy.linalg import norm
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
    ap.add_argument("-ext", "--image_extension", help="extension of the image files, e.g., jpg or png", default='jpg')
    ap.add_argument("-m", "--mesh_filename", help="full filename to input obj file, i.e. the 3D model", required=True)
    ap.add_argument("-i", "--path_to_intrinsics", help="path to intrinsics yaml file", required=True)
    ap.add_argument("-ucci", "--use_color_corrected_images", help="Use previously color corrected images",
                    action='store_true', default=False)
    ap.add_argument("-si", "--skip_images", help="skip images. Useful for fast testing", type=int, default=1)
    ap.add_argument("-vri", "--view_range_image", help="visualize sparse and dense range images", action='store_true',
                    default=False)
    ap.add_argument("-ms", "--marker_size", help="Size in meters of the aruco markers in the images", type=float,
                    required=True)
    ap.add_argument("-vad", "--view_aruco_detections", help="visualize aruco detections in the camera images",
                    action='store_true',
                    default=False)
    ap.add_argument("-va3d", "--view_aruco_3d", help="visualize aruco detections in a 3d window", action='store_true',
                    default=False)
    ap.add_argument("-va3dpc", "--view_aruco_3d_per_camera",
                    help="visualize aruco detections in a 3d window showing all the aruco detections (plot becomes quite dense)",
                    action='store_true',
                    default=False)
    # OptimizationUtils arguments
    ap.add_argument("-sv", "--skip_vertices", help="skip vertices. Useful for fast testing", type=int, default=1)
    ap.add_argument("-z", "--z_inconsistency_threshold", help="threshold for max z inconsistency value", type=float,
                    default=0.05)
    ap.add_argument("-vpv", "--view_projected_vertices", help="visualize projections of vertices onto images",
                    action='store_true', default=False)
    ap.add_argument("-vo", "--view_optimization", help="...", action='store_true', default=False)

    args = vars(ap.parse_args())
    print(args)

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
    def projectToPixel(intrinsic_matrix, distortion, width, height, pts):
        """
        Projects a list of points to the camera defined transform, intrinsics and distortion
        :param transform: a 4x4 homogeneous coordinates matrix which transforms from the world frame to the camera frame
        :param intrinsic_matrix: 3x3 intrinsic camera matrix
        :param distortion: should be as follows: (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])
        :param width: the image width
        :param height: the image height
        :param pts_world: a list of point coordinates (in the world frame) with the following format
        :return: a list of pixel coordinates with the same lenght as pts
        """

        _, n_pts = pts.shape

        # Project the 3D points in the camera's frame to image pixels
        # From https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        pixs = np.zeros((2, n_pts), dtype=np.int)

        k1, k2, p1, p2, k3 = distortion
        # fx, _, cx, _, fy, cy, _, _, _ = intrinsic_matrix
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        x = pts[0, :]
        y = pts[1, :]
        z = pts[2, :]

        dists = norm(pts[0:3, :], axis=0)  # compute distances from point to camera
        xl = np.divide(x, z)  # compute homogeneous coordinates
        yl = np.divide(y, z)  # compute homogeneous coordinates
        r2 = xl ** 2 + yl ** 2  # r square (used multiple times bellow)
        xll = xl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + 2 * p1 * xl * yl + p2 * (r2 + 2 * xl ** 2)
        yll = yl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + p1 * (r2 + 2 * yl ** 2) + 2 * p2 * xl * yl
        pixs[0, :] = fx * xll + cx
        pixs[1, :] = fy * yll + cy

        # Compute mask of valid projections
        valid_z = z > 0
        valid_xpix = np.logical_and(pixs[0, :] >= 0, pixs[0, :] < width)
        valid_ypix = np.logical_and(pixs[1, :] >= 0, pixs[1, :] < height)
        valid_pixs = np.logical_and(valid_z, np.logical_and(valid_xpix, valid_ypix))
        return pixs, valid_pixs, dists


    first_time = True


    def objectiveFunction(data):
        """
        Computes the vector of errors. Each error is associated with a camera, ans is computed from the Euclidean distance
        between the projected coordinates of aruco centers and the coordinates given by the detection of the aruco in the image.
        :param data: points to the camera and aruco dataset
        :return: a vector of resuduals
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

                pixs, valid_pixs, dists = projectToPixel(np.array(camera.rgb.camera_info.K).reshape((3, 3)),
                                                         camera.rgb.camera_info.D,
                                                         camera.rgb.camera_info.width,
                                                         camera.rgb.camera_info.height,
                                                         np.array(aruco_origin_camera, dtype=np.float).reshape((4, 1)))
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


    opt.setVisualizationFunction(visualizationFunction, n_iterations=10)

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
