#!/usr/bin/env python
"""
This example shows an optimizer working with a set of n cameras, changing their pose so that the reprojection error is
minimized.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import argparse  # to read command line arguments
from collections import namedtuple
from copy import deepcopy
from itertools import combinations
import numpy as np
import cv2
from functools import partial
import random
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
from numpy.linalg import norm

import OptimizationUtils.OptimizationUtils as OptimizationUtils


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

def keyPressManager():
    print('keyPressManager.\nPress "c" to continue or "q" to abort.')
    while True:
        plt.waitforbuttonpress(0.01)
        key = cv2.waitKey(15)
        if key == ord('c'):
            print('Pressed "c". Continuing.')
            break
        elif key == ord('q'):
            print('Pressed "q". Aborting.')
            exit(0)


def drawAxis3D(ax, transform, text, axis_scale=0.1, line_width=1.0):
    pt_origin = np.array([[0, 0, 0, 1]], dtype=np.float).transpose()
    x_axis = np.array([[0, 0, 0, 1], [axis_scale, 0, 0, 1]], dtype=np.float).transpose()
    y_axis = np.array([[0, 0, 0, 1], [0, axis_scale, 0, 1]], dtype=np.float).transpose()
    z_axis = np.array([[0, 0, 0, 1], [0, 0, axis_scale, 1]], dtype=np.float).transpose()

    pt_origin = np.dot(transform, pt_origin)
    x_axis = np.dot(transform, x_axis)
    y_axis = np.dot(transform, y_axis)
    z_axis = np.dot(transform, z_axis)

    ax.plot(x_axis[0, :], x_axis[1, :], x_axis[2, :], 'r-', linewidth=line_width)
    ax.plot(y_axis[0, :], y_axis[1, :], y_axis[2, :], 'g-', linewidth=line_width)
    ax.plot(z_axis[0, :], z_axis[1, :], z_axis[2, :], 'b-', linewidth=line_width)
    ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black')  # , size=15, zorder=1


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
    num_cameras = len(dataset_cameras.cameras)
    print(num_cameras)


    # ---------------------------------------
    # --- Utility functions
    # ---------------------------------------
    def matrixToRodrigues(T):
        rods, _ = cv2.Rodrigues(T[0:3, 0:3])
        rods = rods.transpose()
        return rods[0]


    def rodriguesToMatrix(r):
        rod = np.array(r, dtype=np.float)
        matrix = cv2.Rodrigues(rod)
        return matrix[0]


    def traslationRodriguesToTransform(translation, rodrigues):
        R = rodriguesToMatrix(rodrigues)
        T = np.zeros((4, 4), dtype=np.float)
        T[0:3, 0:3] = R
        T[0:3, 3] = translation
        T[3, 3] = 1
        return T


    # ---------------------------------------
    # --- Detect ARUCOS
    # ---------------------------------------

    import cv2.aruco  # Aruco Markers

    ArucoT = namedtuple('ArucoT', 'id center translation rodrigues camera_T_aruco')


    class ArucoConfiguration:
        def __init__(self):
            pass


    dataset_arucos = ArucoConfiguration()
    dataset_arucos.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    dataset_arucos.parameters = cv2.aruco.DetectorParameters_create()

    dataset_arucos.markerSize = 0.082
    dataset_arucos.distortion = np.array(dataset_cameras.cameras[0].rgb.camera_info.D)
    dataset_arucos.intrinsics = np.reshape(dataset_cameras.cameras[0].rgb.camera_info.K, (3, 3))
    dataset_arucos.world_T_aruco = {}

    for i, camera in enumerate(dataset_cameras.cameras):
        camera.rgb.arucos = {}
        print("Camera " + camera.name)

        image = cv2.cvtColor(camera.rgb.image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(image, dataset_arucos.aruco_dict,
                                                  parameters=dataset_arucos.parameters)

        # Estimate pose of each marker
        rotationVecs, translationVecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, dataset_arucos.markerSize,
                                                                               dataset_arucos.intrinsics,
                                                                               dataset_arucos.distortion)

        for j, id in enumerate(ids):
            id = id[0]  # strange thing from cv2
            # if id < 550:  # to test with less arucos
            #     continue

            print("Aruco " + str(id))

            my_corners = corners[j][0][:]  # strange thing from cv2
            x = []
            y = []
            for corner in my_corners:
                x.append(corner[0])
                y.append(corner[1])
            center = (np.average(x), np.average(y))

            rodrigues = np.array(rotationVecs[j][0])
            translation = np.array(translationVecs[j][0])
            camera_T_aruco = np.linalg.inv(traslationRodriguesToTransform(translation, rodrigues))
            # TODO check this
            camera.rgb.arucos[id] = (ArucoT(id, center, translation, rodrigues, camera_T_aruco))
            if not id in dataset_arucos.world_T_aruco:  # add only if there is still not an estimate (made by another camera)
                dataset_arucos.world_T_aruco[id] = np.dot(camera.rgb.matrix, np.linalg.inv(camera.rgb.arucos[id].camera_T_aruco))

    # Display
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text
    for i, camera in enumerate(dataset_cameras.cameras):
        image = deepcopy(camera.rgb.image)
        corners, ids, _ = cv2.aruco.detectMarkers(image, dataset_arucos.aruco_dict,
                                                  parameters=dataset_arucos.parameters)

        for key, aruco in camera.rgb.arucos.items():
            cv2.aruco.drawAxis(image, dataset_arucos.intrinsics, dataset_arucos.distortion, aruco.rodrigues,
                               aruco.translation, 0.05)  # Draw Axis
            cv2.putText(image, "Id:" + str(aruco.id), aruco.center, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.aruco.drawDetectedMarkers(image, corners)
        cv2.namedWindow('cam' + str(i), cv2.WINDOW_NORMAL)
        cv2.imshow('cam' + str(i), image)

    cv2.waitKey(500)
    # cv2.waitKey(0)
    print("Displaying aruco detections for all images. Press any key to continue.")
    # for i, camera in enumerate(dataset_cameras.cameras):
    #     cv2.destroyWindow('cam' + str(i))

    # import matplotlib.pyplot as plt  # Library to do plots 2D
    # import matplotlib.animation as animation  # Animate plots
    # from matplotlib import gridspec  # Make subplott
    # from mpl_toolkits.mplot3d import Axes3D  # Library to do plots 3D

    # Draw projection 3D
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    limit = 1.8
    ax.set_xlim3d(-limit,limit)
    ax.set_ylim3d(-limit,limit)
    ax.set_zlim3d(-limit,limit)

    # Draw world axis
    world_T_world = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float)
    drawAxis3D(ax, world_T_world, "world", axis_scale=0.7, line_width=3)

    # Draw camera's axes
    for camera in dataset_cameras.cameras:
        # camera.rgb.matrix -> camera_T_world
        drawAxis3D(ax, camera.rgb.matrix, "C" + camera.name, axis_scale=0.3, line_width=2)
        for id, aruco in camera.rgb.arucos.items():
            # print("aruco id=" + str(id))
            # print("camera_T_aruco =\n " + str(aruco.camera_T_aruco))
            # print("camera_T_aruco =\n " + str(np.linalg.inv(aruco.camera_T_aruco)))

            aruco_T_world = np.dot(camera.rgb.matrix, np.linalg.inv(aruco.camera_T_aruco))
            # print("aruco_T_world =\n " + str(aruco_T_world))

            # drawAxis3D(ax, aruco_T_world, "A"+ str(id) + "_C" + camera.name)
            drawAxis3D(ax, aruco_T_world, "A"+ str(id))
            # drawAxis3D(ax, aruco_T_world, "")

    keyPressManager() #wait for button press before continuing

    exit(0)



    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('data_cameras', dataset_cameras)
    opt.addModelData('data_arucos', dataset_arucos)


    # Add new rodrigues rotation to each camera
    # for camera in dataset.cameras:
    #     camera.rgb.rodrigues = matrixToRodrigues(camera.rgb.matrix[0:3, 0:3])

    def setterCameraTranslation(data, value, i, axis='x'):
        data.cameras[i].rgb.matrix['xyz'.index(axis), 3] = value


    def getterCameraTranslation(data, i, axis='x'):
        return data.cameras[i].rgb.matrix['xyz'.index(axis), 3]


    # def setter_rotation(dataset, value, i, axis='x'):
    #     rodriguesToMatrix(value)
    #     datasetrotation.rgb.matrix['xyz'.index(axis), 3] = value
    #
    #
    # def getter_rotation(dataset, i, axis='x'):
    #     r = matrixToRodrigues(dataset.cameras[i].rgb.matrix[0:3, 0:3])
    #     return r['xyz'.index(axis)]

    for i, camera in enumerate(dataset_cameras.cameras):

        # Translation
        for axis in 'xyz':
            opt.pushScalarParam(name='C' + camera.name + '_t' + axis, data_key='data_cameras',
                                getter=partial(getterCameraTranslation, i=i, axis=axis),
                                setter=partial(setterCameraTranslation, i=i, axis=axis))
        #
        # # Rotation
        # for axis in 'xyz':
        #     opt.pushScalarParam(name='cam' + camera.name + '_t' + axis, model_key='dataset',
        #             getter=partial(getter_translation, i=i, axis=axis),
        #             setter=partial(setter_translation, i=i, axis=axis))

    # Get a list of the visible aruco ids
    dataset_arucos.ids = []
    for camera in dataset_cameras.cameras:
        for key, aruco in camera.rgb.arucos.items():
            if not key in dataset_arucos.ids:
                dataset_arucos.ids.append(aruco.id)

    dataset_arucos.ids.sort()  # to have a better order of the ids
    print('list of visible ids:' + str(dataset_arucos.ids))


    # print('arucoConfiguration.worldTaruco =' + str(arucoConfiguration.worldTaruco))

    def setterArucoTranslation(data, value, id, axis='x'):
        data.world_T_aruco[id]['xyz'.index(axis), 3] = value


    def getterArucoTranslation(data, id, axis='x'):
        return data.world_T_aruco[id]['xyz'.index(axis), 3]


    # Create the parameters for the aruco's positions
    for id in dataset_arucos.ids:
        # Translation
        for axis in 'xyz':
            opt.pushScalarParam(name='A' + str(id) + '_t' + axis, data_key='data_arucos',
                                getter=partial(getterArucoTranslation, id=id, axis=axis),
                                setter=partial(setterArucoTranslation, id=id, axis=axis))

    opt.printXAndModel()


    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def projectToPixel(transform, intrinsic_matrix, distortion, width, height, pts_world):
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

        # Project the pts_world to the camera reference system
        pts = np.dot(np.linalg.inv(transform), pts_world)
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


    def objectiveFunction(data):

        # Get the data
        data_cameras = data['data_cameras']
        data_arucos = data['data_arucos']

        # Cycle all cameras in the dataset
        for camera in data_cameras.cameras:
            print("Cam " + str(camera.name))
            for key, aruco in camera.rgb.arucos.items():
                print("Aruco " + str(aruco.id))
                print("Pixel center coords (ground truth) = " + str(aruco.center))  # ground truth

                # Find current position of aruco

                world_T_camera = camera.rgb.matrix
                camera_T_aruco = camera.rgb.arucos[aruco.id].camera_T_aruco

                print("Transform world_T_camera:\n" + str(world_T_camera))
                print("Transform camera_T_aruco:\n" + str(camera_T_aruco))
                print("Transform world_T_aruco:\n" + str(np.dot(camera_T_aruco, np.linalg.inv(world_T_camera))))

                print("Transform data_arucos:\n" + str(data_arucos.world_T_aruco[aruco.id]))

                # print("Transform world2Aruco:\n" + str(data_arucos.pose[aruco.id]))
                # print("Transform world2Aruco:\n" + str(camera.rgb.matrix * np.linalg.inv(camera.rgb.arucos[aruco.id].camera_T_aruco)))

                # Extract the translation from the transform matrix and create a np array with a 4,1 point coordinate
                point3D_world = data_arucos.world_T_aruco[aruco.id][0:4, 3]
                print("point3D_world= " + str(point3D_world))

                point3D_camera = np.dot(world_T_camera, point3D_world)
                # point3D_camera = np.dot(np.linalg.inv(world_T_camera), point3D_world)
                print("point3D_camera= " + str(point3D_camera))

                # pixs = projectToPixel(np.linalg.inv(camera.rgb.matrix), data_arucos.intrinsics, data_arucos.distortion, camera.rgb.camera_info.width, camera.rgb.camera_info.height, np.array(point3D, dtype=np.float).reshape((4,1)))
                # print(pixs)
                # camera.rgb.

            pass

            cv2.waitKey(0)
            exit(0)

        # Compute all the pair wise combinations of the set of cameras
        # Each element in the vector of errors is the difference of the average color for the combination
        error = []
        # for cam_a, cam_b in combinations(dataset.cameras, 2):
        #     error.append(cam_a.rgb.avg_changed - cam_b.rgb.avg_changed)

        return error


    opt.setObjectiveFunction(objectiveFunction)
    opt.callObjectiveFunction()  # Just for testing

    exit(0)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------

    for cam_a, cam_b in combinations(dataset_cameras.cameras, 2):
        opt.pushResidual(name='c' + cam_a.name + '-c' + cam_b.name, params=['bias_' + cam_a.name, 'bias_' + cam_b.name])

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()


    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------
    def visualizationFunction(model):
        # Get the dataset from the model dictionary
        dataset = model['dataset']

        for i, camera in enumerate(dataset.cameras):
            cv2.namedWindow('Initial Cam ' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('Initial Cam ' + str(i), camera.rgb.image)

        for i, camera in enumerate(dataset.cameras):
            cv2.namedWindow('Changed Cam ' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('Changed Cam ' + str(i), camera.rgb.image_changed)
        cv2.waitKey(20)


    opt.setVisualizationFunction(visualizationFunction)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    opt.fromXToModel()
    opt.callObjectiveFunction()

    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization()
