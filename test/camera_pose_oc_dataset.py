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
import OptimizationUtils.OptimizationUtils as OptimizationUtils


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

def keyPressManager(self):
    print('keyPressManager.\nPress "c" to continue or "q" to abort.')
    while True:
        key = cv2.waitKey(3)
        if key == ord('c'):
            print('Pressed "c". Continuing.')
            break
        elif key == ord('q'):
            print('Pressed "q". Aborting.')
            exit(0)


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
    dataset = dataset_loader.loadDataset()
    num_cameras = len(dataset.cameras)
    print(num_cameras)

    # ---------------------------------------
    # --- Detect ARUCOS
    # ---------------------------------------
    import cv2.aruco  # Aruco Markers
    markerSize = 0.082
    distortion = np.array(dataset.cameras[0].rgb.camera_info.D)
    intrinsics = np.reshape(dataset.cameras[0].rgb.camera_info.K,(3,3))

    class Detections:
        def __init__(self):
            pass

    detections = Detections()
    detections.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    detections.parameters = cv2.aruco.DetectorParameters_create()

    ArucoT = namedtuple('ArucoT', 'id center rodrigues translation')


    for i, camera in enumerate(dataset.cameras):
        camera.rgb.arucos = []

        image = cv2.cvtColor(camera.rgb.image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(image, detections.aruco_dict, parameters=detections.parameters)

        # Estimate pose of each marker
        rotationVecs, translationVecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSize, intrinsics, distortion)

        for j, id in enumerate(ids):
            id = id[0] #strange thing from cv2

            my_corners = corners[j][0][:]
            x = []
            y = []
            for corner in my_corners:
                x.append(corner[0])
                y.append(corner[1])
            center = (np.average(x), np.average(y))

            rodrigues =tuple(rotationVecs[j][0])
            translation =tuple(translationVecs[j][0])

            camera.rgb.arucos.append(ArucoT(id, center, rodrigues, translation))


    # Display
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text
    for i, camera in enumerate(dataset.cameras):
        image = deepcopy(camera.rgb.image)
        corners, ids, _ = cv2.aruco.detectMarkers(image, detections.aruco_dict, parameters=detections.parameters)

        for aruco in camera.rgb.arucos:
            cv2.aruco.drawAxis(image, intrinsics, distortion, aruco.rodrigues, aruco.translation, 0.05)  # Draw Axis
            cv2.putText(image, "Id:" + str(aruco.id), aruco.center, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.aruco.drawDetectedMarkers(image, corners)
        cv2.namedWindow('cam' + str(i), cv2.WINDOW_NORMAL)
        cv2.imshow('cam' + str(i), image)


    cv2.waitKey(0)




    # def matrixToRodrigues(T):
    #     rods, _ = cv2.Rodrigues(T[0:3, 0:3])
    #     rods = rods.transpose()
    #     return rods[0]
    #
    #
    # def rodriguesToMatrix(r):
    #     rod = np.array(r, dtype=np.float)
    #     matrix = cv2.Rodrigues(rod)
    #     return matrix[0]


    # Add new rodrigues rotation to each camera
    # for camera in dataset.cameras:
    #     camera.rgb.rodrigues = matrixToRodrigues(camera.rgb.matrix[0:3, 0:3])

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('dataset', dataset)
    opt.addModelData('another_thing', [])


    def setter_translation(dataset, value, i, axis='x'):
        dataset.cameras[i].rgb.matrix['xyz'.index(axis), 3] = value


    def getter_translation(dataset, i, axis='x'):
        return dataset.cameras[i].rgb.matrix['xyz'.index(axis), 3]


    # def setter_rotation(dataset, value, i, axis='x'):
    #     rodriguesToMatrix(value)
    #     datasetrotation.rgb.matrix['xyz'.index(axis), 3] = value
    #
    #
    # def getter_rotation(dataset, i, axis='x'):
    #     r = matrixToRodrigues(dataset.cameras[i].rgb.matrix[0:3, 0:3])
    #     return r['xyz'.index(axis)]


    for i, camera in enumerate(dataset.cameras):

        # Translation
        for axis in 'xyz':
            opt.pushScalarParam(name='cam' + camera.name + '_t' + axis, model_key='dataset',
                                getter=partial(getter_translation, i=i, axis=axis),
                                setter=partial(setter_translation, i=i, axis=axis))
        #
        # # Rotation
        # for axis in 'xyz':
        #     opt.pushScalarParam(name='cam' + camera.name + '_t' + axis, model_key='dataset',
        #             getter=partial(getter_translation, i=i, axis=axis),
        #             setter=partial(setter_translation, i=i, axis=axis))
        pass

    # for key in opt.params:
    #     print('key = ' + key)
    #     print(opt.params[key])
    #     print("\n")
    #
    # print(opt.groups)

    exit(0)


    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        # Get the dataset from the model dictionary
        dataset = model['dataset']

        # Apply changes to all camera images using parameter vector
        for camera in dataset.cameras:
            camera.rgb.image_changed = addSafe(camera.rgb.image, camera.rgb.bias)
            camera.rgb.avg_changed = np.average(camera.rgb.image_changed)

        # Compute all the pair wise combinations of the set of cameras
        # Each element in the vector of errors is the difference of the average color for the combination
        error = []
        for cam_a, cam_b in combinations(dataset.cameras, 2):
            error.append(cam_a.rgb.avg_changed - cam_b.rgb.avg_changed)

        return error


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------

    for cam_a, cam_b in combinations(dataset.cameras, 2):
        opt.pushResidual(name='c' + cam_a.name + '-c' + cam_b.name, params=['bias_' + cam_a.name, 'bias_' + cam_b.name])

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()

    exit(0)


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
