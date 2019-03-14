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
import subprocess
from copy import deepcopy
import numpy as np
import cv2
from functools import partial
import matplotlib.pyplot as plt
import plyfile as plyfile
from scipy.spatial.distance import euclidean
import KeyPressManager.KeyPressManager
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OCDatasetLoader.OCArucoDetector as OCArucoDetector
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.utilities as utilities


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------
##
# @brief Executes the command in the shell in a blocking or non-blocking manner
#
# @param cmd a string with teh command to execute
#
# @return
def bash(cmd, blocking=True):
    print("Executing command: " + cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if blocking:
        for line in p.stdout.readlines():
            print line,
        p.wait()


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
    ap.add_argument("-o", "--path_to_output_dataset", help="path to the folder that will contain the output OC dataset",
                    type=str, default=None, required=False)
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
    # --- Extract the rgb_T_depth transform
    # ---------------------------------------
    # Confirm the structure of the txt files
    for camera in dataset_cameras.cameras:
        world_T_camera = np.linalg.inv(camera.rgb.matrix)
        depth_T_world = camera.depth.matrix

        dataset_cameras.depth_T_camera = np.dot(world_T_camera, depth_T_world)

    # exit(0)
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
            print("Cam " + str(camera.name))
            for aruco_id, aruco_detection in camera.rgb.aruco_detections.items():
                # print("Aruco " + str(aruco_id))
                # print("Pixel center coords (ground truth) = " + str(aruco_detection.center))  # ground truth

                # Find current position of aruco
                world_T_camera = np.linalg.inv(camera.rgb.matrix)
                # print('world_to_camera = ' + str(world_T_camera))

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

    # STEP 1
    # Full copy of the dataset

    print('\n---------------------------------------------------------------------------------------------------------')
    print('Creating optimized dataset...\n')
    # Get the dest folder name
    if args['path_to_output_dataset'] is None:
        folders = args['path_to_images'].split('/')
        while '' in folders:
            folders.remove('')

        dataset_name = folders[-1]

        args['path_to_output_dataset'] = args['path_to_images'] + '../' + dataset_name + '_optimized'
        print('path_to_output_dataset= ' + args['path_to_output_dataset'])

    #Delete old
    bash('rm -rf ' + args['path_to_output_dataset'], blocking=True)
    # Copy
    bash('cp -rf ' + args['path_to_images'] + ' ' + args['path_to_output_dataset'], blocking=True)

    # STEP 2
    # Overwrite txt files with new transform

    print('\n---------------------------------------------------------------------------------------------------------')
    print('World to camera transformations')
    for camera in opt.data_models['data_cameras'].cameras:
        print("\nCamera " + str(camera.name) + ':')

        world_T_camera = np.transpose(np.linalg.inv(camera.rgb.matrix))
        print("world_T_camera = " + str(world_T_camera))

        txt_filename = args['path_to_output_dataset'] + '/' + camera.name.zfill(8) + '.txt'
        fh = open(txt_filename, 'w')
        print('Writing to ' + txt_filename)

        # Write to file
        fh.write('3\n')

        for i in range(4):
            fh.write(str(world_T_camera[i][0]) + ' ' + str(world_T_camera[i][1]) + ' ' + str(
                world_T_camera[i][2]) + ' ' + str(world_T_camera[i][3]) + '\n')

        for i in range(4):
            fh.write('0 0 0 0' + '\n')

        for i in range(4):
            fh.write('0 0 0 0' + '\n')

        fh.close()

    # STEP 3
    # point clouds World to depth camera ref frame using old transform
    # then: point cloud from depth frame to world using new (optimized transform)

    print('\n---------------------------------------------------------------------------------------------------------')
    print('Camera to depth transformations')

    for i, camera in enumerate(dataset_cameras.cameras):

        print('\nCamera ' + camera.name + ':')

        ###################################################################
        # Show print .ply file with color

        # Ply file corresponding to current camera
        ply_filename = args['path_to_output_dataset'] + '/' + camera.name.zfill(8) + '.ply'
        print('\nReading pointcloud from ' + ply_filename + '...')

        # Read vertices from point cloud
        imgData = plyfile.PlyData.read(ply_filename)["vertex"]
        numVertex = len(imgData['x'])

        # create array of 3d points                           add 1 to make homogeneous
        xyz = np.c_[imgData['x'], imgData['y'], imgData['z'], np.ones(shape=(imgData['z'].size, 1))]

        # TODO: Use plan for STEP 3
        print("#################################################")
        print('Computing point cloud tranformations')

        # TODO: Define arguments
        # For some awkward reason the local point clouds (ply files) are stored in openGL coordinates.
        opengl2opencv = np.zeros((4, 4))
        opengl2opencv[0, :] = [1, 0, 0, 0]
        opengl2opencv[1, :] = [0, 0, 1, 0]
        opengl2opencv[2, :] = [0, -1, 0, 0]
        opengl2opencv[3, :] = [0, 0, 0, 1]

        opencv2opengl = np.linalg.inv(opengl2opencv)

        old_world_T_depth = np.linalg.inv(camera.depth.matrix)

        depth_T_camera = dataset_cameras.depth_T_camera

        # This is optimized
        camera_T_world = camera.rgb.matrix

        print('old_world_T_depth =' + str(old_world_T_depth))
        print("depth_T_camera = " + str(depth_T_camera))
        print("camera_T_world = " + str(camera_T_world))

        # Data structure for new points
        pointsInNewWorld_opengl = np.zeros(shape=(len(xyz), 4))

        for j in range(numVertex):
            # TODO: Convert from OpenGL to OpenCV:
            # Apply opengl2opencv conversion
            pointInOpenCV = np.dot(opengl2opencv, xyz[j])

            # TODO: Go from world to depth through old transformation:
            # Apply old_world_T_depth transformation
            pointInDepth = np.dot(old_world_T_depth, pointInOpenCV)
            # print('pointInDepth= ' + str(pointInDepth))

            # TODO: Go from depth to world through new optimized transformation:
            # Apply depth_T_camera then apply optimized camera_T_world transformation

            pointInNewWorld_opencv = np.dot(camera_T_world, np.dot(depth_T_camera, pointInDepth))

            pointsInNewWorld_opengl[j] = np.dot(opencv2opengl, pointInNewWorld_opencv)

            # pointsInNewWorld_opengl[j] = pointInDepth

        print("New point cloud= ")
        print(pointsInNewWorld_opengl)


        # Write to the .ply file
        file_object = open(ply_filename, "w")

        # Write file header information
        file_object.write('ply' + '\n')
        file_object.write('format ascii 1.0' + '\n')
        file_object.write('comment ---' + '\n')
        file_object.write('element vertex ' + str(numVertex) + '\n')
        file_object.write('property float x' + '\n')
        file_object.write('property float y' + '\n')
        file_object.write('property float z' + '\n')
        file_object.write('element face 0' + '\n')
        file_object.write('property list uchar uint vertex_indices' + '\n')
        file_object.write('end_header' + '\n')

        # Write the new points
        for j in range(numVertex):
            file_object.write(str(pointsInNewWorld_opengl[j][0]) + ' ' + str(pointsInNewWorld_opengl[j][1]) + ' '
                              + str(pointsInNewWorld_opengl[j][2]) + '\n')

        file_object.close()


    # STEP 4
    # Delete depthimage_speedup folder to enable recalculation
