#!/usr/bin/env python
"""
This code shows how the optimizer class may be used for changing the colors of a set of images so that the average
color in all images is very similar. This is often called color correction.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import argparse  # to read command line arguments
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


def addSafe(i_in, val):
    """Avoids saturation when adding to uint8 images"""
    i_out = i_in.astype(np.float)  # Convert the i to type float
    i_out = np.add(i_out, val)  # Perform the adding of parameters to the i
    i_out = np.maximum(i_out, 0)  # underflow
    i_out = np.minimum(i_out, 255)  # overflow
    return i_out.astype(np.uint8)  # Convert back to uint8 and return


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

    # Change camera's colors just to better see optimization working
    for i, camera in enumerate(dataset.cameras):
        dataset.cameras[i].rgb.image = addSafe(dataset.cameras[i].rgb.image, random.randint(-70, 70))

    # lets add a bias variable to each camera.rgb. This value will be used to change the image and optimize
    for i, camera in enumerate(dataset.cameras):
        # camera.rgb.bias = random.randint(-30, 30)
        camera.rgb.bias = 0

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addModelData('dataset', dataset)
    opt.addModelData('another_thing', [])

    def setter(dataset, value, i):
        dataset.cameras[i].rgb.bias = value

    def getter(dataset, i):
        return dataset.cameras[i].rgb.bias


    # Create specialized getter and setter functions
    for idx_camera, camera in enumerate(dataset.cameras):
        # if idx_camera == 0:# First camera with static color
        #     bound_max = camera.rgb.bias + 0.00001
        #     bound_min = camera.rgb.bias - 0.00001
        # else:
        bound_max = camera.rgb.bias + 150
        bound_min = camera.rgb.bias - 150

        opt.pushScalarParam(name='bias_' + camera.name, data_key='dataset', getter=partial(getter, i=idx_camera),
                            setter=partial(setter, i=idx_camera), bound_max=bound_max, bound_min=bound_min)

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
    # The error is computed from the difference of the average color of one image with another
    # Thus, we will use all pairwise combinations of available images
    # For example, if we have 3 cameras c0, c1 and c2, the residuals should be:
    #    c0-c1, c0-c2, c1-c2

    for cam_a, cam_b in combinations(dataset.cameras, 2):
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
