#!/usr/bin/env python
"""
This is a long, multiline description
"""

#-------------------------------------------------------------------------------
#--- IMPORTS (standard, then third party, then my own modules)
#-------------------------------------------------------------------------------
import argparse  #to read command line arguments
from itertools import combinations
import numpy as np
from copy import deepcopy
import cv2

import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OptimizationUtils.OptimizationUtils as OptimizationUtils

#-------------------------------------------------------------------------------
#--- FUNCTIONS
#-------------------------------------------------------------------------------

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


#-------------------------------------------------------------------------------
#--- MAIN
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    #---------------------------------------
    #--- Parse command line argument
    #---------------------------------------
    ap = argparse.ArgumentParser()

    #Dataset loader arguments
    ap.add_argument("-p", "--path_to_images", help="path to the folder that contains the OC dataset", required=True)
    ap.add_argument("-ext", "--image_extension", help="extension of the image files, e.g., jpg or png", default='jpg')
    ap.add_argument("-m", "--mesh_filename", help="full filename to input obj file, i.e. the 3D model", required=True)
    ap.add_argument("-i", "--path_to_intrinsics", help="path to intrinsics yaml file", required=True)
    ap.add_argument("-ucci", "--use_color_corrected_images", help="Use previously color corrected images", action='store_true', default=False)
    ap.add_argument("-si", "--skip_images", help="skip images. Useful for fast testing", type=int, default=1)
    ap.add_argument("-vri", "--view_range_image", help="visualize sparse and dense range images", action='store_true', default=False)
    
    #OptimizationUtils arguments
    ap.add_argument("-sv", "--skip_vertices", help="skip vertices. Useful for fast testing", type=int, default=1)
    ap.add_argument("-z", "--z_inconsistency_threshold", help="threshold for max z inconsistency value", type=float, default=0.05)
    ap.add_argument("-vpv", "--view_projected_vertices", help="visualize projections of vertices onto images", action='store_true', default=False)
    ap.add_argument("-vo", "--view_optimization", help="...", action='store_true', default=False)


    args = vars(ap.parse_args())
    print(args)

    #---------------------------------------
    #--- INITIALIZATION
    #---------------------------------------
    dataset_loader = OCDatasetLoader.Loader(args)
    dataset = dataset_loader.loadDataset()
    num_cameras = len(dataset.cameras)

    # ---------------------------------------
    # --- PREPARE OPTIMIZATION
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addStaticData('dataset', dataset)

    #A vector containing the bias for the cameras
    bias = [0] * len(dataset.cameras)
    bias[0] = 150      #brighten the first camera
    bias[1] = -140     #darken the second camera

    for i, b in enumerate(bias):
        opt.pushScalarParam('bias' + str(i), b)

    print(opt.params)

    exit(0)

    # ---------------------------------------
    # --- SET THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def addSafe(i_in, val):
        """Avoids saturation when adding to uint8 images"""
        i_out = i_in.astype(np.float) #Convert the i to type float 
        i_out = np.add(i_out, val) #Perform the adding of parameters to the i
        i_out = np.maximum(i_out, 0)   #underflow
        i_out = np.minimum(i_out, 255) #overflow
        return i_out.astype(np.uint8) #Convert back to uint8 and return

    def objectiveFunction(x, static_data, show=False):

        #Get the static data from the dictionary
        dataset = static_data['dataset']
        cameras = dataset.cameras
        num_cameras = len(cameras)

        #Apply changes to all camera images using parameter vector
        changed_images = []
        for i in range(0,num_cameras):
            changed_images.append(addSafe(cameras[i].rgb.image, x[i]))

        #Compute averages for initial and changed images
        initial_avgs = np.zeros((num_cameras))
        changed_avgs = np.zeros((num_cameras))
        for i in range(0,num_cameras):
            initial_avgs[i] = np.average(cameras[i].rgb.image)
            changed_avgs[i] = np.average(changed_images[i])

        #Compute error as the sum of differences between initial and changed averages. Error is a scalar in this example, hence the sum
        error = np.sum(initial_avgs - changed_avgs)

        #Visualization
        for i in range(0,num_cameras):
            cv2.namedWindow('Initial Cam ' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('Initial Cam ' + str(i), cameras[i].rgb.image)
            cv2.namedWindow('Changed Cam ' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('Changed Cam ' + str(i), changed_images[i])
        cv2.waitKey(0)

        return error
    # -----------OBJECTIVE FUNCTION FINISHED ------------------
    
    opt.setObjectiveFunction(objectiveFunction)

    #---------------------------------------
    #--- Create X0 (First Guess)
    #---------------------------------------
       
    #objectiveFunction(x0, dataset)

    opt.callObjectiveFunction(x0)


    #---------------------------------------
    #--- USING THE API
    #---------------------------------------



