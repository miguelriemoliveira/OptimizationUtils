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
#--- HEADER
#-------------------------------------------------------------------------------
__author__ = "Miguel Riem de Oliveira"
__date__ = "2018"
__copyright__ = "Miguel Riem de Oliveira"
__credits__ = ["Miguel Riem de Oliveira"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Miguel Oliveira"
__email__ = "m.riem.oliveira@gmail.com"
__status__ = "Development"

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

    # ---------------------------------------
    # --- Compute the list of points that will be used by the cost function
    # ---------------------------------------

    
    # ---------------------------------------
    # --- PREPARE OPTIMIZATION
    # ---------------------------------------

    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()

    opt.addStaticData('dataset', dataset)

    # ---------------------------------------
    # --- SET THE OBJECTIVE FUNCTION
    # ---------------------------------------

    #Define the objective function
    def objectiveFunction(x, static_data, show=False):

        #cameras = static_data['cameras']
        cameras = static_data.cameras

        #Get value of first pixel of first camera
        pix0 = np.average(cameras[0].rgb.image[:][:][:])
        pix1 = np.average(cameras[1].rgb.image[:][:][:])
        pix2 = np.average(cameras[2].rgb.image[:][:][:])

        print("pix0 = " + str(pix0))
        print("pix1 = " + str(pix1))
        print("pix2 = " + str(pix2))

        image0 = np.add(cameras[0].rgb.image, x[0])
        
        print((cameras[0].rgb.image.dtype ))

        print(np.average(image0))

        cv2.namedWindow('cam0_orig', cv2.WINDOW_NORMAL)
        cv2.imshow('cam0_orig', cameras[0].rgb.image)
        cv2.namedWindow('cam0', cv2.WINDOW_NORMAL)
        cv2.imshow('cam0', image0)

        cv2.waitKey(0)

        #print('visualization in ' + str(time.time() - t) + ' secs')
        e = [pix0-pix1, pix0-pix1, pix1-pix2]
        return e

    #opt.setObjectiveFunction(objectiveFunction)


    objectiveFunction([-200, 0, 0], dataset)

#---------------------------------------
#--- USING THE USER INTERFACE
#---------------------------------------


    #---------------------------------------
    #--- USING THE API
    #---------------------------------------



