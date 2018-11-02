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
    cam_pairs = []
    for cam_a, cam_b in combinations(dataset.cameras, 2):
        print(cam_a.name + ' with ' + cam_b.name)

        # get a list of 3D points by concatenating the 3D measurements of cam_a and cam_b
        pts3D_in_map_a = np.dot(cam_a.depth.matrix, cam_a.depth.vertices[:, 0::args['skip_vertices']])
        pts3D_in_map_b = np.dot(cam_b.depth.matrix, cam_b.depth.vertices[:, 0::args['skip_vertices']])
        pts3D_in_map = np.concatenate([pts3D_in_map_a, pts3D_in_map_b], axis=1)

        # project 3D points to cam_a and to cam_b
        pts2D_a, pts_valid_a, pts_range_a = cam_a.rgb.projectToPixel3(pts3D_in_map)
        pts2D_a = np.where(pts_valid_a, pts2D_a, 0)
        range_meas_a = cam_a.rgb.range_dense[pts2D_a[1,:], pts2D_a[0,:]]
        z_valid_a = abs(pts_range_a - range_meas_a) < args['z_inconsistency_threshold']

        pts2D_b, pts_valid_b, pts_range_b = cam_b.rgb.projectToPixel3(pts3D_in_map)
        pts2D_b = np.where(pts_valid_b, pts2D_b, 0)
        range_meas_b = cam_b.rgb.range_dense[pts2D_b[1,:], pts2D_b[0,:]]
        z_valid_b = abs(pts_range_b - range_meas_b) < args['z_inconsistency_threshold']

        #Compute masks for the valid projections
        mask = np.logical_and(pts_valid_a, pts_valid_b)
        z_mask = np.logical_and(z_valid_a, z_valid_b)
        final_mask = np.logical_and(mask, z_mask)

        #Create a dictionary to describe this camera pair (to be used by the objective function)
        cam_pair = {}
        cam_pair['cam_a'] = cam_a.name
        cam_pair['cam_b'] = cam_b.name
        cam_pair['idx_a'] = [x.name for x in dataset.cameras].index(cam_a.name)
        cam_pair['idx_b'] = [x.name for x in dataset.cameras].index(cam_b.name)
        cam_pair['pts2D_a'] = pts2D_a[:, final_mask]
        cam_pair['pts2D_b'] = pts2D_b[:, final_mask]
        cam_pairs.append(cam_pair)

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

            if args['view_optimization']:
                keyPressManager()

    print(cam_pairs)

    #Compute float and YCrCB images for each camera
    for camera in dataset.cameras:
        camera.rgb.fimage = np.float32(camera.rgb.image) / 255.0
        camera.rgb.ycrcb_image = cv2.cvtColor(camera.rgb.fimage, cv2.COLOR_BGR2YCrCb)
        camera.rgb.lab_image = cv2.cvtColor(camera.rgb.fimage, cv2.COLOR_BGR2LAB)
        #print(camera.rgb.lab_image.dtype)

        # Print the minimum and maximum of lightness.
        #l_channel,a_channel,b_channel = cv2.split(camera.rgb.lab_image)
        #print np.min(l_channel) # 0
        #print np.max(l_channel) # 255

        ## Print the minimum and maximum of a.
        #print np.min(a_channel) # 42
        #print np.max(a_channel) # 226

        ## Print the minimum and maximum of b.
        #print np.min(b_channel) # 20
        #print np.max(b_channel) # 223

        #exit(0)

    # ---------------------------------------
    # --- PREPARE OPTIMIZATION
    # ---------------------------------------

    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()

    opt.addStaticData('dataset', dataset)
    opt.addStaticData('cam_pairs', cam_pairs)

    # ---------------------------------------
    # --- SET THE OBJECTIVE FUNCTION
    # ---------------------------------------

    #Define the objective function
    def objectiveFunction(x, static_data, show=False):

        print(self.test_var)
        cam_pairs = static_data['cam_pairs']
        cameras = static_data['cameras']

        #t = time.time()
        #apply gamma correction for all camera images given the gamma value (x)
        for i, camera in enumerate(cameras):
            idx = i * 256

            table = np.array(x[idx:idx+256])
            #print(table)

            camera.rgb.gc_image = cv2.LUT(camera.rgb.image, table).astype(np.uint8)
        
        #Compute the error with the new corrected images
        e = []
        for i, cam_pair in enumerate(cam_pairs):
            pts2D_a = cam_pair['pts2D_a']
            pts2D_b = cam_pair['pts2D_b']

            cam_a = cameras[cam_pair['idx_a']]
            cam_b = cameras[cam_pair['idx_b']]

            pixs_a = cam_a.rgb.gc_image[pts2D_a[1,:], pts2D_a[0,:]]
            pixs_b = cam_b.rgb.gc_image[pts2D_b[1,:], pts2D_b[0,:]]

            diff = pixs_a - pixs_b
            dist = np.linalg.norm(diff)
            e.append(np.linalg.norm(dist))

        
        #t = time.time();
        if show == True:
            #print("x =\n" + str(['{:.3f}'.format(i) for i in x]))
            print("e =\n" + str(['{:.3f}'.format(i) for i in e]))
            print("avg error =\n" + str('{:.3f}'.format(np.average(e))))

            #for i, camera in enumerate(cameras):
                #cv2.namedWindow(camera.name + '_original', cv2.WINDOW_NORMAL)
                #cv2.imshow(camera.name + '_original', camera.rgb.image)
            for i, camera in enumerate(cameras):
                cv2.namedWindow(camera.name + '_corrected', cv2.WINDOW_NORMAL)
                cv2.imshow(camera.name + '_corrected', camera.rgb.gc_image)
            #cv2.waitKey(0)

        #print('visualization in ' + str(time.time() - t) + ' secs')
        return e

    opt.setObjectiveFunction(objectiveFunction)


#---------------------------------------
#--- USING THE USER INTERFACE
#---------------------------------------


    #---------------------------------------
    #--- USING THE API
    #---------------------------------------



