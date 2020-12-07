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
import math

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.KeyPressManager as KeyPressManager
import cv2
import copy

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities


class Ball:
    def __init__(self):
        self.radius=20.0
        self.x=100.0
        self.y=200.0
        
        self.xs=[]
        self.ys=[]
        
        self.nxs=[]
        self.nys=[]

if __name__ == "__main__":


    image = cv2.imread('clean_ball.jpg')
    edges = cv2.Canny(image,100,200)
    cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    # It will be the number of polynomial degree
    # for now, i will work with 4
        
    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    args = vars(ap.parse_args())
    print(args)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    ball = Ball()
    angles = (np.linspace(0, np.pi *2,100)).tolist()


    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('ball', ball)
    opt.addDataModel('edges',edges)


    # Create specialized getter and setter functions
    def setter(ball, value, field):
        if field=='radius':
            ball.radius = value
        elif field=='x':
            ball.x = value
        elif field=='y':
            ball.y = value



    def getter(ball, field):
        if field=='radius':
            return [ball.radius]
        elif field=='x':
            return [ball.x]
        elif field=='y':
            return [ball.y]



    for field in ['radius','x','y']:
        opt.pushParamScalar(group_name='p' + field, data_key='ball', getter=partial(getter, field=field),
                            setter=partial(setter, field=field))

    def find_nearest_white(img, target):
        nonzero = np.argwhere(img == 255)
        distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
        nearest_index = np.argmin(distances)
        return nonzero[nearest_index]    
    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(models):

        ball = models['ball']
        edges=models['edges']
        xs=[]
        ys=[]
        
        nxs=[]
        nys=[]
        
        error = []
        for angle in angles:
            x=ball.x[0]+ball.radius[0]*math.cos(angle)
            y=ball.y[0]+ball.radius[0]*math.sin(angle)
            
            xs.append(x)
            ys.append(y)
            
            ny,nx=find_nearest_white(edges,[y,x])
            
            nxs.append(nx)
            nys.append(ny)
            distance=math.sqrt(pow(x-nx,2)+pow(y-ny,2))
            error.append(distance)
            
        #visualization    
        ball.xs=xs
        ball.ys=ys
        ball.nxs=nxs
        ball.nys=nys

        return error


    opt.setObjectiveFunction(objectiveFunction)
    opt.callObjectiveFunction()
    
    
    opt.printParameters()
    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for idx,angle in enumerate(angles):
        opt.pushResidual(name='r' + str(idx), params=['pradius','px','py'])

    print('residuals = ' + str(opt.residuals))
    
    

    opt.computeSparseMatrix()
    opt.printSparseMatrix()
    # exit(0)
    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------
    fig = plt.figure()

    cv2.namedWindow('ball_detection')
    cv2.imshow('ball_detection',image)
    
    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0.01, verbose=False):
        exit(0)


    def visualizationFunction(models):
        ball = models['ball']
        edges=models['edges']        
        
        gui_image=copy.deepcopy(image)
        for x,y,nx,ny in zip(ball.xs,ball.ys,ball.nxs,ball.nys):
            x,y,nx,ny=int(x),int(y),int(nx),int(ny)
            cv2.line(gui_image,(x,y),(nx,ny),(0,0,255),1)
        
        for x,y in zip(ball.xs,ball.ys):
            x,y=int(x),int(y)
            cv2.line(gui_image,(x,y),(x,y),(255,0,0),2)
            
        h,w,nc=gui_image.shape
        for pix_x in range(0,w):
            for pix_y in range(0,h):
                if edges[pix_y,pix_x]==255:
                    gui_image[pix_y,pix_x]=(0,255,0)
        cv2.imshow('ball_detection',gui_image)
        cv2.waitKey(20)

        wm = KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False)=='x':
            print("exiting")
            exit(0)

    opt.setVisualizationFunction(visualizationFunction, True)

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # opt.fromXToData()
    # opt.callObjectiveFunction()
    # wm = KeyPressManager.KeyPressManager.WindowManager()
    # if wm.waitForKey():
    #     exit(0)
    #
    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-4})

    wm = KeyPressManager.WindowManager(fig)
    if wm.waitForKey(0.01, verbose=False) == 'x':
        print("exiting")
        exit(0)
