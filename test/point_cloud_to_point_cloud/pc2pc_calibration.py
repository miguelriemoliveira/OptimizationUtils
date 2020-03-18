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
import copy as cp
from functools import partial
import matplotlib.pyplot as plt
import open3d as o3d
import KeyPressManager.KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.transformations as tf

#  Constants
RANDOM_ERROR = 0.4

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities


class SensorTransforms:

    def __init__(self):
        self.t = np.array([0.0, 0.0, 0.0])
        # self.r = np.array([0.0, 0.0, 0.0])

######################################
# function to subsample a pointcloud #
######################################
def subSamplePointCloud(n,in_pc):
    
    points1 = np.array(in_pc.points)
    # print(in_pc.shape[0])
    # print(in_pc.shape[1])
    
    little_points = np.empty(shape = ( (points1.shape[0]/n,points1.shape[1]) ))
    
    ind = 0
    for i in np.arange(0,len(little_points)):
        little_points[i]= points1[ind]
        ind += n


    # print(little_points.shape)
    
    little_cloud = o3d.PointCloud()
    little_cloud.points = o3d.Vector3dVector(little_points)

    # print("writing little cow")
    # o3d.io.write_point_cloud("./test/point_cloud_to_point_cloud/cow.ply",little_cloud) # Write the point cloud

    return little_cloud

#######################################
# Add Gaussian Noise to a point cloud #
#######################################
def addNoise(max_error, cloud):
    
    cloudPoints = np.array(cloud.points)
    noisy_points = np.empty(shape = (np.array(cloudPoints.shape) ))
    
    for i in np.arange(0,len(noisy_points)):
        rand_error = np.random.random_sample( (1, 3) ) * max_error
        noisy_points[i] = cloudPoints[i] + rand_error

    #print(noisy_points.shape)

    return noisy_points


if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    args = vars(ap.parse_args())
    print(args)


    # ---------------------------------------
    # --- CREATE 2 POINT CLOUDS FOR TEST
    # ---------------------------------------
    # read and Show original point cloud
    # point cloud 1
    print("Read the point cloud")
    pcd1 = o3d.io.read_point_cloud("./test/point_cloud_to_point_cloud/cow.ply") # Read the point cloud
    pcd1.paint_uniform_color([1, 0, 0])

    #smaller version of cow
    pcd1 = subSamplePointCloud(5,pcd1)
    
    # pcd2 = o3d.io.read_point_cloud("./test/point_cloud_to_point_cloud/cow.ply") # Read the point cloud
    # pcd2 = subSamplePointCloud(5,pcd2)
    pcd2 = cp.deepcopy(pcd1)
    #downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.2],
                             [-0.139, 0.967, -0.215, 0.1],
                             [0.487, 0.255, 0.835, 0.2], 
                             [0.0, 0.0, 0.0, 1.0]])

    # random noise in pc2    
    pcd2.transform(trans_init)
    noisy_points = addNoise(RANDOM_ERROR,pcd2)

    noisy_cloud = o3d.PointCloud()
    noisy_cloud.points = o3d.Vector3dVector(noisy_points)

    
   
    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    sensorTransforms = SensorTransforms()


    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('sensorTransforms', sensorTransforms)


    # Create specialized getter and setter functions
    def setter(sensorTransforms, value, i):
        if i == 0:
            sensorTransforms.t[0] = np.array(value)
        elif i == 1:
            sensorTransforms.t[1] = np.array(value)
        elif i == 2:
            sensorTransforms.t[2] = np.array(value)
        # elif i == 3:
        #     sensorTransforms.r[0] = np.array(value)
        # elif i == 4:
        #     sensorTransforms.r[1] = np.array(value)
        # elif i == 5:
        #     sensorTransforms.r[2] = np.array(value)


    def getter(sensorTransforms, i):
        if i == 0:
            return [sensorTransforms.t[0]]
        elif i == 1:
            return [sensorTransforms.t[1]]
        elif i == 2:
            return [sensorTransforms.t[2]]
        # elif i == 3:
        #     return [sensorTransforms.r[0]]
        # elif i == 4:
        #     return [sensorTransforms.r[1]]
        # elif i == 5:
        #     return [sensorTransforms.r[2]]


    def getterTranslation(sensorTransforms):
        return sensorTransforms.t.tolist()

    def setterTranslation(sensorTransforms, values):
       sensorTransforms.t[0] = values[0]
       sensorTransforms.t[1] = values[1]
       sensorTransforms.t[2] = values[2]

    # for idx in range(0, 6):
    #     opt.pushParamScalar(group_name='p' + str(idx), data_key='sensorTransforms', getter=partial(getter, i=idx),
    #                         setter=partial(setter, i=idx))

    # for idx in range(0, 3):
    #     opt.pushParamScalar(group_name='p' + str(idx), data_key='sensorTransforms', getter=partial(getter, i=idx),
    #                         setter=partial(setter, i=idx))

    opt.pushParamVector(group_name='t', data_key='sensorTransforms',
                        getter=partial(getterTranslation),
                        setter=partial(setterTranslation),
                        suffix=['x', 'y', 'z'])


    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        sensorTransforms = model['sensorTransforms']

        error = []
        err = 0

        print('sensorTransforms = ' + str(sensorTransforms.t))
       
        # transformMatrix remember to initialize at identity
        # Check if data type is needed
        # ang_loop = np.array(sensorTransforms.r, dtype=np.float64)
        # tr_loop = np.array(sensorTransforms.t, dtype=np.float64)
        tr_loop = np.array([sensorTransforms.t[0],sensorTransforms.t[1],sensorTransforms.t[2]])

        # otf = tf.compose_matrix(scale=None, shear=None, angles=ang_loop, translate=tr_loop, perspective=None)
        otf = tf.compose_matrix(scale=None, shear=None, angles=None, translate=tr_loop, perspective=None)
        
        y = cp.deepcopy(noisy_cloud)
        y = y.transform(otf)

        sourcepts = np.array(y.points)
        targetpts = np.array(pcd1.points)

        #compute error between points in transformed target and source (append or add!?)
        for i in np.arange(0,len(sourcepts)):
            error.append( (sourcepts[i][0] - targetpts[i][0]) * (sourcepts[i][0] - targetpts[i][0])
                        + (sourcepts[i][1] - targetpts[i][1]) * (sourcepts[i][1] - targetpts[i][1])
                        + (sourcepts[i][2] - targetpts[i][2]) * (sourcepts[i][2] - targetpts[i][2]))

        # for i in np.arange(0,len(sourcepts)):
        #     err = err + ( (sourcepts[i][0] - targetpts[i][0]) * (sourcepts[i][0] - targetpts[i][0])
        #                     + (sourcepts[i][1] - targetpts[i][1]) * (sourcepts[i][1] - targetpts[i][1])
        #                     + (sourcepts[i][2] - targetpts[i][2]) * (sourcepts[i][2] - targetpts[i][2]))


        # Para usar todas as combinações 2 a 2 de uma lista
        # for cam_a, cam_b in combinations(dataset.cameras, 2):
        #     error.append(abs(cam_a.rgb.avg_changed - cam_b.rgb.avg_changed))
        #


        # error.append(err)
        # print("first 5 errors = " + str(error[0:5]))
        return error


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for a in range(0, len(noisy_points)):
        # opt.pushResidual(name='pts' + str(a), params=['t', 'r'])
        opt.pushResidual(name='r' + str(a), params=['tx', 'ty', 'tz'])
    # for a in range(0, 1):
    #     opt.pushResidual(name='cloud' + str(a), params=['t', 'r'])

    # print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()

    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------

    # fig = plt.figure()
    # # ax = fig.add_subplot(111)

    #Visualize initial and final cloud
    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    y = cp.deepcopy(noisy_cloud)
    y.paint_uniform_color([0, 1, 0])
    vis.add_geometry(y)
    vis.update_renderer()
    print("Lock in open3D windows (vis)")
    vis.run()


    # wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    # if wm.waitForKey(0., verbose=False):
    #     exit(0)

    def visualizationFunction(model):

        sensorTransforms = model['sensorTransforms']

        # y = polynomial.param0[0] + \
        #     np.multiply(polynomial.param1[0], np.power(x, 1)) + \
        #     np.multiply(polynomial.param2[0], np.power(x, 2)) + \
        #     np.multiply(polynomial.params_3_and_4[0][0], np.power(x, 3)) + \
        #     np.multiply(polynomial.params_3_and_4[1][0], np.power(x, 4))

        # handle_plot[0].set_ydata(y)

        # ang = np.array([sensorTransforms.r[0],sensorTransforms.r[1],sensorTransforms.r[2]])
        tr = np.array([sensorTransforms.t[0],sensorTransforms.t[1],sensorTransforms.t[2]])
        # print("Translation:")
        # print(tr)

        Mv = tf.compose_matrix(scale=None, shear=None, angles=None, translate=tr, perspective=None)
        # Mv = tf.compose_matrix(scale=None, shear=None, angles=ang, translate=tr, perspective=None)
    
        # print("Transformation Matrix: ") 
        # print(Mv)
        y = cp.deepcopy(noisy_cloud)
        y = y.transform(Mv)
        
        y.paint_uniform_color([0, 0, 1])
        # vis.add_geometry(y)
        vis.update_renderer()
        vis.run()

        # wm = KeyPressManager.KeyPressManager.WindowManager(fig)
        # if wm.waitForKey(0.01, verbose=False):
        #     exit(0)

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

    opt.printParameters()
    # opt.printResiduals()
    # opt.printSparseMatrix()
    # exit(0)

    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-2, 'xtol': 1e-2, 'gtol': 1e-2, 'diff_step': 1e-2})

    vis.destroy_window()
    vis.create_window()
    vis.add_geometry(pcd1)
    pcd1.paint_uniform_color([1, 0, 0])


    tr = np.array([sensorTransforms.t[0],sensorTransforms.t[1],sensorTransforms.t[2]])
  
    Mv = tf.compose_matrix(scale=None, shear=None, angles=None, translate=tr, perspective=None)
    
    y = cp.deepcopy(noisy_cloud)
    y = y.transform(Mv)
        
    y.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd1, y])
    # vis.update_renderer()
    # vis.run()

    # wm = KeyPressManager.KeyPressManager.WindowManager()
    # if wm.waitForKey():
    #     exit(0)
