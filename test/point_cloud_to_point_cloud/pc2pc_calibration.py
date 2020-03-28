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
from itertools import combinations
import matplotlib.pyplot as plt
import open3d as o3d
import KeyPressManager.KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import OptimizationUtils.transformations as tf

#  Constants
RANDOM_ERROR = 0.1
MAX_ROT_ERROR = np.pi
MAX_TRANS_ERROR = 0.3
NUM_MODELS = 10
SUB_SAMPLE_FACTOR = 1

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities


class Model:
    def __init__(self, name):
        self.t = [0, 0, 0]
        self.r = [0, 0, 0]
        self.cloud = None
        self.name = str(name)

    def __str__(self):
        return 'Model ' + str(self.name)


######################################
# function to subsample a pointcloud #
######################################
def subSamplePointCloud(n, in_pc):
    points1 = np.array(in_pc.points)
    # print(in_pc.shape[0])
    # print(in_pc.shape[1])

    little_points = np.empty(shape=((points1.shape[0] / n, points1.shape[1])))

    ind = 0
    for i in np.arange(0, len(little_points)):
        little_points[i] = points1[ind]
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
    noisy_points = np.empty(shape=(np.array(cloudPoints.shape)))

    for i in np.arange(0, len(noisy_points)):
        rand_error = np.random.random_sample((1, 3)) * max_error
        noisy_points[i] = cloudPoints[i] + rand_error

    # print(noisy_points.shape)

    return noisy_points


#################################
# Add axes to the visualization #
#################################
def drawAxes(vis):
    arrow_x = o3d.geometry.create_mesh_arrow(cylinder_radius=0.1, cone_radius=0.2, cylinder_height=1.0, cone_height=0.2,
                                             resolution=20, cylinder_split=4, cone_split=1)
    arrow_x.paint_uniform_color([1, 0, 0])
    a = np.array([np.pi / 2, 0, 0])
    t = np.array([0, 0, 0])
    tm = tf.compose_matrix(angles=a, translate=t)
    arrow_x.transform(tm)
    vis.add_geometry(arrow_x)

    arrow_y = o3d.geometry.create_mesh_arrow(cylinder_radius=0.1, cone_radius=0.2, cylinder_height=1.0, cone_height=0.2,
                                             resolution=20, cylinder_split=4, cone_split=1)
    arrow_y.paint_uniform_color([0, 1, 0])
    a = np.array([0, np.pi / 2, 0])
    t = np.array([0, 0, 0])
    tm = tf.compose_matrix(angles=a, translate=t)
    arrow_y.transform(tm)
    vis.add_geometry(arrow_y)

    arrow_z = o3d.geometry.create_mesh_arrow(cylinder_radius=0.1, cone_radius=0.2, cylinder_height=1.0, cone_height=0.2,
                                             resolution=20, cylinder_split=4, cone_split=1)
    arrow_z.paint_uniform_color([0, 0, 1])
    a = np.array([0, 0, np.pi / 2])
    t = np.array([0, 0, 0])
    tm = tf.compose_matrix(angles=a, translate=t)
    arrow_z.transform(tm)
    vis.add_geometry(arrow_z)


if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    args = vars(ap.parse_args())
    print(args)

    # -----------------------------------------
    # --- CREATE SEVERAL POINT CLOUDS FOR TEST
    # -----------------------------------------
    # read and Show original point cloud
    # point cloud 1
    print("Read the point cloud")
    pcd1 = o3d.io.read_point_cloud("./test/point_cloud_to_point_cloud/cow.ply")  # Read the point cloud

    models = []
    pcd1 = subSamplePointCloud(SUB_SAMPLE_FACTOR, pcd1)

    # create other models
    for i in np.arange(0, NUM_MODELS):
        pcd = []
        pcd = cp.deepcopy(pcd1)

        # Apply random rotation and translation to second cloud
        ang = np.random.random_sample((3,)) * MAX_ROT_ERROR
        trans = np.random.random_sample((3,)) * MAX_TRANS_ERROR
        trans_init = tf.compose_matrix(angles=ang, translate=trans)

        # random noise in pc2    
        pcd.transform(trans_init)
        noisy_points = addNoise(RANDOM_ERROR, pcd)

        noisy_cloud = o3d.PointCloud()
        noisy_cloud.points = o3d.Vector3dVector(noisy_points)

        model = Model(i)
        model.cloud = noisy_cloud
        model.t = [0, 0, 0]
        model.r = [0, 0, 0]
        models.append(model)

    # print(models)
    # exit(0)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('models', models)


    # # Create specialized getter and setter functions
    # def setter(sensorTransforms, value, i):
    #     if i == 0:
    #         sensorTransforms.t[0] = np.array(value)
    #     elif i == 1:
    #         sensorTransforms.t[1] = np.array(value)
    #     elif i == 2:
    #         sensorTransforms.t[2] = np.array(value)
    #     elif i == 3:
    #         sensorTransforms.r[0] = np.array(value)
    #     elif i == 4:
    #         sensorTransforms.r[1] = np.array(value)
    #     elif i == 5:
    #         sensorTransforms.r[2] = np.array(value)
    #
    #
    # def getter(sensorTransforms, i):
    #     if i == 0:
    #         return [sensorTransforms.t[0]]
    #     elif i == 1:
    #         return [sensorTransforms.t[1]]
    #     elif i == 2:
    #         return [sensorTransforms.t[2]]
    #     elif i == 3:
    #         return [sensorTransforms.r[0]]
    #     elif i == 4:
    #         return [sensorTransforms.r[1]]
    #     elif i == 5:
    #         return [sensorTransforms.r[2]]

    def getterTranslation(models, i):
        return models[i].t


    def getterRotation(models, i):
        return models[i].r


    def setterTranslation(models, values, i):
        models[i].t[0] = values[0]
        models[i].t[1] = values[1]
        models[i].t[2] = values[2]


    def setterRotation(models, values, i):
        models[i].r[0] = values[0]
        models[i].r[1] = values[1]
        models[i].r[2] = values[2]


    # def getterSensorRotation(data, sensor_key, collection_key):
    #     calibration_parent = data['sensors'][sensor_key]['calibration_parent']
    #     calibration_child = data['sensors'][sensor_key]['calibration_child']
    #     transform_key = calibration_parent + '-' + calibration_child
    #
    #     # We use collection selected_collection and assume they are all the same
    #     quat = data['collections'][collection_key]['transforms'][transform_key]['quat']
    #     hmatrix = transformations.quaternion_matrix(quat)
    #     matrix = hmatrix[0:3, 0:3]
    #
    #     return utilities.matrixToRodrigues(matrix)
    #
    #
    # def setterSensorRotation(data, value, sensor_key):
    #     assert len(value) == 3, "value must be a list with length 3."
    #
    #     matrix = utilities.rodriguesToMatrix(value)
    #     hmatrix = np.identity(4)
    #     hmatrix[0:3, 0:3] = matrix
    #     quat = transformations.quaternion_from_matrix(hmatrix)
    #
    #     calibration_parent = data['sensors'][sensor_key]['calibration_parent']
    #     calibration_child = data['sensors'][sensor_key]['calibration_child']
    #     transform_key = calibration_parent + '-' + calibration_child
    #
    #     for _collection_key in data['collections']:
    #         data['collections'][_collection_key]['transforms'][transform_key]['quat'] = quat


    for i, model in enumerate(models):
        if i == 0:
            # to fix model_0 as reference model, no POS change
            bound_max_t = model.t + np.finfo(np.float32).eps
            bound_min_t = model.t - np.finfo(np.float32).eps
            bound_max_r = model.r + np.finfo(np.float32).eps
            bound_min_r = model.r - np.finfo(np.float32).eps
        else:
            bound_max_t = None
            bound_min_t = None
            bound_max_r = None
            bound_min_r = None

        opt.pushParamVector(group_name='model' + str(i) + '_t', data_key='models',
                            getter=partial(getterTranslation, i=i),
                            setter=partial(setterTranslation, i=i),
                            suffix=['x', 'y', 'z'],
                            bound_max=bound_max_t, bound_min=bound_min_t)

        opt.pushParamVector(group_name='model' + str(i) + '_r', data_key='models',
                            getter=partial(getterRotation, i=i),
                            setter=partial(setterRotation, i=i),
                            suffix=['x', 'y', 'z'],
                            bound_max=bound_max_r, bound_min=bound_min_r)

    opt.printParameters()


    # Push parameters to optimization
    # comb = combinations(np.arange(0, len(models)), 2)
    # for c in comb:
    #     g_name = 't' + str(c[0]) + str(c[1])
    #     opt.pushParamVector(group_name=g_name, data_key='sensorTransforms',
    #                         getter=partial(getterTranslation),
    #                         setter=partial(setterTranslation),
    #                         suffix=['x', 'y', 'z'])
    #     g_name = 'r' + str(c[0]) + str(c[1])
    #     opt.pushParamVector(group_name=g_name, data_key='sensorTransforms',
    #                         getter=partial(getterRotation),
    #                         setter=partial(setterRotation),
    #                         suffix=['x', 'y', 'z'])

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(models):

        models = models['models']

        error = []

        # comb = combinations(np.arange(0,len(models)), 2)
        # for c in comb:
        #     t_pattern = g_name = 't' + str(c[0]) + str(c[1])
        #     tra = opt.getParamsContainingPattern(t_pattern)
        count = 0 
        # equivalente?
        #for c in combinations(models, 2):
        for model_a, model_b in combinations(models, 2):
            count = count + 1
            # model_a = c[0]
            # model_b = c[1]
            print("Iteration: " + str(count) + ": " + str(model_a) + ' with ' + str(model_b))

            trans_a = np.array([model_a.t[0], model_a.t[1], model_a.t[2]])
            angle_a = np.array([model_a.r[0], model_a.r[1], model_a.r[2]])
            tfa = tf.compose_matrix(scale=None, shear=None, angles=angle_a, translate=trans_a, perspective=None)

            temp_a = cp.deepcopy(model_a.cloud)
            temp_a.transform(tfa)
            targetpts = np.array(temp_a.points)

            # v = cp.deepcopy(noisy_cloud)
            # # y = (noisy_cloud)
            # v.transform(Mv)
            # y.points = v.points

            trans_b = np.array([model_b.t[0], model_b.t[1], model_b.t[2]])
            angle_b = np.array([model_b.r[0], model_b.r[1], model_b.r[2]])
            tfb = tf.compose_matrix(scale=None, shear=None, angles=angle_b, translate=trans_b, perspective=None)
            
            temp_b = cp.deepcopy(model_b.cloud)
            temp_b.transform(tfb)
            sourcepts = np.array(temp_b.points)

            #compute error between points in transformed target and source (append or add!?)
            for i in np.arange(0,len(sourcepts)):
                error.append( (sourcepts[i][0] - targetpts[i][0]) * (sourcepts[i][0] - targetpts[i][0])
                            + (sourcepts[i][1] - targetpts[i][1]) * (sourcepts[i][1] - targetpts[i][1])
                            + (sourcepts[i][2] - targetpts[i][2]) * (sourcepts[i][2] - targetpts[i][2]))

        print("first 5 errors = " + str(error[0:5]))
        return error   


    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------

    # for i, model in enumerate(models):
    #     opt.pushParamVector(group_name='model' + str(i) + '_t', data_key='models',
    #                         getter=partial(getterTranslation, i=i),
    #                         setter=partial(setterTranslation, i=i),
    #                         suffix=['x', 'y', 'z'])
    #     opt.pushParamVector(group_name='model' + str(i) + '_r', data_key='models',
    #                         getter=partial(getterRotation, i=i),
    #                         setter=partial(setterRotation, i=i),
    #                         suffix=['x', 'y', 'z'])

    for model_a, model_b in combinations(models, 2):
        print(str(model_a) + ' with ' + str(model_b))

        N = len(model_a.cloud.points)

        params = opt.getParamsContainingPattern('model' + str(model_a.name) + '_')  # for model a
        params.extend(opt.getParamsContainingPattern('model' +str(model_b.name) + '_'))  # for model b

        for n in range(0, N):
            residual_name = 'r_' + model_a.name + '_' + model_b.name + '_' + str(n)
            opt.pushResidual(name=residual_name, params=params)

    opt.printResiduals()

    # comb = combinations(np.arange(0, len(models)), 2)
    # for c in comb:
    #     # for a in range(0, len(models[c[0]])):
    #     for a in range(0, len(models[c[0]].points)):
    #         # opt.pushResidual(name='pts' + str(a), params=['t', 'r'])
    #         opt.pushResidual(name='r' + str(c[0]) + str(c[1]) + str("_") + str(a),
    #                          params=['t' + str(c[0]) + str(c[1]) + 'x',
    #                                  't' + str(c[0]) + str(c[1]) + 'y',
    #                                  't' + str(c[0]) + str(c[1]) + 'z',
    #                                  'r' + str(c[0]) + str(c[1]) + 'x',
    #                                  'r' + str(c[0]) + str(c[1]) + 'y',
    #                                  'r' + str(c[0]) + str(c[1]) + 'z'])
    #     # for a in range(0, 1):
    #     #     opt.pushResidual(name='cloud' + str(a), params=['t', 'r'])

    print('residuals = ' + str(opt.residuals))
    opt.computeSparseMatrix()
    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------

    # fig = plt.figure()
    # # ax = fig.add_subplot(111)

    # Visualize initial and final cloud
    vis = o3d.Visualizer()
    vis.create_window()
    drawAxes(vis)

    clouds_vis = np.empty(len(models), dtype=o3d.PointCloud) # for visualization purposes

    # models[0].cloud.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(models[0].cloud)
    for i, m in enumerate (models):

        # For visualization / update purpose need a copy of clouds
        clouds_vis[i] = cp.deepcopy(m.cloud)
        clouds_vis[i].paint_uniform_color(np.random.uniform(0,1,3))
        vis.add_geometry(clouds_vis[i])

    vis.update_renderer()
    print("Lock in open3D windows (vis) - press q to continue")
    vis.run()

    # # wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    # # if wm.waitForKey(0., verbose=False):
    # #     exit(0)

    def visualizationFunction(model):

        for i, m in enumerate(models):
            current_cloud = m.cloud
            trans_a = np.array([m.t[0], m.t[1], m.t[2]])
            angle_a = np.array([m.r[0], m.r[1], m.r[2]])
            tf_current = tf.compose_matrix(scale=None, shear=None, angles=angle_a, translate=trans_a, perspective=None)

            temp = cp.deepcopy(current_cloud)
            temp.transform(tf_current)
            clouds_vis[i].points = temp.points

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()

            # wm = KeyPressManager.KeyPressManager.WindowManager(fig)
            # if wm.waitForKey(0.01, verbose=False):
            #     exit(0)

    # opt.setVisualizationFunction(visualizationFunction, True)
    opt.setVisualizationFunction(visualizationFunction, False)

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
        optimization_options={'x_scale': 'jac', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-4})


    vis = o3d.Visualizer()
    vis.create_window()
    drawAxes(vis)
    ############################
    # Final models visualization
    for i, m in enumerate(models):
        trans_a = np.array([m.t[0], m.t[1], m.t[2]])
        angle_a = np.array([m.r[0], m.r[1], m.r[2]])
        tf_current = tf.compose_matrix(scale=None, shear=None, angles=angle_a, translate=trans_a, perspective=None)

        clouds_vis[i] = cp.deepcopy(m.cloud)
        clouds_vis[i].paint_uniform_color(np.random.uniform(0,1,3))
        clouds_vis[i].transform(tf_current)
        
        vis.add_geometry(clouds_vis[i])

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
    
    print("press q to quit")
    vis.run()

    # wm = KeyPressManager.KeyPressManager.WindowManager()
    # if wm.waitForKey():
    #     exit(0)
