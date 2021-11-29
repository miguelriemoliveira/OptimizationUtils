#!/usr/bin/python3

import rosbag
import OptimizationUtils.OptimizationUtils as OptimizationUtils
from json_reader import *
import matplotlib.pyplot as plt
import math
import numpy as np
from functools import partial
import sys
import argparse
from statistics import mean
from rospy_message_converter import message_converter
from std_msgs.msg import String


# -----------------------------------------------------
# Define visualization function
# -----------------------------------------------------
def visualizationFunction(data_models):
    # retrieve data models
    # laser_model = data_models['laser_model']
    collection = data_models['collection']
    left_laser_msg = collection['data']['left_laser']
    right_laser_msg = collection['data']['right_laser']
    graphics = data_models['graphics']
    # print('Visualization function called ...')

    # update right transf laser visualization
    graphics['right_laser_transf'][0].set_xdata(right_laser_msg['xs_transf'])
    graphics['right_laser_transf'][0].set_ydata(right_laser_msg['ys_transf'])

    graphics['right_laser_filtered_transf'][0].set_xdata(right_laser_msg['xs_filtered_transf'])
    graphics['right_laser_filtered_transf'][0].set_ydata(right_laser_msg['ys_filtered_transf'])

    for left_x, left_y, closest_right_x, closest_right_y, handle in zip(left_laser_msg['xs_filtered'],
                                                                        left_laser_msg['ys_filtered'],
                                                                        left_laser_msg['closest_points_x'],
                                                                        left_laser_msg['closest_points_y'],
                                                                        graphics['closest_points']):
        xs = [left_x, closest_right_x]
        ys = [left_y, closest_right_y]
        handle[0].set_xdata(xs)
        handle[0].set_ydata(ys)

    wm = OptimizationUtils.KeyPressManager.WindowManager(graphics['fig'])
    if wm.waitForKey(0.01, verbose=False):
        exit(0)


class LaserModel():
    def __init__(self, tx, ty, ang):
        self.tx = tx
        self.ty = ty
        self.ang = ang

    def getCoord(self, x, y):
        if type(self.tx) is list:
            tx = self.tx[0]
        else:
            tx = self.tx

        if type(self.ty) is list:
            ty = self.ty[0]
        else:
            ty = self.ty

        if type(self.ang) is list:
            ang = self.ang[0]
        else:
            ang = self.ang

        matrix = np.array([[math.cos(ang), -math.sin(ang), tx],
                           [math.sin(ang), math.cos(ang), ty],
                           [0, 0, 1]])

        in_matrix = np.array([[x], [y], [1]])

        # print(matrix)
        # exit()
        out_matrix = np.matmul(matrix, in_matrix)

        x_out = out_matrix[0]
        y_out = out_matrix[1]

        return x_out[0], y_out[0]

    def getCoords(self, xs, ys):
        x_ol = []
        y_ol = []
        for idx, y in enumerate(ys):
            x = xs[idx]
            (x_o, y_o) = self.getCoord(x, y)
            x_ol.append(x_o)
            y_ol.append(y_o)
        return x_ol, y_ol


def main():
    dictionary_right = {}
    dictionary_left = {}
    # Command line arguments
    parser = argparse.ArgumentParser(description='LIDAR calibration using OptimizationUtils')
    parser.add_argument('-j', '--json', type=str, required=True,
                        help='.json file to read the data points.')
    args = vars(parser.parse_args())
    print(args['json'])

    # Importing json
    data = jsonImporter(args['json'])

    # Retrieving number of collections
    # use only collection 0

    collection = data['collections']['0']

    left_laser_msg = collection['data']['left_laser']
    right_laser_msg = collection['data']['right_laser']

    # Compute cartesian coordinates left laser
    left_laser_msg['xs'] = []
    left_laser_msg['ys'] = []
    left_laser_msg['xs_filtered'] = []
    left_laser_msg['ys_filtered'] = []
    theta_min = - math.pi
    # theta_max = 0
    theta_max = math.pi
    for idx, range in enumerate(left_laser_msg['ranges']):
        if range > 0:  # valid measurement
            theta = left_laser_msg['angle_min'] + idx * left_laser_msg['angle_increment']
            x = range * math.cos(theta)
            y = range * math.sin(theta)
            left_laser_msg['xs'].append(x)
            left_laser_msg['ys'].append(y)
            if theta > theta_min and theta < theta_max:
                left_laser_msg['xs_filtered'].append(x)
                left_laser_msg['ys_filtered'].append(y)

    # Compute cartesian coordinates left laser
    right_laser_msg['xs'] = []
    right_laser_msg['ys'] = []
    right_laser_msg['xs_filtered'] = []
    right_laser_msg['ys_filtered'] = []
    # theta_min = 0
    theta_min = -math.pi
    theta_max = math.pi
    for idx, range in enumerate(right_laser_msg['ranges']):
        if range > 0:  # valid measurement
            theta = right_laser_msg['angle_min'] + idx * right_laser_msg['angle_increment']
            x = range * math.cos(theta)
            y = range * math.sin(theta)
            right_laser_msg['xs'].append(x)
            right_laser_msg['ys'].append(y)
            if theta > theta_min and theta < theta_max:
                right_laser_msg['xs_filtered'].append(x)
                right_laser_msg['ys_filtered'].append(y)

    # Initiating module
    right_laser_msg['laser_model'] = LaserModel(0, -0.7, -math.pi / 1.7)  # reasonably accurate first guess
    # right_laser_msg['laser_model'] = LaserModel(0, -0.7, math.pi / 1.7) # inaccurate first guess

    right_laser_msg['xs_transf'], right_laser_msg['ys_transf'] = right_laser_msg['laser_model'].getCoords(
        right_laser_msg['xs'], right_laser_msg['ys'])
    right_laser_msg['xs_filtered_transf'], right_laser_msg['ys_filtered_transf'] = \
        right_laser_msg['laser_model'].getCoords(right_laser_msg['xs_filtered'], right_laser_msg['ys_filtered'])

    # Create figure
    graphics = {}
    graphics['fig'] = plt.figure()
    ax = graphics['fig'].gca()
    ax.plot(0, 0)
    ax.grid()
    ax.axis([-5, 5, -5, 5])
    graphics['left_laser'] = ax.plot(left_laser_msg['xs'], left_laser_msg['ys'], '.', label='Left LIDAR data',
                                     color=(0.5, 0.1, 0.1))
    graphics['left_laser_filtered'] = ax.plot(left_laser_msg['xs_filtered'], left_laser_msg['ys_filtered'], 'o',
                                              label='left filtered', markersize=15,
                                              color=(1, 0, 0), markerfacecolor='none')

    graphics['right_laser'] = ax.plot(right_laser_msg['xs'], right_laser_msg['ys'], '.', label='Right LIDAR data',
                                      color=(0.1, 0.5, 0.1))
    graphics['right_laser_filtered'] = ax.plot(right_laser_msg['xs_filtered'], right_laser_msg['ys_filtered'], 'o',
                                               label='right filtered',
                                               color=(0, 1, 0), markerfacecolor='none')

    graphics['right_laser_transf'] = ax.plot(right_laser_msg['xs_transf'], right_laser_msg['ys_transf'], '.',
                                             label='right transf', color=(0.1, 0.1, 0.5))
    graphics['right_laser_filtered_transf'] = ax.plot(right_laser_msg['xs_filtered_transf'],
                                                      right_laser_msg['ys_filtered_transf'], 'o',
                                                      label='right filtered transf', color=(0, 0, 1),
                                                      markerfacecolor='none')
    graphics['closest_points'] = []
    for left_x, left_y in zip(left_laser_msg['xs_filtered'], left_laser_msg['ys_filtered']):
        xs = [left_x, 0]
        ys = [left_y, 0]
        graphics['closest_points'].append(ax.plot(xs, ys, '-.', color=(1., 1., 0.0)))

    ax.legend()
    plt.draw()
    plt.waitforbuttonpress(0)
    # exit(0)

    # ----------------------------------------------------

    # left_xs = left_laser_msg['xs_filtered']
    # left_ys = left_laser_msg['ys_filtered']
    # right_xs = right_laser_msg['xs_filtered']
    # right_ys = right_laser_msg['ys_filtered']

    opt = OptimizationUtils.Optimizer()

    # Add data models
    # opt.addDataModel('laser_model', laser_model)
    opt.addDataModel('collection', collection)
    opt.addDataModel('graphics', graphics)

    # -----------------------------------------------------
    # Define parameters
    # -----------------------------------------------------

    # Laser parameters
    # def getterLaser(collection, prop):
    #     if prop == 'tx':
    #         return [collection['data']['right_laser']['laser_model'].tx]
    #     elif prop == 'ty':
    #         return [collection['data']['right_laser']['laser_model'].ty]
    #     elif prop == 'ang':
    #         return [collection['data']['right_laser']['laser_model'].ang]
    #
    # def setterLaser(collection, value, prop):
    #     if prop == 'tx':
    #         collection['data']['right_laser']['laser_model'].tx = value
    #     elif prop == 'ty':
    #         collection['data']['right_laser']['laser_model'].ty = value
    #     elif prop == 'ang':
    #         collection['data']['right_laser']['laser_model'].ang = value
    #
    # opt.pushParamScalar(group_name='laser_tx', data_key='collection',
    #                     getter=partial(getterLaser, prop='tx'),
    #                     setter=partial(setterLaser, prop='tx'))
    # opt.pushParamScalar(group_name='laser_ty', data_key='collection',
    #                     getter=partial(getterLaser, prop='ty'),
    #                     setter=partial(setterLaser, prop='ty'))
    # opt.pushParamScalar(group_name='laser_ang', data_key='collection',
    #                     getter=partial(getterLaser, prop='ang'),
    #                     setter=partial(setterLaser, prop='ang'))

    def getterPose(collection):
        return [collection['data']['right_laser']['laser_model'].tx,
                collection['data']['right_laser']['laser_model'].ty,
                collection['data']['right_laser']['laser_model'].ang]

    def setterLaser(collection, values):
        collection['data']['right_laser']['laser_model'].tx = values[0]
        collection['data']['right_laser']['laser_model'].ty = values[1]
        collection['data']['right_laser']['laser_model'].ang = values[2]

    opt.pushParamVector(group_name='laser_', data_key='collection',
                        getter=getterPose,
                        setter=setterLaser, suffix=['tx', 'ty', 'ang'])

    opt.printParameters()

    # -----------------------------------------------------
    # Define objective function
    # -----------------------------------------------------
    def objectiveFunction(data_models):
        # retrieve data models
        # laser_model = data_models['laser_model']
        collection = data_models['collection']
        left_laser_msg = collection['data']['left_laser']
        right_laser_msg = collection['data']['right_laser']

        # Initialize the residuals
        errors = []

        # Compute observations from model

        right_laser_msg['xs_transf'], right_laser_msg['ys_transf'] = right_laser_msg['laser_model'].getCoords(
            right_laser_msg['xs'], right_laser_msg['ys'])

        right_laser_msg['xs_filtered_transf'], right_laser_msg['ys_filtered_transf'] = right_laser_msg[
            'laser_model'].getCoords(right_laser_msg['xs_filtered'], right_laser_msg['ys_filtered'])

        # right_xs_model, right_ys_model = laser_model.getCoords(right_xs, right_ys)
        error_threshold = 1.0
        counter = 0
        left_laser_msg['closest_points_x'] = []
        left_laser_msg['closest_points_y'] = []
        # Compute error
        for left_x, left_y in zip(left_laser_msg['xs_filtered'], left_laser_msg['ys_filtered']):
            error_min = sys.float_info.max
            min_x, min_y = 0, 0

            for right_x, right_y in zip(right_laser_msg['xs_filtered_transf'], right_laser_msg['ys_filtered_transf']):

                error = math.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2)
                if error < error_min:
                    error_min = error
                    min_x = right_x
                    min_y = right_y
                    # TODO save index of right minimum for each left

            if error_min < error_threshold:
                errors.append(error_min)
                left_laser_msg['closest_points_x'].append(min_x)
                left_laser_msg['closest_points_y'].append(min_y)
            else:
                left_laser_msg['closest_points_x'].append(left_x)
                left_laser_msg['closest_points_y'].append(left_y)
                errors.append(0)

            # errors from the laser model
        # for idx, x in enumerate(left_xs):
        #     y = left_ys[idx]
        #     error_min = sys.float_info.max
        #     for idx2, x_m in enumerate(right_xs_model):
        #         y_m = right_ys_model[idx2]
        #         counter += 1
        #         error = abs(x - x_m) + abs(y - y_m)
        #         if error < error_min:
        #             error_min = error
        #     errors.append(error_min)
        return errors

    opt.setObjectiveFunction(objectiveFunction)

    # -----------------------------------------------------
    # Define residuals
    # -----------------------------------------------------
    params = opt.getParameters()
    for idx, _ in enumerate(left_laser_msg['xs_filtered']):
        opt.pushResidual(name='laser_r' + str(idx), params=params)

    opt.printResiduals()

    # -----------------------------------------------------
    # Compute sparse matrix
    # -----------------------------------------------------
    opt.computeSparseMatrix()
    opt.printSparseMatrix()

    opt.setVisualizationFunction(visualizationFunction, True)

    # -----------------------------------------------------
    # Start optimization
    # -----------------------------------------------------
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6, 'diff_step': None})
    opt.printParameters()


if __name__ == "__main__":
    main()
