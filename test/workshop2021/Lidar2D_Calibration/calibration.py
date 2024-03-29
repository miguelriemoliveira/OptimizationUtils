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
    dictionary_right={}
    dictionary_left = {}
    # Command line arguments
    parser = argparse.ArgumentParser(description='LIDAR calibration using OptimizationUtils')
    parser.add_argument('-j', '--json', type=str, required=True,
                        help='.json file to read the data points.')
    args = vars(parser.parse_args())
    # # print(args['json'])
    # # message = String(data = args['json'])
    # bag = rosbag.Bag(args['json'])
    #
    # for topic, msg, t in bag.read_messages(topics=['/left_scan']):
    #
    #   dictionary_left.update(message_converter.convert_ros_message_to_dictionary(msg))
    #
    # for topic, msg, t in bag.read_messages(topics=['/right_scan']):
    #
    #   dictionary_right.update(message_converter.convert_ros_message_to_dictionary(msg))
    #
    # bag.close()
    #
    # json_object_left = json.dumps(dictionary_left)
    # json_object_right = json.dumps(dictionary_right)
    # Defining lists
    txs = []
    tys = []
    angs = []
    print(args['json'])
    # Importing json
    data = jsonImporter(args['json'])

    #
    # # Retrieving number of collections
    cols = data['collections']

    # Running the program for each of the collections
    for col in cols:
        # Closing all previous figure
        plt.close('all')

        # Diving the data into left and right LIDAR
        data_left, data_right = divideDict(data, col)
        left_xs, left_ys, right_xs, right_ys, not_left_xs, not_left_ys, not_right_xs, not_right_ys = \
            dataTreatment(data_left, data_right)

        # Initiating module
        laser_model = LaserModel(0, 0, math.pi/4)

        # Initializing and viewing the plot
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(0, 0)
        ax.grid()
        # ax.axis([-20, 20, -20, 20])
        ax.axis([-5, 5, -5, 5])
        handle_left_laser = ax.plot(left_ys, left_xs, 'b+', label='Left LIDAR data')
        handle_not_left_laser = ax.plot(not_left_ys, not_left_xs, 'bo', markersize=2, label='Left LIDAR data not considered')
        handle_initial_right_laser = ax.plot(right_ys, right_xs, 'g+', label='Right LIDAR data before calibration')
        handle_not_right_laser = ax.plot(not_right_ys, not_right_xs, 'go', markersize=2, label='Right LIDAR data not considered')

        handle_right_laser = ax.plot(right_ys, right_xs, 'rx', label='Right LIDAR data after calibration')
        ax.legend()
        ax.invert_xaxis()

        # Partial lidar draws
        fig1 = plt.figure()
        ax1 = fig1.gca()
        ax1.plot(0, 0)
        ax1.grid()
        ax1.axis([-5, 5, -5, 5])
        handle_left_laser1 = ax1.plot(left_ys, left_xs, 'b+', label='Left LIDAR data')
        handle_not_left_laser1 = ax1.plot(not_left_ys, not_left_xs, 'o', markersize=2,
                                        label='Left LIDAR data not considered', color=(0.2,0.2,0.2))
        ax1.invert_xaxis()
        ax1.legend()

        # Partial lidar draws
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.plot(0, 0)
        ax2.grid()
        ax2.axis([-5, 5, -5, 5])
        handle_right_laser1 = ax2.plot(right_ys, right_xs, 'b+', label='Right LIDAR data')
        handle_not_right_laser1 = ax2.plot(not_right_ys, not_right_xs, 'o', markersize=2,
                                        label='Right LIDAR data not considered', color=(0.2,0.2,0.2))
        ax2.invert_xaxis()
        ax2.legend()

        # plt.draw()
        plt.show()
        # plt.waitforbuttonpress(0)

        exit(0)

        opt = OptimizationUtils.Optimizer()

        # Add data models
        opt.addDataModel('laser_model', laser_model)

        # -----------------------------------------------------
        # Define parameters
        # -----------------------------------------------------

        # Laser parameters
        def getterLaser(data, prop):
            if prop == 'tx':
                return [data.tx]
            elif prop == 'ty':
                return [data.ty]
            elif prop == 'ang':
                return [data.ang]

        def setterLaser(data, value, prop):
            if prop == 'tx':
                data.tx = value
            elif prop == 'ty':
                data.ty = value
            elif prop == 'ang':
                data.ang = value

        opt.pushParamScalar(group_name='laser_tx', data_key='laser_model',
                            getter=partial(getterLaser, prop='tx'),
                            setter=partial(setterLaser, prop='tx'))
        opt.pushParamScalar(group_name='laser_ty', data_key='laser_model',
                            getter=partial(getterLaser, prop='ty'),
                            setter=partial(setterLaser, prop='ty'))
        opt.pushParamScalar(group_name='laser_ang', data_key='laser_model',
                            getter=partial(getterLaser, prop='ang'),
                            setter=partial(setterLaser, prop='ang'))

        opt.printParameters()

        # -----------------------------------------------------
        # Define objective function
        # -----------------------------------------------------
        def objectiveFunction(data_models):
            # retrieve data models
            laser_model = data_models['laser_model']

            # Initialize the residuals
            errors = []

            # Compute observations from model
            right_xs_model, right_ys_model = laser_model.getCoords(right_xs, right_ys)

            counter = 0
            # Compute error
            # errors from the laser model
            for idx, x in enumerate(left_xs):
                y = left_ys[idx]
                error_min = sys.float_info.max
                for idx2, x_m in enumerate(right_xs_model):
                    y_m = right_ys_model[idx2]
                    counter += 1
                    error = abs(x - x_m) + abs(y - y_m)
                    if error < error_min:
                        error_min = error
                errors.append(error_min)
            return errors

        opt.setObjectiveFunction(objectiveFunction)

        # -----------------------------------------------------
        # Define residuals
        # -----------------------------------------------------

        for idx, x in enumerate(right_xs):
            opt.pushResidual(name='laser_r' + str(idx), params=['laser_tx', 'laser_ty', 'laser_ang'])

        opt.printResiduals()

        # -----------------------------------------------------
        # Compute sparse matrix
        # -----------------------------------------------------
        opt.computeSparseMatrix()
        opt.printSparseMatrix()

        # -----------------------------------------------------
        # Define visualization function
        # -----------------------------------------------------
        def visualizationFunction(data_models):
            # retrieve data models
            laser_model = data_models['laser_model']
            #
            # print('Visualization function called ...')
            # print('tx=' + str(laser_model.tx))
            # print('ty=' + str(laser_model.ty))
            # print('ang=' + str(laser_model.ang))

            right_xs_model, right_ys_model = laser_model.getCoords(right_xs, right_ys)

            # laser visualization
            handle_right_laser[0].set_xdata(right_xs_model)
            handle_right_laser[0].set_ydata(right_ys_model)

            wm = OptimizationUtils.KeyPressManager.WindowManager(fig)
            if wm.waitForKey(0.01, verbose=False):
                exit(0)

        opt.setVisualizationFunction(visualizationFunction, True)

        # -----------------------------------------------------
        # Start optimization
        # -----------------------------------------------------
        opt.startOptimization(
            optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6, 'diff_step': None})
        opt.printParameters()
        tx = getterLaser(laser_model, 'tx')
        ty = getterLaser(laser_model, 'ty')
        ang = getterLaser(laser_model, 'ang')
        txs.append(tx[0][0])
        tys.append(ty[0][0])
        angs.append(ang[0][0])
        wm = OptimizationUtils.KeyPressManager.WindowManager(fig)
        if wm.waitForKey(0.01, verbose=False):
            exit(0)
    print('The values for calibration are:\n tx: ' + str(-mean(txs))+ ';\n ty: ' + str(-mean(tys)) + ';\n ang: ' + str(-mean(angs)))

if __name__ == "__main__":
    main()
