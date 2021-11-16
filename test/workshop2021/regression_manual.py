#!/usr/bin/env python3

import argparse
import random
import sys

import colorama
import matplotlib.pyplot as plt
import pickle

import numpy as np


# Use a line model
# y = m x + b
class LineModel():
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def getY(self, x):
        return self.m * x + self.b

    def getYs(self, xs):
        return [self.getY(item) for item in xs]


def main():
    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------
    # -----------------------------------------------------
    # Command line arguments
    parser = argparse.ArgumentParser(description='Regression example 1')
    parser.add_argument('-i', '--input', type=str, default='points.pkl', help='Filename to read.')
    args = vars(parser.parse_args())
    print(args)

    # -----------------------------------------------------
    # Load file with data points
    print('Loading file to ' + str(args['input']))
    with open(args['input'], 'rb') as file_handle:
        points = pickle.load(file_handle)

    # print(points)

    # Create line model
    line_model = LineModel(1, 0.5)
    best_line_model = LineModel(1, 0.5)
    total_best_error = sys.float_info.max

    # Visualization
    # create a figure
    plt.plot(0, 0)
    plt.grid()
    plt.axis([-5, 5, -5, 5])

    # draw observations
    xs_observations = [item['x'] for item in points]
    ys_observations = [item['y'] for item in points]
    plt.plot(xs_observations, ys_observations, 'bo')

    # draw line model
    xs = list(np.linspace(-5, 5, 100))
    ys = line_model.getYs(xs)
    handle_model_plot = plt.plot(xs, ys, '-.k')

    # draw best line model
    ys = best_line_model.getYs(xs)
    handle_best_model_plot = plt.plot(xs, ys, '-m')

    # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------
    while True:
        # randomize model parameters
        line_model.m = np.random.uniform(-2, 2)
        line_model.b = np.random.uniform(-5, 5)

        # draw line model
        ys = line_model.getYs(xs)

        # compute error as vertical distance
        ys_observations_from_model = line_model.getYs(xs_observations)

        # errors = [abs(y_o - y_ofm) for y_o, y_ofm in zip(ys_observations, ys_observations_from_model)]
        errors = []
        for y_o, y_ofm in zip(ys_observations, ys_observations_from_model):
            error = abs(y_o - y_ofm)
            errors.append(error)

        total_error = sum(errors)
        print(total_error)

        # update best model (if needed)
        if total_error < total_best_error: # found better line parameters
            total_best_error = total_error
            best_line_model.m = line_model.m
            best_line_model.b = line_model.b

        # Visualization
        plt.setp(handle_model_plot, data=(xs, ys))  # update the line draw
        plt.setp(handle_best_model_plot, data=(xs, best_line_model.getYs(xs)))  # update the best line draw

        plt.draw()
        key = plt.waitforbuttonpress(1)
        if not key is None:
            print('Pressed a key, terminating')
            break

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------


if __name__ == '__main__':
    main()
