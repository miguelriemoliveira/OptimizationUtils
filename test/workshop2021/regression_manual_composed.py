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


class ParabolaModel():
    # y = a * (x - b)** 2 + c
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def getY(self, x):
        return self.a * ((x - self.b) ** 2) + self.c

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

    # Create parabola model
    parabola_model = ParabolaModel(1, 0.5, 1)
    best_parabola_model = ParabolaModel(1, 0.5, 0)

    # Visualization
    # create a figure
    plt.plot(0, 0)
    plt.grid()
    plt.axis([-5, 5, -5, 5])

    # draw observations
    xs_observations_left = [item['x'] for item in points if item['x'] < 0]
    ys_observations_left = [item['y'] for item in points if item['x'] < 0]
    plt.plot(xs_observations_left, ys_observations_left, 'bo')

    xs_observations_right = [item['x'] for item in points if item['x'] >= 0]
    ys_observations_right = [item['y'] for item in points if item['x'] >= 0]
    plt.plot(xs_observations_right, ys_observations_right, 'co')

    handle_error_anchors_left = plt.plot(xs_observations_left, ys_observations_left, '.r')

    # draw line model
    xs_left = list(np.linspace(-5, 0, 100))
    ys = line_model.getYs(xs_left)
    handle_model_left_plot = plt.plot(xs_left, ys, '-.k')

    # draw parabola model
    xs_right = list(np.linspace(0, 5, 100))
    ys = parabola_model.getYs(xs_right)
    handle_model_right_plot = plt.plot(xs_right, ys, '-.k')

    # draw best line model
    handle_best_model_plot = plt.plot(xs_observations_left, best_line_model.getYs(xs_observations_left), '-m')

    # draw best line model
    handle_best_model_plot_right = plt.plot(xs_right, best_parabola_model.getYs(xs_right), '-r')

    # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------
    while True:
        # randomize model parameters
        line_model.m = np.random.uniform(-2, 2)
        line_model.b = np.random.uniform(-5, 5)

        parabola_model.a = np.random.uniform(-5, 5)
        parabola_model.b = np.random.uniform(-5, 5)
        parabola_model.c = np.random.uniform(-5, 5)

        # compute error as vertical distance
        ys_observations_from_model_left = line_model.getYs(xs_observations_left)
        ys_observations_from_model_right = parabola_model.getYs(xs_observations_right)

        # errors = [abs(y_o - y_ofm) for y_o, y_ofm in zip(ys_observations, ys_observations_from_model)]
        errors = []

        # errors from the line model
        for y_o, y_ofm in zip(ys_observations_left, ys_observations_from_model_left):
            error = abs(y_o - y_ofm)
            errors.append(error)

        # errors from the parabola
        for y_o, y_ofm in zip(ys_observations_right, ys_observations_from_model_right):
            error = abs(y_o - y_ofm)
            errors.append(error)

        total_error = sum(errors)
        print(total_error)

        # update best model (if needed)
        if total_error < total_best_error:  # found better line parameters
            total_best_error = total_error
            best_line_model.m = line_model.m
            best_line_model.b = line_model.b
            best_parabola_model.a = parabola_model.a
            best_parabola_model.b = parabola_model.b
            best_parabola_model.c = parabola_model.c


        # Visualization
        # line
        plt.setp(handle_model_left_plot, data=(xs_left, line_model.getYs(xs_left)))  # update the line draw
        plt.setp(handle_best_model_plot, data=(xs_left, best_line_model.getYs(xs_left)))  # update the best line draw
        plt.setp(handle_error_anchors_left, data=(xs_observations_left, ys_observations_from_model_left))  # update the anchor erros left model

        # parabola
        plt.setp(handle_model_right_plot, data=(xs_right, parabola_model.getYs(xs_right)))  # update the line draw
        plt.setp(handle_best_model_plot_right, data=(xs_right, best_parabola_model.getYs(xs_right)))  # update the line draw

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not key is None:
            print('Pressed a key, terminating')
            break

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------


if __name__ == '__main__':
    main()
