#!/usr/bin/env python3

import argparse
import random

import colorama
import matplotlib.pyplot as plt
import pickle

import numpy as np


class LineModel:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def getY(self, x):
        return self.m * x + self.b

    def getYs(self, xs):
        ys = []
        for x in xs:
            ys.append(self.getY(x))
        return ys


class ParabolaModel:
    # y = a*(x-b)^2+c
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def getY(self, x):
        return self.a * (x - self.b) ** 2 + self.c

    def getYs(self, xs):
        ys = []
        for x in xs:
            ys.append(self.getY(x))
        return ys


def main():
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

    # -----------------------------------------------------
    # Plot data points
    xs_observations_left = [item[0] for item in points if item[0] < 0]
    ys_observations_left = [item[1] for item in points if item[0] < 0]
    xs_observations_right = [item[0] for item in points if item[0] >= 0]
    ys_observations_right = [item[1] for item in points if item[0] >= 0]
    plt.plot(xs_observations_left, ys_observations_left, '+c')
    plt.plot(xs_observations_right, ys_observations_right, '+b')
    plt.axis([-5, 5, -5, 5])
    plt.grid()

    # Create models
    model_left = LineModel(m=1.1, b=-2)
    model_right = ParabolaModel(a=1.5, b=0.5, c=1.0)

    # compute error given observations and model
    ys_model_observations_left = model_left.getYs(xs_observations_left)
    h_plot_model_observations = plt.plot(xs_observations_left, ys_model_observations_left, 'or')

    ys_model_observations_right = model_right.getYs(xs_observations_right)
    h_plot_model_observations = plt.plot(xs_observations_right, ys_model_observations_right, 'or')

    # plot current model
    xs_model_left = list(np.linspace(-5, 0, 100))
    ys_model_left = model_right.getYs(xs_model_left)
    xs_model_right = list(np.linspace(0, 5, 100))
    ys_model_right = model_right.getYs(xs_model_right)
    h_plot_model = plt.plot(xs_model_left.extend(xs_model_right), ys_model_left.extend(ys_model_right), '-.b')

    # print(h_plot_model)
    plt.draw()
    plt.waitforbuttonpress(1)

    # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------
    # best_line_model = LineModel(1, 0)
    best_line_model = ParabolaModel(1, 0,0)
    best_total_error = 10000000000  # just a large value
    ys_best_model = best_line_model.getYs(xs_model)
    h_plot_best_model = plt.plot(xs_model, ys_best_model, '-m')

    while True:
        model_right.a = random.uniform(-4, 4)
        model_right.b = random.uniform(-4, 4)
        model_right.c = random.uniform(-4, 4)

        # compute error given observations and model
        ys_model_observations_right = model_right.getYs(xs_observations_right)

        error = []
        for x, y_from_model, y_observed in zip(xs_observations_right, ys_model_observations_right, ys_observations_right):
            error.append(abs(y_observed - y_from_model))

        total_error = sum(error)
        print('Total error is: ' + str(total_error))

        # Replace best model if error is smaller
        if total_error < best_total_error:  # found a new minimum, replacing best
            best_line_model.a = model_right.a
            best_line_model.b = model_right.b
            best_line_model.c = model_right.c
            best_total_error = total_error
            print(colorama.Fore.RED + 'Found new best model with total error ' + str(best_total_error)
                  + colorama.Style.RESET_ALL)
            plt.title('Best model a=' +
                      str(round(best_line_model.a, 3)) + ' b=' +
                      str(round(best_line_model.b, 3)) + ' c=' +
                      str(round(best_line_model.c, 3)) + ' error=' +
                      str(round(best_total_error, 3)))

        # Visualization

        # update plot for best model
        plt.setp(h_plot_best_model, data=(xs_model, best_line_model.getYs(xs_model)))

        # update plot for current model
        plt.setp(h_plot_model_observations, data=(xs_observations_right, ys_model_observations_right))

        # plot line model
        ys_model = model_right.getYs(xs_model)
        plt.setp(h_plot_model, data=(xs_model, ys_model))

        plt.draw()
        key = plt.waitforbuttonpress(0.1)
        if key is True:  # press any key to exit
            break

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------


if __name__ == '__main__':
    main()
