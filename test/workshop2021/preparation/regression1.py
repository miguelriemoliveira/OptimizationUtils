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
    xs_observations = [item[0] for item in points]
    ys_observations = [item[1] for item in points]
    plt.plot(xs_observations, ys_observations, '+b')
    plt.axis([-5, 5, -5, 5])
    plt.grid()
    # plt.draw()
    # plt.waitforbuttonpress(1)

    # Create line model
    line_model = LineModel(m=1.1, b=-2)  # TODO randomize initial values

    # compute error given observations and model
    ys_model_observations = line_model.getYs(xs_observations)
    h_plot_model_observations = plt.plot(xs_observations, ys_model_observations, 'or')

    # plot current line model
    xs_model = list(np.linspace(-5, 5, 100))
    ys_model = line_model.getYs(xs_model)

    h_plot_model = plt.plot(xs_model, ys_model, '-.k')
    # print(h_plot_model)
    plt.draw()
    plt.waitforbuttonpress(1)

    # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------
    best_line_model = LineModel(1, 0)
    best_total_error = 10000000000  # just a large value
    ys_best_model = best_line_model.getYs(xs_model)
    h_plot_best_model = plt.plot(xs_model, ys_best_model, '-m')

    while True:
        line_model.m = random.uniform(-2, 2)
        line_model.b = random.uniform(-4, 4)

        # compute error given observations and model
        ys_model_observations = line_model.getYs(xs_observations)

        error = []
        for x, y_from_model, y_observed in zip(xs_observations, ys_model_observations, ys_observations):
            error.append(abs(y_observed - y_from_model))

        total_error = sum(error)
        print('Total error is: ' + str(total_error))

        # Replace best model if error is smaller
        if total_error < best_total_error: # found a new minimum, replacing best
            best_line_model.m = line_model.m
            best_line_model.b = line_model.b
            best_total_error = total_error
            print(colorama.Fore.RED + 'Found new best model with total error ' + str(best_total_error)
                  + colorama.Style.RESET_ALL)
            plt.title('Best model m=' +
                      str(round(best_line_model.m,3)) + ' b=' +
                      str(round(best_line_model.b,3)) + ' error=' +
                      str(round(best_total_error,3)))


        # Visualization

        # update plot for best model
        plt.setp(h_plot_best_model, data=(xs_model, best_line_model.getYs(xs_model)))

        # update plot for current model
        plt.setp(h_plot_model_observations, data=(xs_observations, ys_model_observations))

        # plot line model
        ys_model = line_model.getYs(xs_model)
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
