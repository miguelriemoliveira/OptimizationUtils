#!/usr/bin/env python3

import argparse
import random
import sys
import colorama
import matplotlib.pyplot as plt
import pickle
import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import OptimizationUtils.KeyPressManager as KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils

def main():
    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------
    # -----------------------------------------------------
    # Command line arguments
    parser = argparse.ArgumentParser(description='Regression example 1')
    parser.add_argument('-inp', '--input_numpy', type=str, required=True,
                        help='Filename to read the data points JIH observations (numpy).')
    parser.add_argument('-mi', '--monotonically_increasing', action='store_true', default=False,
                        help='Force function to be monotonically increasing')
    parser.add_argument('-deg', '--degree', type=int, required=True, help = 'Degree of the polynomial regression')
    
    args = vars(parser.parse_args())
    print(args)

    # -----------------------------------------------------
    # Load file with data points
    print('Loading file to ' + str(args['input_numpy']))
    with open(args['input_numpy'], 'rb') as file_handle:
        jih = np.load(file_handle)

    jih_b = jih[0, :, :]

    # Convert histogram into observations
    tgt_colors = []
    src_colors = []
    for tgt_color in range(jih_b.shape[1]):
        for src_color in range(jih_b.shape[0]):
            if jih_b[src_color, tgt_color] > 0:
                src_colors.extend(
                    [src_color] * jih_b[src_color, tgt_color])  # create observations from histogram
                tgt_colors.extend(
                    [tgt_color] * jih_b[src_color, tgt_color])  # create observations from histogram

    xs_obs = tgt_colors
    ys_obs = src_colors
    
    # Numpy Polynomial Regression
    
    np_poly = np.polyfit(xs_obs, ys_obs, args['degree'])
    p = np.poly1d(np_poly)
    
   
    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------
    # create a figure
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(0, 0)
    ax.grid()
    ax.axis([0, 255, 0, 255])

    # draw observations

    ax.plot(xs_obs, ys_obs, 'bo')

    # draw parabola model
    xs = list(np.linspace(0, 255, 256))
    ys = p(xs)
    
    ax.plot(xs, ys, '-g')
    plt.waitforbuttonpress(0)
    

    filename = '{}_np_polynomial'.format(args['degree'])
    fig.savefig(filename)

    



if __name__ == '__main__':
    main()