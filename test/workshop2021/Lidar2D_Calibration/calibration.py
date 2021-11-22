#!/usr/bin/python3

import OptimizationUtils.OptimizationUtils as OptimizationUtils
from json_reader import *

def main():

    # Calling json_reader functions
    data_left, data_right = jsonImporter('data/data_collected.json')
    left_xs, left_ys, right_xs, right_ys = dataViewer(data_left, data_right)

    # Initializing and viewing the plot
    plt.plot(0, 0)
    plt.grid()
    plt.axis([-20, 20, -20, 20])

    plt.plot(left_xs, left_ys, 'bo')
    plt.plot(right_xs, right_ys, 'ro')
    plt.show()

    # opt = OptimizationUtils.Optimizer()

if __name__ == "__main__":
    main()