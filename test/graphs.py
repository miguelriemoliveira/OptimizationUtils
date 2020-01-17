#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS
# -------------------------------------------------------------------------------
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import inv
import argparse
import cv2

if __name__ == "__main__":
    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-json", "--json_file",
                    help="Json file containing input dataset from optimization procedure.", type=str, required=True)

    args = vars(ap.parse_args())
    # print("\nArgument list=" + str(args) + '\n')

    # ---------------------------------------
    # --- INITIALIZATION Read data from file and read sensors that will be compared
    # ---------------------------------------
    """ Loads a json file containing the chessboards poses for each collection"""
    f = open(args['json_file'], 'r')

    data = json.load(f)
    n_iter = 0
    n_eva = 0

    for key, value in data.items():
        n_iter += 1
        for key2, value2 in data[key].items():
            n_eva += 1

    avg_errors_left_laser = np.zeros((1, n_eva), np.float32)
    avg_errors_right_laser = np.zeros((1, n_eva), np.float32)
    avg_errors_top_left_camera = np.zeros((1, n_eva), np.float32)
    avg_errors_top_right_camera = np.zeros((1, n_eva), np.float32)
    i = 0
    for key, value in data.items():
        for key2, value2 in data[key].items():
            avg_errors_left_laser[0, i] = value2['per_sensor']['left_laser']['avg']
            avg_errors_right_laser[0, i] = value2['per_sensor']['right_laser']['avg']
            avg_errors_top_left_camera[0, i] = value2['per_sensor']['top_left_camera']['avg']
            avg_errors_top_right_camera[0, i] = value2['per_sensor']['top_right_camera']['avg']
            i += 1

    print(n_iter)
    print(n_eva)
    # print("left laser:")
    # print(avg_errors_left_laser)
    # print("right laser:")
    # print(avg_errors_right_laser)
    # print("left camera:")
    # print(avg_errors_top_left_camera)
    # print("right laser:")
    # print(avg_errors_top_right_camera)

    laser_max_l = np.amax(avg_errors_left_laser)
    laser_max_r = np.amax(avg_errors_right_laser)
    laser_min_l = np.amin(avg_errors_left_laser)
    laser_min_r = np.amin(avg_errors_right_laser)
    laser_max = np.amax([laser_max_l, laser_max_r])
    laser_min = np.amin([laser_min_l, laser_min_r])

    camera_max_l = np.amax(avg_errors_top_left_camera)
    camera_max_r = np.amax(avg_errors_top_right_camera)
    camera_min_l = np.amin(avg_errors_top_left_camera)
    camera_min_r = np.amin(avg_errors_top_right_camera)
    camera_max = np.amax([camera_max_l, camera_max_r])
    camera_min = np.amin([camera_min_l, camera_min_r])

    # -------------------------------------------------------------------
    # ------ SEE THE DIFFERENCE IN A SCATTER PLOT
    # -------------------------------------------------------------------
    colors = cm.tab20b(np.linspace(0, 1, 4))
    avg_errors_left_laser = avg_errors_left_laser.reshape(-1, 1)
    avg_errors_top_left_camera = avg_errors_top_left_camera.reshape(-1, 1)
    avg_errors_top_right_camera = avg_errors_top_right_camera.reshape(-1, 1)
    avg_errors_right_laser = avg_errors_right_laser.reshape(-1, 1)
    t = np.arange(start=0, stop=n_eva, step=1)

    fig, ax1 = plt.subplots()
    # plt.xlabel('Number of Cost Function Evaluations')
    # plt.ylabel('y error (pixels)')

    plt.grid(True, color='k', linestyle='--', linewidth=0.1)
    string = "Residual average values for each sensor, along the optimization procedure"
    plt.title(string)

    # delta = 0.5
    ax1.set_xlim(0, 160)
    ax1.set_ylim(0, laser_max + laser_max/10)

    color = 'xkcd:rich blue'
    ax1.set_xlabel('Cost Function Evaluations')
    ax1.set_ylabel('meters (m)', color=color)
    ax1.plot(t, avg_errors_left_laser, color='xkcd:dark blue', label="left_laser") # color='xkcd:amber'
    ax1.plot(t, avg_errors_right_laser, color='xkcd:dull blue', label="right_laser") # color='xkcd:deep sea blue
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(0, camera_max + camera_max/10)

    color = 'xkcd:brown'
    ax2.set_ylabel('pixels (pix)', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, avg_errors_top_left_camera, color='xkcd:dark orange', label="top_left_camera")
    ax2.plot(t, avg_errors_top_right_camera, color='xkcd:mud', label="top_right_camera")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc='upper right')
    plt.show()
