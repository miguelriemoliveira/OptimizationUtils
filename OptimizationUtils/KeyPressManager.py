#!/usr/bin/env python
"""
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------
def drawAxis3D(ax, transform, text, axis_scale=0.1, line_width=1.0):
    pt_origin = np.array([[0, 0, 0, 1]], dtype=np.float).transpose()
    x_axis = np.array([[0, 0, 0, 1], [axis_scale, 0, 0, 1]], dtype=np.float).transpose()
    y_axis = np.array([[0, 0, 0, 1], [0, axis_scale, 0, 1]], dtype=np.float).transpose()
    z_axis = np.array([[0, 0, 0, 1], [0, 0, axis_scale, 1]], dtype=np.float).transpose()

    pt_origin = np.dot(transform, pt_origin)
    x_axis = np.dot(transform, x_axis)
    y_axis = np.dot(transform, y_axis)
    z_axis = np.dot(transform, z_axis)

    ax.plot(x_axis[0, :], x_axis[1, :], x_axis[2, :], 'r-', linewidth=line_width)
    ax.plot(y_axis[0, :], y_axis[1, :], y_axis[2, :], 'g-', linewidth=line_width)
    ax.plot(z_axis[0, :], z_axis[1, :], z_axis[2, :], 'b-', linewidth=line_width)
    ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black')  # , size=15, zorder=1


# -------------------------------------------------------------------------------
# CLASS
# -------------------------------------------------------------------------------
class WindowManager:

    def __init__(self, figs=None):
        """
        :type fig: figure handle or list of figure handles
        """
        # Handle the argument fig as a figure handle or a list of figure handles
        if type(figs) is list:
            self.figs = figs
        else:
            self.figs = [figs]

        if not self.figs[0] is None:
            for fig in self.figs:
                fig.canvas.mpl_connect('key_press_event', self.mplKeyPressCallback)

    def mplKeyPressCallback(self, event):
        self.mpl_pressed_key = event.key

    def waitForKey(self, time_to_wait=None, verbose=True, message=None):
        """
        Waits for a key to be pressed
        :return: True if should abort program, false if not
        """
        t = time.time()
        if verbose:
            if message is None:
                message = 'keyPressManager: Press "c" to continue or "q" to abort.'
            print(message)

        self.mpl_pressed_key = None
        while True:
            key = cv2.waitKey(5)

            if not self.figs is None:
                for fig in self.figs:
                    fig.canvas.flush_events()
                    fig.canvas.draw()
                    plt.waitforbuttonpress(0.005)

            if key == ord('c') or self.mpl_pressed_key == 'c':
                print('Pressed "c". Continuing.')
                return False
            elif key == ord('q') or self.mpl_pressed_key == 'q':
                print('Pressed "q". Aborting.')
            elif key == ord('x') or self.mpl_pressed_key == 'x':
                print('Pressed "x". Returning with code x.')
                return 'x'

            if not time_to_wait is None:
                if (time.time() - t) > time_to_wait:
                    if verbose:
                        print('Time to wait elapsed. Returning.')
                    return False
