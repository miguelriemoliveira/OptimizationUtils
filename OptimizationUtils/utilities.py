#!/usr/bin/env python
"""
A set of utilities to be used in the optimization algorithms
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------

import numpy as np
import cv2


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

# ---------------------------------------
# --- Drawing functions
# ---------------------------------------

def drawSquare2D(image, x, y, size, color=(0, 0, 255), thickness=1):
    """
    Draws a square on the image
    :param image:
    :param x:
    :param y:
    :param color:
    :param thickness:
    """

    w, h, _ = image.shape
    if x - size < 0 or x + size > w or y - size < 0 or y + size > h:
        # print("Cannot draw square")
        return None

    # tl, tr, bl, br -> top left, top right, bottom left, bottom right
    tl = (int(x - size), int(y - size))
    tr = (int(x + size), int(y - size))
    br = (int(x + size), int(y + size))
    bl = (int(x - size), int(y + size))

    cv2.line(image, tl, tr, color, thickness)
    cv2.line(image, tr, br, color, thickness)
    cv2.line(image, br, bl, color, thickness)
    cv2.line(image, bl, tl, color, thickness)


def drawAxis3D(ax, transform, text, axis_scale=0.1, line_width=1.0, handles=None):
    """
    Draws (or replots) a 3D reference system
    :param ax:
    :param transform:
    :param text:
    :param axis_scale:
    :param line_width:
    :param hin: handles in
    """
    pt_origin = np.array([[0, 0, 0, 1]], dtype=np.float).transpose()
    x_axis = np.array([[0, 0, 0, 1], [axis_scale, 0, 0, 1]], dtype=np.float).transpose()
    y_axis = np.array([[0, 0, 0, 1], [0, axis_scale, 0, 1]], dtype=np.float).transpose()
    z_axis = np.array([[0, 0, 0, 1], [0, 0, axis_scale, 1]], dtype=np.float).transpose()

    pt_origin = np.dot(transform, pt_origin)
    x_axis = np.dot(transform, x_axis)
    y_axis = np.dot(transform, y_axis)
    z_axis = np.dot(transform, z_axis)

    if handles == None:
        handles_out = {}
        handles_out['x'] = ax.plot(x_axis[0, :], x_axis[1, :], x_axis[2, :], 'r-', linewidth=line_width)[0]
        handles_out['y'] = ax.plot(y_axis[0, :], y_axis[1, :], y_axis[2, :], 'g-', linewidth=line_width)[0]
        handles_out['z'] = ax.plot(z_axis[0, :], z_axis[1, :], z_axis[2, :], 'b-', linewidth=line_width)[0]
        handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black')
        return handles_out
    else:
        handles['x'].set_xdata(x_axis[0, :])
        handles['x'].set_ydata(x_axis[1, :])
        handles['x'].set_3d_properties(zs=x_axis[2, :])

        handles['y'].set_xdata(y_axis[0, :])
        handles['y'].set_ydata(y_axis[1, :])
        handles['y'].set_3d_properties(zs=y_axis[2, :])

        handles['z'].set_xdata(z_axis[0, :])
        handles['z'].set_ydata(z_axis[1, :])
        handles['z'].set_3d_properties(zs=z_axis[2, :])

        handles['text'].set_position((pt_origin[0, 0], pt_origin[1, 0]))
        handles['text'].set_3d_properties(z=pt_origin[2, 0], zdir='y')


def drawAxis3DOrigin(ax, transform, text, line_width=1.0, fontsize=12, handles=None):
    """
    Draws (or replots) a 3D Point
    :param ax:
    :param transform:
    :param text:
    :param line_width:
    :param fontsize:
    :param hin: handles in
    """
    pt_origin = np.array([[0, 0, 0, 1]], dtype=np.float).transpose()
    pt_origin = np.dot(transform, pt_origin)

    if handles is None:
        handles_out = {}
        print(pt_origin[2, 0])
        handles_out['point'] = ax.plot([pt_origin[0, 0], pt_origin[0, 0]], [pt_origin[1, 0], pt_origin[1, 0]],
                                       [pt_origin[2, 0], pt_origin[2, 0]], 'k.')[0]
        handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black',
                                      fontsize=fontsize)
        return handles_out
    else:
        handles['point'].set_xdata([pt_origin[0, 0], pt_origin[0, 0]])
        handles['point'].set_ydata([pt_origin[1, 0], pt_origin[1, 0]])
        handles['point'].set_3d_properties(zs=[pt_origin[2, 0], pt_origin[2, 0]])

        handles['text'].set_position((pt_origin[0, 0], pt_origin[1, 0]))
        handles['text'].set_3d_properties(z=pt_origin[2, 0], zdir='x')


# ---------------------------------------
# --- Geometry functions
# ---------------------------------------

def matrixToRodrigues(T):
    rods, _ = cv2.Rodrigues(T[0:3, 0:3])
    rods = rods.transpose()
    return rods[0]


def rodriguesToMatrix(r):
    rod = np.array(r, dtype=np.float)
    matrix = cv2.Rodrigues(rod)
    return matrix[0]


def traslationRodriguesToTransform(translation, rodrigues):
    R = rodriguesToMatrix(rodrigues)
    T = np.zeros((4, 4), dtype=np.float)
    T[0:3, 0:3] = R
    T[0:3, 3] = translation
    T[3, 3] = 1
    return T


