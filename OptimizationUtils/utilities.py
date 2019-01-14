#!/usr/bin/env python
"""
A set of utilities to be used in the optimization algorithms
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
from copy import deepcopy

import KeyPressManager
import numpy as np
import cv2


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

# ---------------------------------------
# --- Drawing functions
# ---------------------------------------
from numpy.linalg import norm


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


# ---------------------------------------
# --- Computer Vision functions
# ---------------------------------------
def projectToCameraPair(intrinsic_matrix_a, distortion_a, width_a, height_a, map_T_cam_a, image_a, depth_a,
                        intrinsic_matrix_b, distortion_b, width_b, height_b, map_T_cam_b, image_b, depth_b,
                        pts3D_in_map, z_inconsistency_threshold = 0.1, visualize=False):
    # type: (object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object) -> (np.array, np.array, np.array)

    # project 3D points to cam_a image (first the transformation from map to camera is done)
    pts3D_in_cam_a = np.dot(map_T_cam_a, pts3D_in_map)
    pts2D_a, pts_valid_a, pts_range_a = projectToCamera(
        np.array(intrinsic_matrix_a).reshape((3, 3)),
        distortion_a,
        width_a,
        height_a,
        pts3D_in_cam_a)

    pts2D_a = np.where(pts_valid_a, pts2D_a, 0)
    range_meas_a = depth_a[pts2D_a[1, :], pts2D_a[0, :]]
    z_valid_a = abs(pts_range_a - range_meas_a) < z_inconsistency_threshold

    # project 3D points to cam_b
    pts3D_in_cam_b = np.dot(map_T_cam_b, pts3D_in_map)
    pts2D_b, pts_valid_b, pts_range_b = projectToCamera(
        np.array(intrinsic_matrix_b).reshape((3, 3)),
        distortion_b,
        width_b,
        height_b,
        pts3D_in_cam_b)

    pts2D_b = np.where(pts_valid_b, pts2D_b, 0)
    range_meas_b = depth_b[pts2D_b[1, :], pts2D_b[0, :]]
    z_valid_b = abs(pts_range_b - range_meas_b) < z_inconsistency_threshold

    # Compute masks for the valid projections

    mask = np.logical_and(pts_valid_a, pts_valid_b)
    z_mask = np.logical_and(z_valid_a, z_valid_b)
    final_mask = np.logical_and(mask, z_mask)

    if visualize:
        print("pts2d_a has " + str(np.count_nonzero(pts_valid_a)) + ' valid projections')
        print("pts2d_b has " + str(np.count_nonzero(pts_valid_b)) + ' valid projections')
        print("there are " + str(np.count_nonzero(mask)) + ' valid projection pairs')

        cam_a_image = deepcopy(image_a)
        cam_b_image = deepcopy(image_b)
        for i, val in enumerate(mask):
            if pts_valid_a[i] == True:
                x0 = pts2D_a[0, i]
                y0 = pts2D_a[1, i]
                cv2.line(cam_a_image, (x0, y0), (x0, y0), color=(80, 80, 80), thickness=2)

            if pts_valid_b[i] == True:
                x0 = pts2D_b[0, i]
                y0 = pts2D_b[1, i]
                cv2.line(cam_b_image, (x0, y0), (x0, y0), color=(80, 80, 80), thickness=2)

            if val == True:
                x0 = pts2D_a[0, i]
                y0 = pts2D_a[1, i]
                cv2.line(cam_a_image, (x0, y0), (x0, y0), color=(0, 0, 210), thickness=2)

                x0 = pts2D_b[0, i]
                y0 = pts2D_b[1, i]
                cv2.line(cam_b_image, (x0, y0), (x0, y0), color=(0, 0, 210), thickness=2)

            if z_mask[i] == True:
                x0 = pts2D_a[0, i]
                y0 = pts2D_a[1, i]
                cv2.line(cam_a_image, (x0, y0), (x0, y0), color=(0, 210, 0), thickness=2)

                x0 = pts2D_b[0, i]
                y0 = pts2D_b[1, i]
                cv2.line(cam_b_image, (x0, y0), (x0, y0), color=(0, 210, 0), thickness=2)

        cv2.namedWindow('cam_a', cv2.WINDOW_NORMAL)
        cv2.imshow('cam_a', cam_a_image)
        cv2.namedWindow('cam_b', cv2.WINDOW_NORMAL)
        cv2.imshow('cam_b', cam_b_image)

        wm = KeyPressManager.KeyPressManager.WindowManager()
        if wm.waitForKey(time_to_wait=None, verbose=False):
            exit(0)

    return pts2D_a, pts2D_b, final_mask

def projectToCamera(intrinsic_matrix, distortion, width, height, pts):
    """
    Projects a list of points to the camera defined transform, intrinsics and distortion
    :param intrinsic_matrix: 3x3 intrinsic camera matrix
    :param distortion: should be as follows: (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])
    :param width: the image width
    :param height: the image height
    :param pts: a list of point coordinates (in the camera frame) with the following format
    :return: a list of pixel coordinates with the same lenght as pts
    """

    _, n_pts = pts.shape

    # Project the 3D points in the camera's frame to image pixels
    # From https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    pixs = np.zeros((2, n_pts), dtype=np.int)

    k1, k2, p1, p2, k3 = distortion
    # fx, _, cx, _, fy, cy, _, _, _ = intrinsic_matrix
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    x = pts[0, :]
    y = pts[1, :]
    z = pts[2, :]

    dists = norm(pts[0:3, :], axis=0)  # compute distances from point to camera
    xl = np.divide(x, z)  # compute homogeneous coordinates
    yl = np.divide(y, z)  # compute homogeneous coordinates
    r2 = xl ** 2 + yl ** 2  # r square (used multiple times bellow)
    xll = xl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + 2 * p1 * xl * yl + p2 * (r2 + 2 * xl ** 2)
    yll = yl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + p1 * (r2 + 2 * yl ** 2) + 2 * p2 * xl * yl
    pixs[0, :] = fx * xll + cx
    pixs[1, :] = fy * yll + cy

    # Compute mask of valid projections
    valid_z = z > 0
    valid_xpix = np.logical_and(pixs[0, :] >= 0, pixs[0, :] < width)
    valid_ypix = np.logical_and(pixs[1, :] >= 0, pixs[1, :] < height)
    valid_pixs = np.logical_and(valid_z, np.logical_and(valid_xpix, valid_ypix))
    return pixs, valid_pixs, dists


def addSafe(i_in, val):
    """Avoids saturation when adding to uint8 images"""
    i_out = i_in.astype(np.float)  # Convert the i to type float
    i_out = np.add(i_out, val)  # Perform the adding of parameters to the i
    i_out = np.maximum(i_out, 0)  # underflow
    i_out = np.minimum(i_out, 255)  # overflow
    return i_out.astype(np.uint8)  # Convert back to uint8 and return


def printNumPyArray(arrays):
    """

    :type arrays: [{name, array}]
    """

    for name in arrays:
        array = arrays[name]
        print(name + ': shape ' + str(array.shape) + ' type ' + str(array.dtype) + ':\n' + str(array))




