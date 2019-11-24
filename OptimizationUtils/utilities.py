#!/usr/bin/env python
"""
A set of utilities to be used in the optimization algorithms
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
from copy import deepcopy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tf import transformations

import KeyPressManager
import numpy as np
import cv2
from matplotlib import cm

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

# ---------------------------------------
# --- Drawing functions
# ---------------------------------------
from numpy.linalg import norm


def drawCross2D(image, x, y, size, color=(0, 0, 255), thickness=1):
    """
    Draws a square on the image
    :param image:
    :param x:
    :param y:
    :param color:
    :param thickness:
    """

    h, w, _ = image.shape
    if x - size < 0 or x + size > w or y - size < 0 or y + size > h:
        # print("Cannot draw square")
        return None

    # tl, tr, bl, br -> top left, top right, bottom left, bottom right
    left = (int(x - size), int(y))
    right = (int(x + size), int(y))
    top = (int(x), int(y - size))
    bottom = (int(x), int(y + size))

    cv2.line(image, left, right, color, thickness)
    cv2.line(image, top, bottom, color, thickness)


def drawSquare2D(image, x, y, size, color=(0, 0, 255), thickness=1):
    """
    Draws a square on the image
    :param image:
    :param x:
    :param y:
    :param color:
    :param thickness:
    """

    h, w, _ = image.shape
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


def drawPoints3D(ax, transform, pts, color=[0, 0, 0], marker_size=1.0, line_width=1.0, marker='.', mfc=None,
                 text=None, text_color=[0, 0, 0], sensor_color = [0,0,0], handles=None):
    """
    Draws (or replots) a 3D reference system
    :param handles:
    :param mfc:
    :param marker:
    :param ax:
    :param transform:
    :param pts:
    :param color:
    :param line_width:
    """
    if mfc is None:
        mfc = color

    mfc = list(mfc[0:3])  # remove alpha channel and convert to list

    if not transform is None:
        pts = np.dot(transform, pts)

    center_pt = np.average(pts, axis=1)
    limit_pts = pts[:, [0, pts.shape[1]-1]]
    # limit_pts = pts[:, [-1]]

    if handles is None:
        handles_out = {}
        handles_out['pts'] = ax.plot(pts[0, :], pts[1, :], pts[2, :], marker, color=color, markersize=marker_size,
                                     linewidth=line_width)[0]

        handles_out['pts_limits'] = ax.plot(limit_pts[0, :], limit_pts[1, :], limit_pts[2, :], 'o', color=sensor_color, markersize=5,
                                     linewidth=line_width*2, mfc='none')[0]
        if not text is None:
            handles_out['text'] = ax.text(center_pt[0], center_pt[1], center_pt[2], text, color=text_color)
        return handles_out
    else:
        handles['pts'].set_xdata(pts[0, :])
        handles['pts'].set_ydata(pts[1, :])
        handles['pts'].set_3d_properties(zs=pts[2, :])

        handles['pts_limits'].set_xdata(limit_pts[0, :])
        handles['pts_limits'].set_ydata(limit_pts[1, :])
        handles['pts_limits'].set_3d_properties(limit_pts[2, :])

        if not text is None:
            handles['text'].set_position((center_pt[0], center_pt[1]))
            handles['text'].set_3d_properchessboard_pointsties(z=center_pt[2], zdir='y')


def drawChessBoard(ax, transform, pts, text, chess_num_x, chess_num_y, color='black', axis_scale=0.1, line_width=1.0,
                   handles=None):
    """
    Draws (or replots) a 3D reference system
    :param ax:
    :param transform:
    :param line_width:
    :param handles:
    """

    pt_origin = np.array([[0, 0, 0, 1]], dtype=np.float).transpose()
    x_axis = np.array([[0, 0, 0, 1], [axis_scale, 0, 0, 1]], dtype=np.float).transpose()
    y_axis = np.array([[0, 0, 0, 1], [0, axis_scale, 0, 1]], dtype=np.float).transpose()
    z_axis = np.array([[0, 0, 0, 1], [0, 0, axis_scale, 1]], dtype=np.float).transpose()

    pt_origin = np.dot(transform, pt_origin)
    x_axis = np.dot(transform, x_axis)
    y_axis = np.dot(transform, y_axis)
    z_axis = np.dot(transform, z_axis)

    pts = np.dot(transform, pts)
    # idxs = [0, chess_num_x - 1, chess_num_x * chess_num_y - 1, chess_num_x * (chess_num_y - 1), 0]
    # corners = pts[:, idxs]

    if handles is None:
        handles_out = {}

        counter = 0
        for idx_y in range(0, chess_num_y):
            idxs_x = [idx_y * chess_num_x, idx_y * chess_num_x + chess_num_x - 1]
            handles_out['chessboard_pts_' + str(counter)] = \
            ax.plot(pts[0, idxs_x], pts[1, idxs_x], pts[2, idxs_x], '-', linewidth=1.0, color=color)[0]
            counter += 1

        for idx_x in range(0, chess_num_x):
            idxs_y = [idx_x, chess_num_x * chess_num_y - chess_num_x + idx_x]
            handles_out['chessboard_pts_' + str(counter)] = \
            ax.plot(pts[0, idxs_y], pts[1, idxs_y], pts[2, idxs_y], '-', linewidth=1.0, color=color)[0]
            counter += 1

        handles_out['x'] = ax.plot(x_axis[0, :], x_axis[1, :], x_axis[2, :], 'r-', linewidth=line_width)[0]
        handles_out['y'] = ax.plot(y_axis[0, :], y_axis[1, :], y_axis[2, :], 'g-', linewidth=line_width)[0]
        handles_out['z'] = ax.plot(z_axis[0, :], z_axis[1, :], z_axis[2, :], 'b-', linewidth=line_width)[0]
        handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black')

        # x = list(corners[0,:])
        # y = list(corners[1,:])
        # z = list(corners[2,:])
        # verts = [zip(x, y, z)]
        # handles_out['fill'] = ax.add_collection3d(Poly3DCollection(verts,facecolors=[.5, .5, .5], linewidths=1, alpha=0.1 ))
        # # handles_out['fill'] =

        return handles_out
    else:

        counter = 0
        for idx_y in range(0, chess_num_y):
            idxs_x = [idx_y * chess_num_x, idx_y * chess_num_x + chess_num_x - 1]
            handles['chessboard_pts_' + str(counter)].set_xdata(pts[0, idxs_x])
            handles['chessboard_pts_' + str(counter)].set_ydata(pts[1, idxs_x])
            handles['chessboard_pts_' + str(counter)].set_3d_properties(zs=pts[2, idxs_x])
            counter += 1

        for idx_x in range(0, chess_num_x):
            idxs_y = [idx_x, chess_num_x * chess_num_y - chess_num_x + idx_x]
            handles['chessboard_pts_' + str(counter)].set_xdata(pts[0, idxs_y])
            handles['chessboard_pts_' + str(counter)].set_ydata(pts[1, idxs_y])
            handles['chessboard_pts_' + str(counter)].set_3d_properties(zs=pts[2, idxs_y])
            counter += 1

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


def drawAxis3D_plotly(ax, transform, text, text_color = [0,0,0], axis_scale=0.1, line_width=1.0, handles=None):
    """
    Draws (or replots) a 3D reference system
    :param text_color:
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
        handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color=text_color, fontsize=10)
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

def drawAxis3D(ax, transform, text, text_color = [0,0,0], axis_scale=0.1, line_width=1.0, handles=None):
    """
    Draws (or replots) a 3D reference system
    :param text_color:
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
        handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color=text_color, fontsize=10)
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
    T[0, 3] = translation[0]
    T[1, 3] = translation[1]
    T[2, 3] = translation[2]
    T[3, 3] = 1
    return T


def translationQuaternionToTransform(trans, quat):
    matrix = transformations.quaternion_matrix(quat)
    matrix[0, 3] = trans[0]
    matrix[1, 3] = trans[1]
    matrix[2, 3] = trans[2]
    matrix[3, 3] = 1
    # print(str(matrix))
    return matrix


def getAggregateTransform(chain, transforms, mode=0):
    AT = np.eye(4, dtype=np.float)

    if mode == 0:
        for link in chain:
            parent = link['parent']
            child = link['child']
            trans = transforms[link['key']]['trans']
            quat = transforms[link['key']]['quat']
            parent_T_child = translationQuaternionToTransform(trans, quat)
            # print(parent + '_T_' + child + ' =\n' + str(parent_T_child))
            AT = np.dot(AT, parent_T_child)
        # print(parent + '_T_' + child + ' =\n' + str(AT))
    else:
        for link in chain[::-1]:
            parent = link['parent']
            child = link['child']
            trans = transforms[link['key']]['trans']
            quat = transforms[link['key']]['quat']
            parent_T_child = translationQuaternionToTransform(trans, quat)

            # print(parent + '_T_' + child + ' =\n' + str(parent_T_child))
            # AT = np.dot(AT, parent_T_child)
            AT = np.dot(parent_T_child, AT)
            # AT = np.dot(AT, parent_T_child)

    return AT


# ---------------------------------------
# --- Computer Vision functions
# ---------------------------------------
def projectToCameraPair(cam_a, cam_b, pts3D_in_map, z_inconsistency_threshold=0.1, visualize=False):
    # project 3D points to cam_a image (first the transformation from map to camera is done)
    pts3D_in_cam_a = cam_a.rgb.transformToCamera(pts3D_in_map)
    pts2D_a, pts_valid_a, pts_range_a = cam_a.rgb.projectToCamera(pts3D_in_cam_a)

    pts2D_a = np.where(pts_valid_a, pts2D_a, 0)
    range_meas_a = cam_a.rgb.range_dense[(pts2D_a[1, :]).astype(np.int), (pts2D_a[0, :]).astype(np.int)]
    z_valid_a = abs(pts_range_a - range_meas_a) < z_inconsistency_threshold

    # project 3D points to cam_b
    pts3D_in_cam_b = cam_b.rgb.transformToCamera(pts3D_in_map)
    pts2D_b, pts_valid_b, pts_range_b = cam_b.rgb.projectToCamera(pts3D_in_cam_b)

    pts2D_b = np.where(pts_valid_b, pts2D_b, 0)
    range_meas_b = cam_b.rgb.range_dense[(pts2D_b[1, :]).astype(np.int), (pts2D_b[0, :]).astype(np.int)]
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
    pixs = np.zeros((2, n_pts), dtype=np.float)

    k1, k2, p1, p2, k3 = distortion
    # fx, _, cx, _, fy, cy, _, _, _ = intrinsic_matrix
    # print('intrinsic=\n' + str(intrinsic_matrix))
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


def projectWithoutDistorcion(intrinsic_matrix, width, height, pts):
    """
    Projects a list of points to the camera defined transform, intrinsics and distortion
    :param intrinsic_matrix: 3x3 intrinsic camera matrix
    :param width: the image width
    :param height: the image height
    :param pts: a list of point coordinates (in the camera frame) with the following format
    :return: a list of pixel coordinates with the same lenght as pts
    """

    _, n_pts = pts.shape

    # Project the 3D points in the camera's frame to image pixels without considering the distorcion
    # From https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    pixs = np.zeros((2, n_pts), dtype=np.float)

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

    pixs[0, :] = fx * xl + cx
    pixs[1, :] = fy * yl + cy

    # Compute mask of valid projections
    valid_z = z > 0
    valid_xpix = np.logical_and(pixs[0, :] >= 0, pixs[0, :] < width)
    valid_ypix = np.logical_and(pixs[1, :] >= 0, pixs[1, :] < height)
    valid_pixs = np.logical_and(valid_z, np.logical_and(valid_xpix, valid_ypix))
    return pixs, valid_pixs, dists


def addSafe(image_in, val):
    """Avoids saturation when adding to uint8 images

    :param image_in: the image.
    :param val:
    :return:
    """
    image_out = image_in.astype(np.float)  # Convert the i to type float
    image_out = np.add(image_out, val)  # Perform the adding of parameters to the i
    image_out = np.maximum(image_out, 0)  # underflow
    image_out = np.minimum(image_out, 255)  # overflow
    return image_out.astype(np.uint8)  # Convert back to uint8 and return


def deVignetting(image, parameters):
    """ Devignettes and image using a devignetting function defined by a nth order polynomial.

    :param image: the input image to be devignetted.
    :param parameters: a list of n parameters to build the polynomial from
    """

    # f(x) = p1*x^5 + p2*x^4 + p3*x^3 + p4*x^2 + p5*x + p6


def adjustGamma(image, gamma=1.0):
    # type: (np.ndarray, float) -> np.ndarray
    """ Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values

    :param image: image to be altered.
    :param gamma: gamma value to use.
    :return: Altered image.
    """
    #
    if type(gamma) is list:
        gamma = gamma[0]

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def adjustLAB(image_in, l_bias=0.0, a_bias=0.0, b_bias=0.0, l_scale=1.0, a_scale=1.0, b_scale=1.0):
    image = image_in.copy()

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float)

    image_lab[:, :, 0] += l_bias * 255
    image_lab[:, :, 0] = image_lab[:, :, 0] * l_scale
    image_lab

    image_lab[:, :, 1] += a_bias * 255
    image_lab[:, :, 1] = image_lab[:, :, 1] * a_scale

    image_lab[:, :, 2] += b_bias * 255
    image_lab[:, :, 2] = image_lab[:, :, 1] * b_scale

    image_lab = np.maximum(image_lab, 0)  # underflow
    image_lab = np.minimum(image_lab, 255)  # overflow
    image_lab = image_lab.astype(np.uint8)

    image_out = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)

    return image_out
    # return np.uint8(rgb * 255.0)


def printNumPyArray(arrays):
    for name in arrays:
        array = arrays[name]
        print(name + ': shape ' + str(array.shape) + ' type ' + str(array.dtype) + ':\n' + str(array))


def drawProjectionErrors(img1, points1, img2, points2, errors, fig_name, skip=1):
    """ Draws an image pair reprojections similar to the opencv draw matches

    :param img1: An openCV image nd array in a grayscale or color format.
    :param points1: np array with shape (2, n)
    :param img2: An openCV image ndarray in a grayscale or color format.
    :param points2: np array with shape (2, n)
    :param errors: np array with shape (1, n)
    :param fig_name: string
    :return:
    """

    # We're drawing the images side by side. Get dimensions accordingly.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    if len(points1) == 0 or len(points1) != len(points2):  # nothing to draw
        cv2.imshow(fig_name, new_img)
        return

    r = 15
    thickness = 2

    for i, (x1, y1, x2, y2) in enumerate(
            zip(points1[0, ::skip], points1[1, ::skip], points2[0, ::skip], points2[1, ::skip])):
        x2 = x2 + img1.shape[1]
        idx = 255 - int(min(errors[i], 255))
        color = cm.RdYlGn(idx)
        c = round(color[2] * 255), round(color[1] * 255), round(color[0] * 255)  # matplot is RGB, opencv is BGR
        cv2.line(new_img, (x1, y1), (x2, y2), c, thickness)
        cv2.circle(new_img, (x1, y1), r, c, thickness)
        cv2.circle(new_img, (x2, y2), r, c, thickness)

    cv2.imshow(fig_name, new_img)


# https://gist.github.com/isker/11be0c50c4f78cad9549
def drawMatches(img1, kp1, img2, kp2, matches, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.

    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.

    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)

    plt.figure(figsize=(15, 15))
    plt.imshow(new_img)
    plt.show()