# geom.py ---
#
# Filename: geom.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Oct  5 14:53:24 2017 (+0200)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:

import numpy as np
from utils import loadh5
from transformations import quaternion_from_matrix

def parse_geom(geom):

    parsed_geom = {}
    parsed_geom["K"] = geom[:9].reshape((3, 3))
    parsed_geom["R"] = geom[9:18].reshape((3, 3))
    parsed_geom["t"] = geom[18:21].reshape((3, 1))
    parsed_geom["img_size"] = geom[21:23].reshape((2,))
    parsed_geom["K_inv"] = geom[23:32].reshape((3, 3))
    parsed_geom["q"] = geom[32:36].reshape([4, 1])
    parsed_geom["q_inv"] = geom[36:40].reshape([4, 1])

    return parsed_geom

def load_geom(geom_file, scale_factor=1.0, flip_R=False):
    # load geometry file
    geom_dict = loadh5(geom_file)
    # Check if principal point is at the center
    K = geom_dict["K"]
    # assert(abs(K[0, 2]) < 1e-3 and abs(K[1, 2]) < 1e-3)
    # Rescale calbration according to previous resizing
    S = np.asarray([[scale_factor, 0, 0],
                    [0, scale_factor, 0],
                    [0, 0, 1]])
    K = np.dot(S, K)
    geom_dict["K"] = K
    # Transpose Rotation Matrix if needed
    if flip_R:
        R = geom_dict["R"].T.copy()
        geom_dict["R"] = R
    # append things to list
    geom_list = []
    geom_info_name_list = ["K", "R", "T", "imsize"]
    for geom_info_name in geom_info_name_list:
        geom_list += [geom_dict[geom_info_name].flatten()]
    # Finally do K_inv since inverting K is tricky with theano
    geom_list += [np.linalg.inv(geom_dict["K"]).flatten()]
    # Get the quaternion from Rotation matrices as well
    q = quaternion_from_matrix(geom_dict["R"])
    geom_list += [q.flatten()]
    # Also add the inverse of the quaternion
    q_inv = q.copy()
    np.negative(q_inv[1:], q_inv[1:])
    geom_list += [q_inv.flatten()]
    # Add to list
    geom = np.concatenate(geom_list)
    return geom

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M


def np_unskew_symmetric(M):

    v = np.concatenate([
        0.5 * (M[:, 7] - M[:, 5])[None],
        0.5 * (M[:, 2] - M[:, 6])[None],
        0.5 * (M[:, 3] - M[:, 1])[None],
    ], axis=1)

    return v


def get_episqr(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()

    ys = x2Fx1**2

    return ys.flatten()


def get_episym(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 * (
        1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) +
        1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()


def get_sampsons(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 / (
        Fx1[..., 0]**2 + Fx1[..., 1]**2 + Ftx2[..., 0]**2 + Ftx2[..., 1]**2
    )

    return ys.flatten()


#
# geom.py ends here
