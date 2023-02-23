"""
Adapted from:
https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018
"""

import numpy as np
import tensorflow as tf

#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)


Aux_M2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)


Aux_M3 = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1]], dtype=np.float64)


Aux_M4 = np.array([
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)


Aux_M5 = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)


Aux_M6 = np.array([
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0]], dtype=np.float64)


Aux_M71 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)


Aux_M72 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, -1, 0]], dtype=np.float64)


Aux_M8 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, -1]], dtype=np.float64)


Aux_Mb = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)


def solve_DLT(pred_h4p, training, constrained=True, scale=None):
    bs = tf.shape(pred_h4p)[0]
    if training:
        h = pred_h4p.get_shape()[1].value
        w = pred_h4p.get_shape()[2].value
    else:
        h = tf.shape(pred_h4p)[1]
        w = tf.shape(pred_h4p)[2]

    pts_1 = tf.constant([-1, -1, 1, -1, -1, 1, 1, 1], dtype=tf.float32)
    pts_1 = tf.reshape(pts_1, (1, 1, 1, 8, 1))
    pts_1 = tf.tile(pts_1, (bs, h, w, 1, 1))
    pts_2 = pred_h4p[..., None] + pts_1

    # Auxiliary tensors used to create Ax = b equation
    M1 = tf.constant(Aux_M1, tf.float32)
    M1_tensor = tf.reshape(M1, [1, 1, 1, 8, 8])
    M1_tile = tf.tile(M1_tensor, [bs, h, w, 1, 1])

    M2 = tf.constant(Aux_M2, tf.float32)
    M2_tensor = tf.reshape(M2, [1, 1, 1, 8, 8])
    M2_tile = tf.tile(M2_tensor, [bs, h, w, 1, 1])

    M3 = tf.constant(Aux_M3, tf.float32)
    M3_tensor = tf.reshape(M3, [1, 1, 1, 8, 1])
    M3_tile = tf.tile(M3_tensor, [bs, h, w, 1, 1])

    M4 = tf.constant(Aux_M4, tf.float32)
    M4_tensor = tf.reshape(M4, [1, 1, 1, 8, 8])
    M4_tile = tf.tile(M4_tensor, [bs, h, w, 1, 1])

    M5 = tf.constant(Aux_M5, tf.float32)
    M5_tensor = tf.reshape(M5, [1, 1, 1, 8, 8])
    M5_tile = tf.tile(M5_tensor, [bs, h, w, 1, 1])

    M6 = tf.constant(Aux_M6, tf.float32)
    M6_tensor = tf.reshape(M6, [1, 1, 1, 8, 1])
    M6_tile = tf.tile(M6_tensor, [bs, h, w, 1, 1])

    M71 = tf.constant(Aux_M71, tf.float32)
    M71_tensor = tf.reshape(M71, [1, 1, 1, 8, 8])
    M71_tile = tf.tile(M71_tensor, [bs, h, w, 1, 1])

    M72 = tf.constant(Aux_M72, tf.float32)
    M72_tensor = tf.reshape(M72, [1, 1, 1, 8, 8])
    M72_tile = tf.tile(M72_tensor, [bs, h, w, 1, 1])

    M8 = tf.constant(Aux_M8, tf.float32)
    M8_tensor = tf.reshape(M8, [1, 1, 1, 8, 8])
    M8_tile = tf.tile(M8_tensor, [bs, h, w, 1, 1])

    Mb = tf.constant(Aux_Mb, tf.float32)
    Mb_tensor = tf.reshape(Mb, [1, 1, 1, 8, 8])
    Mb_tile = tf.tile(Mb_tensor, [bs, h, w, 1, 1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = tf.matmul(M1_tile, pts_1)  # Column 1
    A2 = tf.matmul(M2_tile, pts_1)  # Column 2
    A3 = M3_tile                   # Column 3
    A4 = tf.matmul(M4_tile, pts_1)  # Column 4
    A5 = tf.matmul(M5_tile, pts_1)  # Column 5
    A6 = M6_tile                   # Column 6
    A7 = tf.matmul(M71_tile, pts_2) * tf.matmul(M72_tile, pts_1)  # Column 7
    A8 = tf.matmul(M71_tile, pts_2) * tf.matmul(M8_tile, pts_1)  # Column 8

    if constrained:
        A_mat = tf.concat([A1, A2, A4, A5, A7, A8], axis=-1)
    else:
        A_mat = tf.concat([A1, A2, A3, A4, A5, A6, A7, A8], axis=-1)
    # Form b matrix
    b_mat = tf.matmul(Mb_tile, pts_2)

    # Solve the Ax = b
    if constrained:
        A_t_mat = tf.matrix_transpose(A_mat)
        A_mat = tf.matmul(A_t_mat, A_mat)
        b_mat = tf.matmul(A_t_mat, b_mat)
        H_6el = tf.matrix_solve(A_mat, b_mat)
        H_6el = tf.squeeze(H_6el, axis=-1)

        if scale is not None:
            H_4el = H_6el[:, :, :, 0:4]
            H_4el = H_4el * scale
            H_2el = H_6el[:, :, :, 4:6]
            H_6el = tf.concat([H_4el, H_2el], axis=-1)

        h_zeros = tf.zeros([bs, h, w, 1])
        h_ones = tf.ones([bs, h, w, 1])
        h3 = tf.expand_dims(tf.concat([h_zeros, h_zeros, h_ones], axis=-1), axis=-1)
        H_6el = tf.reshape(H_6el, [bs, h, w, 3, 2])   # BATCH_SIZE x 3 x 3
        H_mat = tf.concat([H_6el, h3], axis=-1)
    else:
        H_8el = tf.matrix_solve(A_mat, b_mat)  # BATCH_SIZE x 8.
        H_8el = tf.squeeze(H_8el, axis=-1)

        if scale is not None:
            H_6el = H_8el[:, :, :, 0:6]
            H_6el = H_6el * scale
            H_2el = H_8el[:, :, :, 6:8]
            H_8el = tf.concat([H_6el, H_2el], axis=-1)

        h_ones = tf.ones([bs, h, w, 1])
        H_9el = tf.concat([H_8el, h_ones], -1)
        H_mat = tf.reshape(H_9el, [bs, h, w, 3, 3])   # BATCH_SIZE x 3 x 3

    has_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(H_mat), tf.float32))
    H_mat = tf.cond(
        tf.equal(has_nan, 0),
        lambda: H_mat,
        lambda: tf.tile(tf.reshape(tf.eye(3), [1, 1, 1, 3, 3]), [bs, h, w, 1, 1])
    )
    
    return H_mat


if __name__ == "__main__":
    pred_h4p = np.array([0.5, -0.3, 0.15, 0.9, -1, -0.2, 0.5, 1.2])
    pred_h4p = tf.constant(pred_h4p, dtype=tf.float32)
    pred_h4p = tf.reshape(pred_h4p, (1, 1, 1, 8))
    H_mat = solve_DLT(pred_h4p, True, False)

    rng = tf.range(-1, 2)
    x, y = tf.meshgrid(rng, rng)
    x = tf.reshape(x, (-1, ))
    y = tf.reshape(y, (-1, ))
    xy = tf.reshape(tf.stack([x, y], axis=-1), [1, 1, 1, -1, 2])
    xy = tf.tile(xy, [tf.shape(H_mat)[0], tf.shape(
        H_mat)[1], tf.shape(H_mat)[2], 1, 1])
    xy = tf.cast(xy, tf.float32)
    ones = tf.ones_like(xy[:, :, :, :, 0])[..., None]
    xy_homo = tf.concat([xy, ones], axis=-1)

    pert_xy = tf.matmul(xy_homo, H_mat, transpose_b=True)
    homo_scale = tf.expand_dims(pert_xy[:, :, :, :, -1], axis=-1)
    pert_xy = pert_xy[:, :, :, :, 0:2]
    pert_xy = tf.clip_by_value(tf.math.divide_no_nan(pert_xy, homo_scale), -10., 10.)

    with tf.Session() as sess:
        print(sess.run(pert_xy))
