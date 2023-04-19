from .network import Network

import tensorflow as tf


class ASLFeatNet(Network):
    def setup(self):
        det_config = self.extra_args["det_config"]
        deform_desc = det_config["deform_desc"]

        (
            self.feed("data")
            .conv_bn(3, 32, 1, name="conv0")
            .conv(3, 32, 1, biased=False, relu=False, name="conv1")
            .batch_normalization(relu=True, name="conv1/bn")
            .conv_bn(3, 64, 2, name="conv2")
            .conv(3, 64, 1, biased=False, relu=False, name="conv3")
            .batch_normalization(relu=True, name="conv3/bn")
            .conv_bn(3, 128, 2, name="conv4")
            .conv_bn(3, 128, 1, name="conv5")
        )

        if deform_desc > 0:
            if deform_desc == 1:
                deform_type = "u"
            elif deform_desc == 2:
                deform_type = "a"
            elif deform_desc == 3:
                deform_type = "h"
            else:
                raise NotImplementedError

            (
                self.feed("conv5")
                .deform_conv_bn(3, 128, 1, deform_type=deform_type, name="conv6_0")
                .deform_conv_bn(3, 128, 1, deform_type=deform_type, name="conv6_1")
                .deform_conv(3, 128, 1, biased=False, relu=False, deform_type=deform_type, name="conv6")
            )
        else:
            (
                self.feed("conv5")
                .conv_bn(3, 128, 1, name="conv6_0")
                .conv_bn(3, 128, 1, name="conv6_1")
                .conv(3, 128, 1, biased=False, relu=False, name="conv6")
            )

        dense_feat_map = self.layers["conv6"]

        kpt_n = det_config["kpt_n"]

        if det_config["multi_level"]:
            comb_names = ["conv1", "conv3", "conv6"]
            comb_weights = tf.constant([1, 2, 3], dtype=tf.float32)
            comb_weights /= tf.reduce_sum(comb_weights)
            scale = [3, 2, 1]
        else:
            comb_names = ["conv6"]
            comb_weights = tf.constant([1], dtype=tf.float32)
            scale = [1]

        ori_h = tf.shape(self.inputs["data"])[1]
        ori_w = tf.shape(self.inputs["data"])[2]

        comb_score_map = None

        for idx, tmp_name in enumerate(comb_names):
            tmp_feat_map = self.layers[tmp_name]
            prep_dense_feat_map = tmp_feat_map

            if det_config["use_peakiness"]:
                alpha, beta = self.peakiness_score(
                    prep_dense_feat_map, ksize=3, need_norm=det_config["need_norm"], dilation=scale[idx], name=tmp_name
                )
            else:
                alpha, beta = self.d2net_score(
                    prep_dense_feat_map, ksize=3, need_norm=det_config["need_norm"], dilation=scale[idx], name=tmp_name
                )

            score_vol = alpha * beta
            score_map = tf.reduce_max(score_vol, axis=-1, keepdims=True)
            score_map = tf.image.resize(score_map, (ori_h, ori_w))
            tmp_comb_weights = comb_weights[idx] * score_map

            if comb_score_map is None:
                comb_score_map = tmp_comb_weights
            else:
                comb_score_map += tmp_comb_weights

        score_map = comb_score_map

        kpt_inds, kpt_score = self.extract_kpts(
            score_map,
            k=kpt_n,
            score_thld=det_config["score_thld"],
            edge_thld=det_config["edge_thld"],
            nms_size=det_config["nms_size"],
            eof_size=det_config["eof_mask"],
        )

        if det_config["kpt_refinement"]:
            offsets = tf.squeeze(self.kpt_refinement(score_map), axis=-2)
            offsets = tf.gather_nd(offsets, kpt_inds, batch_dims=1)
            offsets = tf.clip_by_value(offsets, -0.5, 0.5)
            kpt_inds = tf.cast(kpt_inds, tf.float32) + offsets
        else:
            kpt_inds = tf.cast(kpt_inds, tf.float32)

        self.endpoints["descs"] = tf.nn.l2_normalize(interpolate(kpt_inds / 4, dense_feat_map), axis=-1, name="descs")
        self.endpoints["kpts"] = tf.stack([kpt_inds[:, :, 1], kpt_inds[:, :, 0]], axis=-1, name="kpts")
        self.endpoints["scores"] = tf.identity(kpt_score, name="scores")

    def peakiness_score(self, inputs, ksize=3, need_norm=True, dilation=1, name="conv"):
        if need_norm:
            from tensorflow.python.training.moving_averages import assign_moving_average

            with tf.compat.v1.variable_scope("tower", reuse=self.reuse):
                moving_instance_max = tf.compat.v1.get_variable(
                    "%s/instance_max" % name, (), initializer=tf.constant_initializer(1), trainable=False
                )
            decay = 0.99

            if self.training:
                instance_max = tf.reduce_max(inputs)
                with tf.control_dependencies([assign_moving_average(moving_instance_max, instance_max, decay)]):
                    inputs = inputs / moving_instance_max
            else:
                inputs = inputs / moving_instance_max

        pad_inputs = tf.pad(inputs, [[0, 0], [dilation, dilation], [dilation, dilation], [0, 0]], mode="REFLECT")
        avg_inputs = tf.nn.pool(pad_inputs, [ksize, ksize], "AVG", padding="VALID", dilations=[dilation, dilation])
        alpha = tf.math.softplus(inputs - avg_inputs)
        beta = tf.math.softplus(inputs - tf.reduce_mean(inputs, axis=-1, keepdims=True))
        return alpha, beta

    def d2net_score(self, inputs, ksize=3, need_norm=True, dilation=1, name="conv"):
        channel_wise_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
        beta = inputs / (channel_wise_max + 1e-6)

        if need_norm:
            from tensorflow.python.training.moving_averages import assign_moving_average

            with tf.compat.v1.variable_scope("tower", reuse=self.reuse):
                moving_instance_max = tf.compat.v1.get_variable(
                    "%s/instance_max" % name, (), initializer=tf.constant_initializer(1), trainable=False
                )
            decay = 0.99

            if self.training:
                instance_max = tf.reduce_max(inputs)
                with tf.control_dependencies([assign_moving_average(moving_instance_max, instance_max, decay)]):
                    exp_logit = tf.exp(inputs / moving_instance_max)
            else:
                exp_logit = tf.exp(inputs / moving_instance_max)
        else:
            exp_logit = tf.exp(inputs)

        pad_exp_logit = tf.pad(
            exp_logit, [[0, 0], [dilation, dilation], [dilation, dilation], [0, 0]], constant_values=1
        )
        sum_logit = tf.nn.pool(pad_exp_logit, [ksize, ksize], "AVG", "VALID", dilation_rate=[dilation, dilation]) * (
            ksize ** 2
        )
        alpha = exp_logit / (sum_logit + 1e-6)
        return alpha, beta

    def extract_kpts(self, score_map, k=256, score_thld=0, edge_thld=0, nms_size=3, eof_size=5):
        h = tf.shape(score_map)[1]
        w = tf.shape(score_map)[2]

        mask = score_map > score_thld
        if nms_size > 0:
            nms_mask = tf.nn.max_pool(score_map, ksize=[1, nms_size, nms_size, 1], strides=[1, 1, 1, 1], padding="SAME")
            nms_mask = tf.equal(score_map, nms_mask)
            mask = tf.logical_and(nms_mask, mask)
        if eof_size > 0:
            eof_mask = tf.ones((1, h - 2 * eof_size, w - 2 * eof_size, 1), dtype=tf.float32)
            eof_mask = tf.pad(eof_mask, [[0, 0], [eof_size, eof_size], [eof_size, eof_size], [0, 0]])
            eof_mask = tf.cast(eof_mask, tf.bool)
            mask = tf.logical_and(eof_mask, mask)
        if edge_thld > 0:
            edge_mask = self.edge_mask(score_map, 1, dilation=3, edge_thld=edge_thld)
            mask = tf.logical_and(edge_mask, mask)

        mask = tf.reshape(mask, (h, w))
        score_map = tf.reshape(score_map, (h, w))
        indices = tf.where(mask)
        scores = tf.gather_nd(score_map, indices)
        sample = tf.argsort(scores, direction="DESCENDING")[0:k]

        indices = tf.expand_dims(tf.gather(indices, sample), axis=0)
        scores = tf.expand_dims(tf.gather(scores, sample), axis=0)

        return indices, scores

    def kpt_refinement(self, inputs):
        n_channel = inputs.get_shape()[-1]

        di_filter = tf.reshape(tf.constant([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]), (3, 3, 1, 1))
        dj_filter = tf.reshape(tf.constant([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]), (3, 3, 1, 1))
        dii_filter = tf.reshape(tf.constant([[0, 1.0, 0], [0, -2.0, 0], [0, 1.0, 0]]), (3, 3, 1, 1))
        dij_filter = tf.reshape(0.25 * tf.constant([[1.0, 0, -1.0], [0, 0.0, 0], [-1.0, 0, 1.0]]), (3, 3, 1, 1))
        djj_filter = tf.reshape(tf.constant([[0, 0, 0], [1.0, -2.0, 1.0], [0, 0, 0]]), (3, 3, 1, 1))

        dii_filter = tf.tile(dii_filter, (1, 1, n_channel, 1))
        dii = tf.nn.depthwise_conv2d(inputs, filter=dii_filter, strides=[1, 1, 1, 1], padding="SAME")

        dij_filter = tf.tile(dij_filter, (1, 1, n_channel, 1))
        dij = tf.nn.depthwise_conv2d(inputs, filter=dij_filter, strides=[1, 1, 1, 1], padding="SAME")

        djj_filter = tf.tile(djj_filter, (1, 1, n_channel, 1))
        djj = tf.nn.depthwise_conv2d(inputs, filter=djj_filter, strides=[1, 1, 1, 1], padding="SAME")

        det = dii * djj - dij * dij

        inv_hess_00 = tf.math.divide_no_nan(djj, det)
        inv_hess_01 = tf.math.divide_no_nan(-dij, det)
        inv_hess_11 = tf.math.divide_no_nan(dii, det)

        di_filter = tf.tile(di_filter, (1, 1, n_channel, 1))
        di = tf.nn.depthwise_conv2d(inputs, filter=di_filter, strides=[1, 1, 1, 1], padding="SAME")

        dj_filter = tf.tile(dj_filter, (1, 1, n_channel, 1))
        dj = tf.nn.depthwise_conv2d(inputs, filter=dj_filter, strides=[1, 1, 1, 1], padding="SAME")

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)

        return tf.stack([step_i, step_j], axis=-1)

    def edge_mask(self, inputs, n_channel, dilation=1, edge_thld=5):
        # non-edge
        dii_filter = tf.reshape(tf.constant([[0, 1.0, 0], [0, -2.0, 0], [0, 1.0, 0]]), (3, 3, 1, 1))
        dij_filter = tf.reshape(0.25 * tf.constant([[1.0, 0, -1.0], [0, 0.0, 0], [-1.0, 0, 1.0]]), (3, 3, 1, 1))
        djj_filter = tf.reshape(tf.constant([[0, 0, 0], [1.0, -2.0, 1.0], [0, 0, 0]]), (3, 3, 1, 1))

        dii_filter = tf.tile(dii_filter, (1, 1, n_channel, 1))

        pad_inputs = tf.pad(inputs, [[0, 0], [dilation, dilation], [dilation, dilation], [0, 0]], constant_values=0)

        dii = tf.nn.depthwise_conv2d(
            pad_inputs, filter=dii_filter, strides=[1, 1, 1, 1], padding="VALID", dilations=[dilation] * 2
        )

        dij_filter = tf.tile(dij_filter, (1, 1, n_channel, 1))
        dij = tf.nn.depthwise_conv2d(
            pad_inputs, filter=dij_filter, strides=[1, 1, 1, 1], padding="VALID", dilations=[dilation] * 2
        )

        djj_filter = tf.tile(djj_filter, (1, 1, n_channel, 1))
        djj = tf.nn.depthwise_conv2d(
            pad_inputs, filter=djj_filter, strides=[1, 1, 1, 1], padding="VALID", dilations=[dilation] * 2
        )

        det = dii * djj - dij * dij
        tr = dii + djj
        thld = (edge_thld + 1) ** 2 / edge_thld
        is_not_edge = tf.logical_and(tr * tr / det <= thld, det > 0)
        return is_not_edge


def interpolate(pos, inputs, batched=True, nd=True):
    if not batched:
        pos = tf.expand_dims(pos, 0)
        inputs = tf.expand_dims(inputs, 0)

    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]

    i = pos[:, :, 0]
    j = pos[:, :, 1]

    i_top_left = tf.clip_by_value(tf.cast(tf.math.floor(i), tf.int32), 0, h - 1)
    j_top_left = tf.clip_by_value(tf.cast(tf.math.floor(j), tf.int32), 0, w - 1)

    i_top_right = tf.clip_by_value(tf.cast(tf.math.floor(i), tf.int32), 0, h - 1)
    j_top_right = tf.clip_by_value(tf.cast(tf.math.ceil(j), tf.int32), 0, w - 1)

    i_bottom_left = tf.clip_by_value(tf.cast(tf.math.ceil(i), tf.int32), 0, h - 1)
    j_bottom_left = tf.clip_by_value(tf.cast(tf.math.floor(j), tf.int32), 0, w - 1)

    i_bottom_right = tf.clip_by_value(tf.cast(tf.math.ceil(i), tf.int32), 0, h - 1)
    j_bottom_right = tf.clip_by_value(tf.cast(tf.math.ceil(j), tf.int32), 0, w - 1)

    dist_i_top_left = i - tf.cast(i_top_left, tf.float32)
    dist_j_top_left = j - tf.cast(j_top_left, tf.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    if nd:
        w_top_left = w_top_left[..., None]
        w_top_right = w_top_right[..., None]
        w_bottom_left = w_bottom_left[..., None]
        w_bottom_right = w_bottom_right[..., None]

    interpolated_val = (
        w_top_left * tf.gather_nd(inputs, tf.stack([i_top_left, j_top_left], axis=-1), batch_dims=1)
        + w_top_right * tf.gather_nd(inputs, tf.stack([i_top_right, j_top_right], axis=-1), batch_dims=1)
        + w_bottom_left * tf.gather_nd(inputs, tf.stack([i_bottom_left, j_bottom_left], axis=-1), batch_dims=1)
        + w_bottom_right * tf.gather_nd(inputs, tf.stack([i_bottom_right, j_bottom_right], axis=-1), batch_dims=1)
    )

    if not batched:
        interpolated_val = tf.squeeze(interpolated_val, axis=0)
    return interpolated_val
