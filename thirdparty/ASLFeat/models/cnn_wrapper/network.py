#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
CNN layer wrapper.

Please be noted about following issues:

1. The center and scale paramter are disabled by default for all BN-related layers, as they have
shown little influence on final performance. In particular, scale params is officially considered
unnecessary as oftentimes followed by ReLU.

2. By default we apply L2 regularization only on kernel or bias parameters, but not learnable BN
coefficients (i.e. center/scale) as suggested in ResNet paper. Be noted to add regularization terms
into tf.GraphKeys.REGULARIZATION_LOSSES if you are desgining custom layers.

3. Since many of models are converted from Caffe, we are by default setting the epsilon paramter in
BN to 1e-5 as that is in Caffe, while 1e-3 in TensorFlow. It may cause slightly different behavior
if you are using models from other deep learning toolboxes.
"""

import numpy as np
import tensorflow as tf

# Zero padding in default. 'VALID' gives no padding.
DEFAULT_PADDING = "SAME"


def caffe_like_padding(input_tensor, padding):
    """A padding method that has same behavior as Caffe's."""

    def PAD(x):
        return [x, x]

    if len(input_tensor.get_shape()) == 4:
        padded_input = tf.pad(input_tensor, [PAD(0), PAD(padding), PAD(padding), PAD(0)], "CONSTANT")
    elif len(input_tensor.get_shape()) == 5:
        padded_input = tf.pad(input_tensor, [PAD(0), PAD(padding), PAD(padding), PAD(padding), PAD(0)], "CONSTANT")
    return padded_input


def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        """Layer decoration."""
        # We allow to construct low-level layers instead of high-level networks.
        if self.inputs is None or (len(args) > 0 and isinstance(args[0], tf.Tensor)):
            layer_output = op(self, *args, **kwargs)
            return layer_output
        # Automatically set a name if not provided.
        name = kwargs.setdefault("name", self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if not self.terminals:
            raise RuntimeError("No input variables found for layer %s." % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    """Class NetWork."""

    def __init__(
        self,
        inputs,
        is_training,
        dropout_rate=0.5,
        seed=None,
        epsilon=1e-5,
        reuse=False,
        fcn=True,
        regularize=True,
        **kwargs
    ):
        # The input nodes for this network
        self.inputs = inputs
        # If true, the resulting variables are set as trainable
        self.trainable = is_training if isinstance(is_training, bool) else True
        # If true, variables are shared between feature towers
        self.reuse = reuse
        # If true, layers like batch normalization or dropout are working in training mode
        self.training = is_training
        # Dropout rate
        self.dropout_rate = dropout_rate
        # Seed for randomness
        self.seed = seed
        # Add regularizer for parameters.
        self.regularizer = tf.keras.regularizers.L2(1.0) if regularize else None
        # The epsilon paramater in BN layer.
        self.bn_epsilon = epsilon
        self.extra_args = kwargs
        # Endpoints.
        self.endpoints = {}
        if inputs is not None:
            # The current list of terminal nodes
            self.terminals = []
            # Mapping from layer names to layers
            self.layers = dict(inputs)
            # If true, dense layers will be omitted in network construction
            self.fcn = fcn
            self.setup()

    def setup(self):
        """Construct the network."""
        raise NotImplementedError("Must be implemented by the subclass.")

    def load(self, data_path, session, ignore_missing=False, exclude_var=None):
        """Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = np.load(data_path, encoding="latin1").item()
        if exclude_var is not None:
            keyword = exclude_var.split(",")
        assign_op = []
        for op_name in data_dict:
            if exclude_var is not None:
                find_keyword = False
                for tmp_keyword in keyword:
                    if op_name.find(tmp_keyword) >= 0:
                        find_keyword = True
                if find_keyword:
                    continue

            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():

                    try:
                        var = tf.get_variable(param_name)
                        assign_op.append(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise
                        else:
                            print(Notify.WARNING, ":".join([op_name, param_name]), "is omitted.", Notify.ENDC)
        session.run(assign_op)

    def feed(self, *args):
        """Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert args
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError("Unknown layer name fed: %s" % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        """Returns the current network output."""
        return self.terminals[-1]

    def get_output_by_name(self, layer_name):
        """
        Get graph node by layer name
        :param layer_name: layer name string
        :return: tf node
        """
        return self.layers[layer_name]

    def get_unique_name(self, prefix):
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return "%s_%d" % (prefix, ident)

    def change_inputs(self, input_tensors):
        assert len(input_tensors) == 1
        for key in input_tensors:
            self.layers[key] = input_tensors[key]

    @layer
    def conv(
        self,
        input_tensor,
        kernel_size,
        filters,
        strides,
        name,
        relu=True,
        dilation_rate=1,
        padding=DEFAULT_PADDING,
        biased=True,
        reuse=False,
        kernel_init=None,
        bias_init=tf.zeros_initializer,
        separable=False,
    ):
        """2D/3D convolution."""
        kwargs = {
            "filters": filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "activation": tf.nn.relu if relu else None,
            "use_bias": biased,
            "dilation_rate": dilation_rate,
            "trainable": self.trainable,
            "reuse": self.reuse or reuse,
            "bias_regularizer": self.regularizer if biased else None,
            "kernel_initializer": kernel_init,
            "bias_initializer": bias_init,
            "name": name,
        }

        if separable:
            kwargs["depthwise_regularizer"] = self.regularizer
            kwargs["pointwise_regularizer"] = self.regularizer
        else:
            kwargs["kernel_regularizer"] = self.regularizer

        if isinstance(padding, str):
            padded_input = input_tensor
            kwargs["padding"] = padding
        else:
            padded_input = caffe_like_padding(input_tensor, padding)
            kwargs["padding"] = "VALID"

        if len(input_tensor.get_shape()) == 4:
            if not separable:
                return tf.compat.v1.layers.conv2d(padded_input, **kwargs)
            else:
                return tf.compat.v1.layers.separable_conv2d(padded_input, **kwargs)
        elif len(input_tensor.get_shape()) == 5:
            if not separable:
                return tf.compat.v1.layers.conv3d(padded_input, **kwargs)
            else:
                raise NotImplementedError("No official implementation for separable_conv3d")
        else:
            raise ValueError("Improper input rank for layer: " + name)

    @layer
    def conv_bn(
        self,
        input_tensor,
        kernel_size,
        filters,
        strides,
        name,
        relu=True,
        center=False,
        scale=False,
        dilation_rate=1,
        padding=DEFAULT_PADDING,
        biased=False,
        separable=False,
        reuse=False,
    ):
        conv = self.conv(
            input_tensor,
            kernel_size,
            filters,
            strides,
            name,
            relu=False,
            dilation_rate=dilation_rate,
            padding=padding,
            biased=biased,
            reuse=reuse,
            separable=separable,
        )
        conv_bn = self.batch_normalization(conv, name + "/bn", center=center, scale=scale, relu=relu, reuse=reuse)
        return conv_bn

    @layer
    def deconv(
        self,
        input_tensor,
        kernel_size,
        filters,
        strides,
        name,
        relu=True,
        padding=DEFAULT_PADDING,
        biased=True,
        reuse=False,
    ):
        """2D/3D deconvolution."""
        kwargs = {
            "filters": filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "activation": tf.nn.relu if relu else None,
            "use_bias": biased,
            "trainable": self.trainable,
            "reuse": self.reuse or reuse,
            "kernel_regularizer": self.regularizer,
            "bias_regularizer": self.regularizer if biased else None,
            "name": name,
        }

        if isinstance(padding, str):
            padded_input = input_tensor
            kwargs["padding"] = padding
        else:
            padded_input = caffe_like_padding(input_tensor, padding)
            kwargs["padding"] = "VALID"

        if len(input_tensor.get_shape()) == 4:
            return tf.compat.v1.layers.conv2d_transpose(padded_input, **kwargs)
        elif len(input_tensor.get_shape()) == 5:
            return tf.compat.v1.layers.conv3d_transpose(padded_input, **kwargs)
        else:
            raise ValueError("Improper input rank for layer: " + name)

    @layer
    def deconv_bn(
        self,
        input_tensor,
        kernel_size,
        filters,
        strides,
        name,
        relu=True,
        center=False,
        scale=False,
        padding=DEFAULT_PADDING,
        biased=False,
        reuse=False,
    ):
        deconv = self.deconv(
            input_tensor, kernel_size, filters, strides, name, relu=False, padding=padding, biased=biased, reuse=reuse
        )
        deconv_bn = self.batch_normalization(deconv, name + "/bn", center=center, scale=scale, relu=relu, reuse=reuse)
        return deconv_bn

    @layer
    def relu(self, input_tensor, name=None):
        """ReLu activation."""
        return tf.nn.relu(input_tensor, name=name)

    @layer
    def max_pool(self, input_tensor, pool_size, strides, name, padding=DEFAULT_PADDING):
        """Max pooling."""
        if isinstance(padding, str):
            padded_input = input_tensor
            padding_type = padding
        else:
            padded_input = caffe_like_padding(input_tensor, padding)
            padding_type = "VALID"

        return tf.compat.v1.layers.max_pooling2d(
            padded_input, pool_size=pool_size, strides=strides, padding=padding_type, name=name
        )

    @layer
    def avg_pool(self, input_tensor, pool_size, strides, name, padding=DEFAULT_PADDING):
        """ "Average pooling."""
        if isinstance(padding, str):
            padded_input = input_tensor
            padding_type = padding
        else:
            padded_input = caffe_like_padding(input_tensor, padding)
            padding_type = "VALID"
        return tf.compat.v1.layers.average_pooling2d(
            padded_input, pool_size=pool_size, strides=strides, padding=padding_type, name=name
        )

    @layer
    def concat(self, input_tensors, axis, name):
        return tf.concat(values=input_tensors, axis=axis, name=name)

    @layer
    def add(self, input_tensors, name):
        return tf.add_n(input_tensors, name=name)

    @layer
    def fc(self, input_tensor, num_out, name, biased=True, relu=True, flatten=True, reuse=False):
        # To behave same to Caffe.
        if flatten:
            flatten_tensor = tf.compat.v1.layers.flatten(input_tensor)
        else:
            flatten_tensor = input_tensor
        return tf.compat.v1.layers.dense(
            flatten_tensor,
            units=num_out,
            use_bias=biased,
            activation=tf.nn.relu if relu else None,
            trainable=self.trainable,
            reuse=self.reuse or reuse,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer if biased else None,
            name=name,
        )

    @layer
    def fc_bn(
        self, input_tensor, num_out, name, biased=False, relu=True, center=False, scale=False, flatten=True, reuse=False
    ):
        # To behave same to Caffe.
        fc = self.fc(input_tensor, num_out, name, relu=False, biased=biased, flatten=flatten, reuse=reuse)
        fc_bn = self.batch_normalization(fc, name + "/bn", center=center, scale=scale, relu=relu, reuse=reuse)
        return fc_bn

    @layer
    def softmax(self, input_tensor, name, dim=-1):
        return tf.nn.softmax(input_tensor, dim=dim, name=name)

    @layer
    def batch_normalization(self, input_tensor, name, center=False, scale=False, relu=False, reuse=False):
        """Batch normalization."""
        output = tf.compat.v1.layers.batch_normalization(
            input_tensor,
            center=center,
            scale=scale,
            fused=True,
            training=self.training,
            trainable=self.trainable,
            reuse=self.reuse or reuse,
            epsilon=self.bn_epsilon,
            gamma_regularizer=None,  # self.regularizer if scale else None,
            beta_regularizer=None,  # self.regularizer if center else None,
            name=name,
        )
        if relu:
            output = self.relu(output, name + "/relu")
        return output

    @layer
    def context_normalization(self, input_tensor, name):
        """The input is a feature matrix with a shape of BxNx1xD"""
        mean, variance = tf.nn.moments(input_tensor, axes=[1], keep_dims=True)
        output = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, self.bn_epsilon)
        return output

    @layer
    def l2norm(self, input_tensor, name, axis=-1):
        return tf.nn.l2_normalize(input_tensor, axis=axis, name=name)

    @layer
    def squeeze(self, input_tensor, axis=None, name=None):
        return tf.squeeze(input_tensor, axis=axis, name=name)

    @layer
    def reshape(self, input_tensor, shape, name=None):
        return tf.reshape(input_tensor, shape, name=name)

    @layer
    def flatten(self, input_tensor, name=None):
        return tf.compat.v1.layers.flatten(input_tensor, name=name)

    @layer
    def tanh(self, input_tensor, name=None):
        return tf.tanh(input_tensor, name=name)

    @layer
    def deform_conv(
        self,
        input_tensor,
        kernel_size,
        filters,
        strides,
        name,
        relu=True,
        deform_type="u",
        modulated=True,
        biased=True,
        dilation_rate=1,
        padding=DEFAULT_PADDING,
        kernel_init=None,
        bias_init=tf.zeros_initializer,
        reuse=False,
    ):
        def _pad_input(inputs, kernel_size, strides, dilation_rate):
            if self.training:
                in_shape = inputs.get_shape().as_list()[1:3]
            else:
                in_shape = tf.shape(inputs)[1:3]

            padding_list = []
            dilated_filter_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
            for i in range(2):
                same_output = (in_shape[i] + strides - 1) // strides
                valid_output = (in_shape[i] - dilated_filter_size + strides) // strides

                p = dilated_filter_size - 1
                p_0 = p // 2

                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    padding_list += [p_0, p - p_0]

            padding = [
                [0, 0],
                [padding_list[0], padding_list[1]],  # top, bottom padding
                [padding_list[2], padding_list[3]],  # left, right padding
                [0, 0],
            ]
            inputs = tf.pad(inputs, padding)
            return inputs

        def _get_conv_indices(feature_map_size, kernel_size, strides, dilation_rate):
            """the x, y coordinates in the window when a filter sliding on the feature map

            :param feature_map_size:
            :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
            """
            feat_h, feat_w = [i for i in feature_map_size[0:2]]

            x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
            x, y = [tf.reshape(i, [1, feat_h, feat_w, 1]) for i in [x, y]]  # shape [1, h, w, 1]
            x, y = [
                tf.image.extract_patches(
                    i,
                    [1, kernel_size, kernel_size, 1],
                    [1, strides, strides, 1],
                    [1, dilation_rate, dilation_rate, 1],
                    "VALID",
                )
                for i in [x, y]
            ]  # shape [1, out_h, out_w, filter_h * filter_w]
            return y, x

        offset_num = kernel_size ** 2

        # add padding if needed
        if padding == "SAME":
            inputs = _pad_input(input_tensor, kernel_size, strides, dilation_rate)
        # some length
        batch_size = tf.shape(inputs)[0]
        if self.training:
            in_h, in_w = [inputs.get_shape()[i].value for i in range(1, 3)]
            ori_h, ori_w = [input_tensor.get_shape()[i].value for i in range(1, 3)]
        else:
            in_h, in_w = [tf.shape(inputs)[i] for i in range(1, 3)]
            ori_h, ori_w = [tf.shape(input_tensor)[i] for i in range(1, 3)]

        with tf.compat.v1.variable_scope("deform_param", reuse=reuse):
            if deform_type == "a":
                # similarity est
                ori = self.conv(
                    inputs,
                    kernel_size,
                    2,
                    strides,
                    name + "/ori",
                    relu=False,
                    dilation_rate=dilation_rate,
                    padding="VALID",
                    kernel_init=tf.zeros_initializer,
                    bias_init=tf.constant_initializer([1.0, 0.0]),
                    biased=True,
                    reuse=reuse,
                )
                ori = tf.nn.l2_normalize(ori, axis=-1)
                cos = ori[:, :, :, 0, None]
                sin = ori[:, :, :, 1, None]
                R = tf.stack([tf.concat([cos, sin], axis=-1), tf.concat([-sin, cos], axis=-1)], axis=-2)
                scale = self.conv(
                    inputs,
                    kernel_size,
                    1,
                    strides,
                    name + "/scale",
                    relu=False,
                    dilation_rate=dilation_rate,
                    padding="VALID",
                    kernel_init=tf.zeros_initializer,
                    bias_init=tf.constant_initializer(0.0),
                    biased=True,
                    reuse=reuse,
                )
                scale = tf.exp(tf.tanh(scale))
                comb_A = scale[..., None] * R
                # aff est
                if False:
                    aff = self.conv(
                        inputs,
                        kernel_size,
                        3,
                        strides,
                        name + "/aff",
                        relu=False,
                        dilation_rate=dilation_rate,
                        padding="VALID",
                        kernel_init=tf.zeros_initializer,
                        bias_init=tf.zeros_initializer,
                        biased=True,
                        reuse=reuse,
                    )
                    aff = self.tanh(aff)

                    a00 = 1.0 + aff[:, :, :, 0]
                    a01 = tf.zeros_like(aff[:, :, :, 0])
                    a10 = aff[:, :, :, 1]
                    a11 = 1.0 + aff[:, :, :, 2]

                    det = tf.sqrt(tf.abs(a00 * a11 + 1e-10))
                    b2a2 = tf.abs(a00 + 1e-10)
                    a00_new = b2a2 / det
                    a01_new = a01
                    a10_new = (a10 * a00) / (b2a2 * det)
                    a11_new = det / b2a2
                    A = tf.stack(
                        [tf.stack([a00_new, a01_new], axis=-1), tf.stack([a10_new, a11_new], axis=-1)], axis=-2
                    )

                    comb_A = tf.matmul(comb_A, A)

                rng = tf.range(-(kernel_size // 2), kernel_size // 2 + 1)
                x, y = tf.meshgrid(rng, rng)
                x = tf.reshape(x, (-1,))
                y = tf.reshape(y, (-1,))
                xy = tf.reshape(tf.stack([x, y], axis=-1), [1, 1, 1, -1, 2])
                xy = tf.tile(xy, [tf.shape(comb_A)[0], tf.shape(comb_A)[1], tf.shape(comb_A)[2], 1, 1])
                xy = tf.cast(xy, tf.float32)

                offset = tf.matmul(xy, comb_A) - xy
                offset = tf.stack([offset[:, :, :, :, 1], offset[:, :, :, :, 0]], axis=-1)
                if self.training:
                    offset = tf.reshape(
                        offset, [batch_size, comb_A.get_shape()[1].value, comb_A.get_shape()[2], offset_num * 2]
                    )
                else:
                    offset = tf.reshape(offset, [batch_size, tf.shape(comb_A)[1], tf.shape(comb_A)[2], offset_num * 2])
            elif deform_type == "h":
                h4p_offset = self.conv(
                    inputs,
                    kernel_size,
                    8,
                    strides,
                    name + "/offset",
                    relu=False,
                    dilation_rate=dilation_rate,
                    padding="VALID",
                    kernel_init=tf.zeros_initializer,
                    bias_init=tf.zeros_initializer,
                    biased=True,
                    reuse=reuse,
                )
                h4p_offset = self.tanh(h4p_offset) * 0.95
                if False:
                    H_mat = solve_DLT(h4p_offset, self.training)
                else:
                    scale = self.conv(
                        inputs,
                        kernel_size,
                        1,
                        strides,
                        name + "/scale",
                        relu=False,
                        dilation_rate=dilation_rate,
                        padding="VALID",
                        kernel_init=tf.zeros_initializer,
                        bias_init=tf.constant_initializer(0.0),
                        biased=True,
                        reuse=reuse,
                    )
                    scale = tf.exp(tf.tanh(scale))
                    H_mat = solve_DLT(h4p_offset, self.training, scale=scale)

                rng = tf.range(-(kernel_size // 2), kernel_size // 2 + 1)
                x, y = tf.meshgrid(rng, rng)
                x = tf.reshape(x, (-1,))
                y = tf.reshape(y, (-1,))
                xy = tf.reshape(tf.stack([x, y], axis=-1), [1, 1, 1, -1, 2])
                xy = tf.tile(xy, [tf.shape(H_mat)[0], tf.shape(H_mat)[1], tf.shape(H_mat)[2], 1, 1])
                xy = tf.cast(xy, tf.float32)
                ones = tf.ones_like(xy[:, :, :, :, 0])[..., None]
                xy_homo = tf.concat([xy, ones], axis=-1)

                pert_xy = tf.matmul(xy_homo, H_mat, transpose_b=True)
                homo_scale = tf.expand_dims(pert_xy[:, :, :, :, -1], axis=-1)
                pert_xy = pert_xy[:, :, :, :, 0:2]
                pert_xy = tf.clip_by_value(tf.math.divide_no_nan(pert_xy, homo_scale), -10.0, 10.0)

                offset = pert_xy - xy
                offset = tf.stack([offset[:, :, :, :, 1], offset[:, :, :, :, 0]], axis=-1)
                if self.training:
                    offset = tf.reshape(
                        offset, [batch_size, H_mat.get_shape()[1].value, H_mat.get_shape()[2], offset_num * 2]
                    )
                else:
                    offset = tf.reshape(offset, [batch_size, tf.shape(H_mat)[1], tf.shape(H_mat)[2], offset_num * 2])
            elif deform_type == "u":
                offset = self.conv(
                    inputs,
                    kernel_size,
                    offset_num * 2,
                    strides,
                    name + "/offset",
                    relu=False,
                    dilation_rate=dilation_rate,
                    padding="VALID",
                    kernel_init=tf.zeros_initializer,
                    bias_init=tf.zeros_initializer,
                    biased=True,
                    reuse=reuse,
                )
            if modulated:
                amplitude = self.conv(
                    inputs,
                    kernel_size,
                    offset_num,
                    strides,
                    name + "/amplitude",
                    relu=False,
                    dilation_rate=dilation_rate,
                    padding="VALID",
                    kernel_init=tf.zeros_initializer,
                    bias_init=tf.zeros_initializer,
                    biased=True,
                    reuse=reuse,
                )
                amplitude = tf.math.sigmoid(amplitude)

        if self.training:
            out_h, out_w = [offset.get_shape()[i].value for i in range(1, 3)]
        else:
            out_h, out_w = [tf.shape(offset)[i] for i in range(1, 3)]
        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, offset_num, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = _get_conv_indices([in_h, in_w], kernel_size, strides, dilation_rate)
        y, x = [tf.reshape(i, [batch_size, out_h, out_w, offset_num]) for i in [y, x]]
        y, x = [tf.cast(i, tf.float32) for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, tf.cast(in_h, tf.float32) - 1)
        x = tf.clip_by_value(x, 0, tf.cast(in_w, tf.float32) - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), tf.int32) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [tf.gather_nd(inputs, tf.stack(i, axis=-1), batch_dims=1) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.cast(i, tf.float32) for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        if modulated:
            pixels *= tf.expand_dims(amplitude, -1)

        c = inputs.get_shape()[-1]
        if False:
            with tf.compat.v1.variable_scope(name + "/agg", reuse=self.reuse):
                weights = tf.compat.v1.get_variable("weights", [kernel_size ** 2, c], dtype=tf.float32)
                out = tf.reduce_sum(pixels * weights, axis=-2)
        else:
            with tf.compat.v1.variable_scope(name, reuse=self.reuse):
                weights = tf.compat.v1.get_variable(
                    "kernel", (kernel_size, kernel_size, c, filters), dtype=tf.float32, initializer=kernel_init
                )
                weights = tf.reshape(weights, (1, 1, kernel_size ** 2, c, filters))
                out = tf.nn.conv3d(input=pixels, filters=weights, strides=[1, 1, 1, 1, 1], padding="VALID", name=None)
                out = tf.squeeze(out, axis=-2)

            if biased:
                bias = tf.compat.v1.get_variable("bias", (filters,), dtype=tf.float32, initializer=bias_init)
                out = tf.nn.bias_add(out, bias)

        if relu:
            out = tf.nn.relu(out)
        return out

    @layer
    def deform_conv_bn(
        self,
        input_tensor,
        kernel_size,
        filters,
        strides,
        name,
        relu=True,
        center=False,
        scale=False,
        deform_type="u",
        modulated=True,
        biased=False,
        dilation_rate=1,
        padding=DEFAULT_PADDING,
        reuse=False,
    ):
        deform_conv = self.deform_conv(
            input_tensor,
            kernel_size,
            filters,
            strides,
            name,
            relu=False,
            deform_type=deform_type,
            modulated=modulated,
            dilation_rate=dilation_rate,
            padding=padding,
            biased=biased,
            reuse=reuse,
        )
        deform_conv_bn = self.batch_normalization(
            deform_conv, name + "/bn", center=center, scale=scale, relu=relu, reuse=reuse
        )
        return deform_conv_bn
