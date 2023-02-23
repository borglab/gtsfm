import sys
import math

import cv2
import numpy as np
import tensorflow as tf


from .base_model import BaseModel
from .cnn_wrapper.aslfeat import ASLFeatNet

sys.path.append("..")

tf.compat.v1.disable_eager_execution()


class FeatModel(BaseModel):
    endpoints = None
    default_config = {"max_dim": 2048}

    def _init_model(self):
        return

    def _run(self, data):
        assert len(data.shape) == 3
        max_dim = max(data.shape[0], data.shape[1])
        H, W, _ = data.shape

        if max_dim > self.config["max_dim"]:
            downsample_ratio = self.config["max_dim"] / float(max_dim)
            data = cv2.resize(data, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
            data = data[..., np.newaxis]
        data_size = data.shape

        if self.config["config"]["multi_scale"]:
            scale_f = 1 / (2 ** 0.50)
            min_scale = max(0.3, 128 / max(H, W))
            n_scale = math.floor(max(math.log(min_scale) / math.log(scale_f), 1))
            sigma = 0.8
        else:
            n_scale = 1

        descs, kpts, scores = [], [], []
        for i in range(n_scale):
            if i > 0:
                data = cv2.GaussianBlur(data, None, sigma / scale_f)
                data = cv2.resize(data, dsize=None, fx=scale_f, fy=scale_f)[..., np.newaxis]

            feed_dict = {"input:0": np.expand_dims(data, 0)}
            returns = self.sess.run(self.endpoints, feed_dict=feed_dict)
            descs.append(np.squeeze(returns["descs"], axis=0))
            kpts.append(
                np.squeeze(returns["kpts"], axis=0) * np.array([W / data.shape[1], H / data.shape[0]], dtype=np.float32)
            )
            scores.append(np.squeeze(returns["scores"], axis=0))

        descs = np.concatenate(descs, axis=0)
        kpts = np.concatenate(kpts, axis=0)
        scores = np.concatenate(scores, axis=0)

        idxs = np.negative(scores).argsort()[0 : self.config["config"]["kpt_n"]]

        descs = descs[idxs]
        kpts = kpts[idxs]
        scores = scores[idxs]
        return descs, kpts, scores

    def _construct_network(self):
        ph_imgs = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, None, 1), name="input")
        print(ph_imgs)
        mean, variance = tf.nn.moments(tf.cast(ph_imgs, tf.float32), axes=[1, 2], keepdims=True)
        norm_input = tf.nn.batch_normalization(ph_imgs, mean, variance, None, None, 1e-5)
        config_dict = {"det_config": self.config["config"]}
        tower = ASLFeatNet({"data": norm_input}, is_training=False, resue=False, **config_dict)
        self.endpoints = tower.endpoints
