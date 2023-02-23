#!/usr/bin/env python3
"""
Copyright 2020, Zixin Luo, HKUST.
Image matching example.
"""
import yaml
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.opencvhelper import MatcherWrapper

from models import get_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def load_imgs(img_paths, max_dim):
    rgb_list = []
    gray_list = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
    return rgb_list, gray_list


def extract_local_features(gray_list, model_path, config):
    model = get_model('feat_model')(model_path, **config)
    descs = []
    kpts = []
    for gray_img in gray_list:
        desc, kpt, _ = model.run_test_data(gray_img)
        print('feature_num', kpt.shape[0])
        descs.append(desc)
        kpts.append(kpt)
    return descs, kpts


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    # parse input
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # load testing images.
    rgb_list, gray_list = load_imgs(config['img_paths'], config['net']['max_dim'])
    # extract regional features.
    descs, kpts = extract_local_features(gray_list, config['model_path'], config['net'])
    # feature matching and draw matches.
    matcher = MatcherWrapper()
    match, mask = matcher.get_matches(
        descs[0], descs[1], kpts[0], kpts[1],
        ratio=config['match']['ratio_test'], cross_check=config['match']['cross_check'],
        err_thld=3, ransac=True, info='ASLFeat')
    # draw matches
    disp = matcher.draw_matches(rgb_list[0], kpts[0], rgb_list[1], kpts[1], match, mask)

    output_name = 'disp.jpg'
    print('image save to', output_name)
    plt.imsave(output_name, disp)


if __name__ == '__main__':
    tf.compat.v1.app.run()
