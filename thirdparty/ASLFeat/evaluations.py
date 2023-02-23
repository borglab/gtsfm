#!/usr/bin/env python3

"""
Copyright 2020, Zixin Luo, HKUST.
Evaluation script.
"""

import os
import yaml

import tensorflow as tf
import progressbar

from datasets import get_dataset
from models import get_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def extract_feat(config):
    """Extract augmented features."""
    prog_bar = progressbar.ProgressBar()
    config['stage'] = 'det'
    dataset = get_dataset(config['data_name'])(**config)
    prog_bar.max_value = dataset.data_length
    test_set = dataset.get_test_set()

    model = get_model('feat_model')(config['model_path'], **(config['net']))
    idx = 0
    while True:
        try:
            data = next(test_set)
            if config['overwrite'] or not os.path.exists(data['dump_path']):
                desc, kpt, score = model.run_test_data(data['image'])
                dump_data = {}
                dump_data['dump_data'] = (desc, kpt, score)
                dump_data['image_path'] = data['image_path']
                dump_data['dump_path'] = data['dump_path']
                dataset.format_data(dump_data)
            prog_bar.update(idx)
            idx += 1
        except dataset.end_set:
            break
    model.close()


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config['dump_root'] is not None and not os.path.exists(config['dump_root']):
        os.mkdir(config['dump_root'])
    extract_feat(config)


if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['config'])
    tf.compat.v1.app.run()
