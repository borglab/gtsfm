import os
import glob
import numpy as np
import tensorflow as tf

from utils.common import Notify
from .base_dataset import BaseDataset


class Aachen(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        base_path = config['data_root']
        base_path = os.path.join(base_path, 'images', 'images_upright')
        seq_paths = config['data_split']
        image_paths = []
        for tmp_seq in seq_paths:
            seq_path = os.path.join(base_path, tmp_seq)
            image_paths.extend(glob.glob(os.path.join(seq_path, '*.jpg')))
        if config['truncate'] is not None:
            print(Notify.WARNING, "Truncate from",
                  config['truncate'][0], "to", config['truncate'][1], Notify.ENDC)
            image_paths = image_paths[config['truncate'][0]:config['truncate'][1]]
        dump_paths = [image_paths[i] + self.config['post_format']['suffix'] for i in range(len(image_paths))]
        print(Notify.INFO, "Found images:", len(image_paths), Notify.ENDC)

        self.data_length = len(image_paths)
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])

        files = {'image_paths': image_paths, 'dump_paths': dump_paths}
        return files

    def _format_data(self, data):
        dump_path = data['dump_path'].decode('utf-8')
        feat = data['dump_data'][0]
        kpt = data['dump_data'][1]
        with open(dump_path, 'wb') as fout:
            np.savez(fout, keypoints=kpt, descriptors=feat)