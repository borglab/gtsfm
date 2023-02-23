import os
import glob
from struct import pack
import tensorflow as tf
import numpy as np

from utils.common import Notify
from .base_dataset import BaseDataset


class Eth(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        base_path = config['data_root']
        suffix = self.config['post_format']['suffix']

        img_paths = []
        dump_paths = []

        data_split = config['data_split']

        types = ('*.jpg', '*.png', '*.JPG', '*.PNG')
        for seq in data_split:
            dataset_folder = os.path.join(base_path, seq)
            image_folder = os.path.join(dataset_folder, 'images')
            dump_folder = os.path.join(dataset_folder, 'reconstruction' + suffix)
            kpt_folder = os.path.join(dump_folder, 'keypoints')
            desc_folder = os.path.join(dump_folder, 'descriptors')
            if not os.path.exists(kpt_folder):
                os.makedirs(kpt_folder)
            if not os.path.exists(desc_folder):
                os.makedirs(desc_folder)
            for filetype in types:
                image_list = glob.glob(os.path.join(image_folder, filetype))
                dump_list = [os.path.join(dump_folder, 'to_be_replaced', os.path.basename(i) + '.bin') for i in image_list]
                img_paths.extend(image_list)
                dump_paths.extend(dump_list)
        self.data_length = len(img_paths)
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])

        files = {'image_paths': img_paths, 'dump_paths': dump_paths}
        return files

    def _format_data(self, data):
        dump_path = data['dump_path'].decode('utf-8')
        kpt_dump_path = dump_path.replace('to_be_replaced', 'keypoints')
        desc_dump_path = dump_path.replace('to_be_replaced', 'descriptors')

        desc = data['dump_data'][0].astype(np.float32)
        kpt = data['dump_data'][1].astype(np.float32)
        zeros = np.zeros_like(kpt)
        kpt = np.concatenate([kpt, zeros], axis=-1)

        num_features = desc.shape[0]
        loc_dim = kpt.shape[1]
        feat_dim = desc.shape[1]

        det_head = np.stack((num_features, loc_dim)).astype(np.int32)
        det_head = pack('2i', *det_head)

        desc_head = np.stack((num_features, feat_dim)).astype(np.int32)
        desc_head = pack('2i', *desc_head)

        kpt = pack('f' * loc_dim * num_features, *(kpt.flatten()))
        desc = pack('f' * feat_dim * num_features, *(desc.flatten()))

        with open(kpt_dump_path, 'wb') as fout:
            fout.write(det_head)
            if num_features > 0:
                fout.write(kpt)

        with open(desc_dump_path, 'wb') as fout:
            fout.write(desc_head)
            if num_features > 0:
                fout.write(desc)