import os
import glob
import h5py
import tensorflow as tf

from utils.common import Notify
from .base_dataset import BaseDataset


class Gl3d(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        base_path = config['data_root']

        proj_list = os.path.join(base_path, 'list', config['data_split'], 'imageset_all.txt')
        proj_paths = open(proj_list).read().splitlines()

        image_paths = []
        dump_paths = []

        for tmp_seq in proj_paths:
            dump_folder = os.path.join(config['dump_root'], tmp_seq)
            if not os.path.exists(dump_folder):
                os.makedirs(dump_folder)
            image_folder = os.path.join(base_path, 'data', tmp_seq, 'undist_images')
            image_list = glob.glob(os.path.join(image_folder, '*.jpg'))
            image_paths.extend(image_list)
            dump_list = [os.path.join(dump_folder, os.path.basename(i) + config['post_format']['suffix']) for i in image_list]
            dump_paths.extend(dump_list)

        if config['truncate'] is not None:
            print(Notify.WARNING, "Truncate from",
                  config['truncate'][0], "to", config['truncate'][1], Notify.ENDC)
            image_paths = image_paths[config['truncate'][0]:config['truncate'][1]]

        print(Notify.INFO, "Found images:", len(image_paths), Notify.ENDC)

        self.data_length = len(image_paths)
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])

        files = {'image_paths': image_paths, 'dump_paths': dump_paths}
        return files

    def _format_data(self, data):
        dump_path = data['dump_path'].decode('utf-8')
        if not os.path.exists(dump_path):
            gen_f = h5py.File(dump_path, 'w')
            feat = data['dump_data'][0]
            kpt = data['dump_data'][1]
            _ = gen_f.create_dataset('descriptors', data=feat)
            _ = gen_f.create_dataset('keypoints', data=kpt)