import os
import tensorflow as tf
import numpy as np
from struct import pack

from utils.common import Notify
from .base_dataset import BaseDataset


def write_feature_repo(kpt_coord, desc, output_path):
    assert desc.dtype == np.uint8
    if kpt_coord.size > 1:
        num_features = kpt_coord.shape[0]
    else:
        num_features = 0

    feature_name = 1413892435
    loc_dim = 5
    des_dim = 128

    head = np.stack((feature_name, 0, num_features, loc_dim, des_dim))
    head = pack('5i', *head)

    if num_features > 0:
        zero_pad = np.ones((num_features, 3)) * 2
        kpt_coord = np.concatenate((kpt_coord, zero_pad), axis=-1).astype(np.float32)
        kpt_coord = pack('f' * loc_dim * num_features, *(kpt_coord.flatten()))

        desc = pack('B' * des_dim * num_features, *(desc.flatten()))

    level_num = 1
    per_level_num = num_features

    level_pack = np.stack((level_num, per_level_num))
    level_num = pack('2i', *level_pack)

    with open(output_path, 'wb') as fout:
        fout.write(head)
        if num_features > 0:
            fout.write(kpt_coord)
            fout.write(desc)
        fout.write(level_num)


class Tat(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        base_path = config['data_root']

        img_paths = []
        dump_paths = []

        for d in config['data_list']:
            image_list = os.path.join(base_path, d, 'output', 'preprocess', 'image_list.txt')
            dfeat_folder = os.path.join(base_path, d, 'output', 'preprocess', 'dfeat')
            if not os.path.exists(dfeat_folder):
                os.mkdir(dfeat_folder)
            if not os.path.exists(image_list):
                exit(-1)
            tmp_img_paths = open(image_list).read().splitlines()
            img_paths.extend(tmp_img_paths)
            for idx, _ in enumerate(tmp_img_paths):
                basename = str(idx).strip().zfill(8) + '.dfeat'
                dump_paths.append(os.path.join(dfeat_folder, basename))

        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])

        if config['truncate'] is not None:
            print(Notify.WARNING, "Truncate from",
                  config['truncate'][0], "to", config['truncate'][1], Notify.ENDC)
            img_paths = img_paths[config['truncate'][0]:config['truncate'][1]]
            dump_paths = dump_paths[config['truncate'][0]:config['truncate'][1]]

        self.data_length = len(img_paths)

        files = {'image_paths': img_paths, 'dump_paths': dump_paths}
        return files

    def _format_data(self, data):
        dump_path = data['dump_path'].decode('utf-8')
        desc = data['dump_data'][0]
        desc = (desc * 128 + 128).astype(np.uint8)
        kpt = data['dump_data'][1]
        write_feature_repo(kpt, desc, dump_path)
