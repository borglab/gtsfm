import os
from struct import pack
import tensorflow as tf
import numpy as np

from utils.common import Notify
from .base_dataset import BaseDataset


class Fmbench(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        base_path = config['data_root']
        dump_folder = config['dump_root']

        img_paths = []
        dump_dict = {}

        data_split = config['data_split']

        for seq in data_split:
            pairs = np.loadtxt(os.path.join(base_path, 'Dataset', seq, 'pairs_with_gt.txt'))
            pairs_which = np.loadtxt(os.path.join(
                base_path, 'Dataset', seq, 'pairs_which_dataset.txt'), dtype=str)
            numpairs = len(pairs)

            for i in range(numpairs):
                img_path_l = os.path.join(
                    base_path, 'Dataset', seq, pairs_which[i], 'Images', '%.8d.jpg' % int(pairs[i, 0]))
                if img_path_l not in dump_dict:
                    dump_dict[img_path_l] = []
                    img_paths.append(img_path_l)
                dump_path_l = os.path.join(dump_folder, seq, '%.4d_l.' % (i+1))
                dump_dict[img_path_l].append(dump_path_l)

                img_path_r = os.path.join(
                    base_path, 'Dataset', seq, pairs_which[i], 'Images', '%.8d.jpg' % int(pairs[i, 1]))
                if img_path_r not in dump_dict:
                    dump_dict[img_path_r] = []
                    img_paths.append(img_path_r)
                dump_path_r = os.path.join(dump_folder, seq, '%.4d_r.' % (i+1))
                dump_dict[img_path_r].append(dump_path_r)

        dump_paths = []
        for i in img_paths:
            dump_paths.append(','.join(dump_dict[i]))

        if config['truncate'] is not None:
            print(Notify.WARNING, "Truncate from",
                  config['truncate'][0], "to", config['truncate'][1], Notify.ENDC)
            img_paths = img_paths[config['truncate'][0]:config['truncate'][1]]
            dump_paths = dump_paths[config['truncate'][0]:config['truncate'][1]]

        self.data_length = len(img_paths)
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])

        files = {'image_paths': img_paths, 'dump_paths': dump_paths}
        return files

    def _format_data(self, data):
        dump_root = self.config['dump_root']
        dump_path = data['dump_path'].decode('utf-8')
        dump_path = dump_path.split(',')
        ds_name = dump_path[0].split('/')[-2]
        filename = os.path.basename(dump_path[0])
        if not os.path.exists(dump_root):
            os.mkdir(dump_root)
        sub_dump_folder = os.path.join(dump_root, ds_name)
        if not os.path.exists(sub_dump_folder):
            os.mkdir(sub_dump_folder)

        desc_file = os.path.join(sub_dump_folder, filename + 'descriptors')
        kpt_file = os.path.join(sub_dump_folder, filename + 'keypoints')

        desc = data['dump_data'][0]
        kpt = data['dump_data'][1]

        num_features = desc.shape[0]
        loc_dim = kpt.shape[1]
        feat_dim = desc.shape[1]

        det_head = np.stack((num_features, loc_dim)).astype(np.int32)
        det_head = pack('2i', *det_head)

        desc_head = np.stack((num_features, feat_dim)).astype(np.int32)
        desc_head = pack('2i', *desc_head)

        kpt = pack('f' * loc_dim * num_features, *(kpt.flatten()))
        desc = pack('f' * feat_dim * num_features, *(desc.flatten()))

        with open(kpt_file, 'wb') as fout:
            fout.write(det_head)
            if num_features > 0:
                fout.write(kpt)

        with open(desc_file, 'wb') as fout:
            fout.write(desc_head)
            if num_features > 0:
                fout.write(desc)

        if len(dump_path) > 1:
            for i in range(1, len(dump_path)):
                same_filename = dump_path[i].split('/')[-1].split('_')
                same_filename = '%.4d_%s' % (int(same_filename[0]), same_filename[1][0])
                same_desc_file = os.path.join(sub_dump_folder, same_filename + '.descriptors')
                same_kpt_file = os.path.join(sub_dump_folder, same_filename + '.keypoints')
                if os.path.exists(same_desc_file):
                    os.remove(same_desc_file)
                os.symlink(desc_file, same_desc_file)
                if os.path.exists(same_kpt_file):
                    os.remove(same_kpt_file)
                os.symlink(kpt_file, same_kpt_file)
