import h5py
import os
import pickle
import numpy as np
from sequence import Sequence


class Dataset(object):
    def __init__(self, dataset_path, dump_dir, dump_file, seqs, mode, desc_name, vis_th, pair_num, pair_path=None):
        self.dataset_path = dataset_path
        self.dump_dir = dump_dir
        self.dump_file = os.path.join(dump_dir, dump_file)
        self.seqs = seqs
        self.mode = mode
        self.desc_name = desc_name
        self.vis_th = vis_th
        self.pair_num = pair_num
        self.pair_path = pair_path
        self.dump_data()

    def collect(self):
        data_type = ['xs','ys','Rs','ts', 'ratios', 'mutuals',\
            'cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s']
        pair_idx = 0
        with h5py.File(self.dump_file, 'w') as f:
            data = {}
            for tp in data_type:
                data[tp] = f.create_group(tp)
            for seq in self.seqs:
                print(seq)
                data_seq = {}
                for tp in data_type:
                    data_seq[tp] = pickle.load(open(self.dump_dir+'/'+seq+'/'+self.desc_name+'/'+self.mode+'/'+str(tp)+'.pkl','rb'))
                seq_len = len(data_seq['xs'])

                for i in range(seq_len):
                    for tp in data_type:
                        data_item = data_seq[tp][i]
                        if tp in ['cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s']:
                            data_item = np.asarray([data_item])
                        data_i =  data[tp].create_dataset(str(pair_idx), data_item.shape, dtype=np.float32)
                        data_i[:] = data_item.astype(np.float32)
                    pair_idx = pair_idx + 1
                print('pair idx now ' +str(pair_idx))

    def dump_data(self):
        # make sure you have already saved the features
        for seq in self.seqs:
            pair_name = None if self.pair_path is None else self.pair_path+'/'+seq.rstrip("/")+'-te-'+str(self.pair_num)+'-pairs.pkl'
            dataset_path = self.dataset_path+'/'+seq+'/'+self.mode
            dump_dir = self.dump_dir+'/'+seq+'/'+self.desc_name+'/'+self.mode
            print(dataset_path)
            dataset = Sequence(dataset_path, dump_dir, self.desc_name, self.vis_th, self.pair_num, pair_name)
            print('dump intermediate files.')
            dataset.dump_intermediate()
            print('dump matches.')
            dataset.dump_datasets()
        print('collect pkl.')
        self.collect()
        
