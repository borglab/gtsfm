#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
HSequence evaluation tools.
"""

import os
import pickle
import random
import glob
import cv2
import numpy as np

from utils.opencvhelper import SiftWrapper


class HSeqData(object):
    def __init__(self):
        self.img = []
        self.patch = []
        self.kpt_param = []
        self.coord = []
        self.homo = []
        self.img_feat = []
        self.scaling = 1


class HSeqUtils(object):
    def __init__(self, config):
        self.seqs = []
        i_seqs = []
        v_seqs = []
        for seq in os.listdir(config['root']):
            if seq[0:2] == 'i_' and seq not in config['ignored_i']:
                i_seqs.append(os.path.join(config['root'], seq))
            if seq[0:2] == 'v_' and seq not in config['ignored_v']:
                v_seqs.append(os.path.join(config['root'], seq))
        i_seqs.sort()
        v_seqs.sort()

        if 'v' in config['seq']:
            self.seqs.extend(v_seqs)
        if 'i' in config['seq']:
            self.seqs.extend(i_seqs)

        self.seqs = self.seqs[config['start_idx']:]
        self.seq_num = len(self.seqs)
        # for detector config
        self.max_dim = config['max_dim']

    def get_data(self, seq_idx):
        hseq_data = HSeqData()
        seq_name = self.seqs[seq_idx]

        scaling = 1
        for img_idx in range(1, 7):
            # read images.
            img = cv2.imread(os.path.join(seq_name, '%d.ppm' % img_idx))
            # resize images if needed (regarding only the first image).
            if img_idx == 1:
                long_side = np.max(img.shape[0:2])
                if long_side > self.max_dim and self.max_dim > 0:
                    scaling = self.max_dim / long_side
            if scaling != 1:
                img = cv2.resize(img, (0, 0), fx=scaling, fy=scaling)
            # read homography matrix.
            if img_idx > 1:
                homo_mat = open(os.path.join(seq_name, 'H_1_%d' % img_idx)).read().splitlines()
                homo_mat = np.array([float(i) for i in ' '.join(homo_mat).split()])
                homo_mat = np.reshape(homo_mat, (3, 3))
            else:
                homo_mat = None

            hseq_data.img.append(img)
            hseq_data.homo.append(homo_mat)
        hseq_data.scaling = scaling

        return seq_name, hseq_data
