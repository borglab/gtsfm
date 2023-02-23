#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Inference script.
"""

import os

from queue import Queue
from threading import Thread

import math
import yaml

import cv2
import numpy as np

import tensorflow as tf

from models import get_model
from utils.hseq_utils import HSeqUtils
from utils.evaluator import Evaluator

FLAGS = tf.compat.v1.app.flags.FLAGS

# general config.
tf.compat.v1.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def loader(hseq_utils, producer_queue):
    for seq_idx in range(hseq_utils.seq_num):
        seq_name, hseq_data = hseq_utils.get_data(seq_idx)

        for i in range(6):
            gt_homo = [seq_idx, seq_name, hseq_data.scaling] if i == 0 else hseq_data.homo[i]
            producer_queue.put([hseq_data.img[i], gt_homo])
    producer_queue.put(None)

def extractor(patch_queue, model, consumer_queue):
    while True:
        queue_data = patch_queue.get()
        if queue_data is None:
            consumer_queue.put(None)
            return
        img, gt_homo = queue_data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        descs, kpts, _ = model.run_test_data(np.expand_dims(gray, axis=-1))
        consumer_queue.put([img, kpts, descs, gt_homo])
        patch_queue.task_done()

def matcher(consumer_queue, sess, evaluator, config):
    record = []
    while True:
        queue_data = consumer_queue.get()
        if queue_data is None:
            return
        record.append(queue_data)
        if len(record) < 6:
            continue
        ref_img, ref_kpts, ref_descs, seq_info = record[0]

        eval_stats = np.array((0, 0, 0, 0, 0, 0, 0, 0), np.float32)

        seq_idx = seq_info[0]
        seq_name = seq_info[1]
        scaling = seq_info[2]
        print(seq_idx, seq_name)

        for i in range(1, 6):
            test_img, test_kpts, test_descs, gt_homo = record[i]
            # get MMA
            num_feat = min(ref_kpts.shape[0], test_kpts.shape[0])
            if num_feat > 0:
                mma_putative_matches = evaluator.feature_matcher(
                    sess, ref_descs, test_descs)
            else:
                mma_putative_matches = []
            mma_inlier_matches = evaluator.get_inlier_matches(
                ref_kpts, test_kpts, mma_putative_matches, gt_homo, scaling)
            num_mma_putative = len(mma_putative_matches)
            num_mma_inlier = len(mma_inlier_matches)
            # get covisible keypoints
            ref_mask, test_mask = evaluator.get_covisible_mask(ref_kpts, test_kpts,
                                                               ref_img.shape, test_img.shape,
                                                               gt_homo, scaling)
            cov_ref_coord, cov_test_coord = ref_kpts[ref_mask], test_kpts[test_mask]
            cov_ref_feat, cov_test_feat = ref_descs[ref_mask], test_descs[test_mask]
            num_cov_feat = (cov_ref_coord.shape[0] + cov_test_coord.shape[0]) / 2
            # get gt matches
            gt_num = evaluator.get_gt_matches(cov_ref_coord, cov_test_coord, gt_homo, scaling)
            # establish putative matches
            if num_cov_feat > 0:
                putative_matches = evaluator.feature_matcher(
                    sess, cov_ref_feat, cov_test_feat)
            else:
                putative_matches = []
            num_putative = max(len(putative_matches), 1)
            # get homography accuracy
            correctness = evaluator.compute_homography_accuracy(cov_ref_coord, cov_test_coord, ref_img.shape, putative_matches, gt_homo, scaling)
            # get inlier matches
            inlier_matches = evaluator.get_inlier_matches(
                cov_ref_coord, cov_test_coord, putative_matches, gt_homo, scaling)
            num_inlier = len(inlier_matches)

            eval_stats += np.array((1, # counter
                                    num_feat, # feature number
                                    gt_num / max(num_cov_feat, 1), # repeatability
                                    num_inlier / max(num_putative, 1), # precision
                                    num_inlier / max(num_cov_feat, 1), # matching score
                                    num_inlier / max(gt_num, 1),  # recall
                                    num_mma_inlier / max(num_mma_putative, 1),
                                    correctness)) / 5  # MMA

        print(int(eval_stats[1]), eval_stats[2:])
        evaluator.stats['all_eval_stats'] += eval_stats
        if os.path.basename(seq_name)[0] == 'i':
            evaluator.stats['i_eval_stats'] += eval_stats
        if os.path.basename(seq_name)[0] == 'v':
            evaluator.stats['v_eval_stats'] += eval_stats

        record = []

def hseq_eval():
    with open(FLAGS.config, 'r') as f:
        test_config = yaml.load(f, Loader=yaml.FullLoader)
    # Configure dataset
    hseq_utils = HSeqUtils(test_config['hseq'])
    # Configure evaluation
    evaluator = Evaluator(test_config['eval'])
    # Construct inference networks.
    model = get_model('feat_model')(test_config['model_path'], **(test_config['net']))
    # Create the initializier.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    producer_queue = Queue(maxsize=18)
    consumer_queue = Queue()

    producer0 = Thread(target=loader, args=(hseq_utils, producer_queue))
    producer0.daemon = True
    producer0.start()

    producer1 = Thread(target=extractor, args=(producer_queue, model, consumer_queue))
    producer1.daemon = True
    producer1.start()

    consumer = Thread(target=matcher, args=(consumer_queue, model.sess, evaluator, test_config['eval']))
    consumer.daemon = True
    consumer.start()

    producer0.join()
    producer1.join()
    consumer.join()

    evaluator.print_stats('i_eval_stats')
    evaluator.print_stats('v_eval_stats')
    evaluator.print_stats('all_eval_stats')

if __name__ == '__main__':
    tf.compat.v1.flags.mark_flags_as_required(['config'])
    hseq_eval()
