import numpy as np
import cv2
import tensorflow as tf


class Evaluator(object):
    def __init__(self, config):
        self.mutual_check = True
        self.err_thld = config['err_thld']
        self.matches = self.bf_matcher_graph()
        self.stats = {
            'i_eval_stats': np.array((0, 0, 0, 0, 0, 0, 0, 0), np.float32),
            'v_eval_stats': np.array((0, 0, 0, 0, 0, 0, 0, 0), np.float32),
            'all_eval_stats': np.array((0, 0, 0, 0, 0, 0, 0, 0), np.float32),
        }

    def homo_trans(self, coord, H):
        kpt_num = coord.shape[0]
        homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
        proj_coord = np.matmul(H, homo_coord.T).T
        proj_coord = proj_coord / proj_coord[:, 2][..., None]
        proj_coord = proj_coord[:, 0:2]
        return proj_coord

    def bf_matcher_graph(self):
        descriptors_a = tf.compat.v1.placeholder(tf.float32, (None, None), 'descriptor_a')
        descriptors_b = tf.compat.v1.placeholder(tf.float32, (None, None), 'descriptor_b')
        sim = tf.linalg.matmul(descriptors_a, descriptors_b, transpose_b=True)
        ids1 = tf.range(0, tf.shape(sim)[0])
        nn12 = tf.math.argmax(sim, axis=1, output_type=tf.int32)
        if self.mutual_check:
            nn21 = tf.math.argmax(sim, axis=0, output_type=tf.int32)
            mask = tf.equal(ids1, tf.gather(nn21, nn12))
            matches = tf.stack([tf.boolean_mask(ids1, mask), tf.boolean_mask(nn12, mask)])
        else:
            matches = tf.stack([ids1, nn12])
        return matches

    def mnn_matcher(self, sess, descriptors_a, descriptors_b):
        input_dict = {
            "descriptor_a:0": descriptors_a,
            "descriptor_b:0": descriptors_b
        }
        matches = sess.run(self.matches, input_dict)
        return matches.T

    def feature_matcher(self, sess, ref_feat, test_feat):
        matches = self.mnn_matcher(sess, ref_feat, test_feat)
        matches = [cv2.DMatch(matches[i][0], matches[i][1], 0) for i in range(matches.shape[0])]
        return matches

    def get_covisible_mask(self, ref_coord, test_coord, ref_img_shape, test_img_shape, gt_homo, scaling=1.):
        ref_coord = ref_coord / scaling
        test_coord = test_coord / scaling

        proj_ref_coord = self.homo_trans(ref_coord, gt_homo)
        proj_test_coord = self.homo_trans(test_coord, np.linalg.inv(gt_homo))

        ref_mask = np.logical_and(
            np.logical_and(proj_ref_coord[:, 0] < test_img_shape[1] - 1,
                           proj_ref_coord[:, 1] < test_img_shape[0] - 1),
            np.logical_and(proj_ref_coord[:, 0] > 0, proj_ref_coord[:, 1] > 0)
        )

        test_mask = np.logical_and(
            np.logical_and(proj_test_coord[:, 0] < ref_img_shape[1] - 1,
                           proj_test_coord[:, 1] < ref_img_shape[0] - 1),
            np.logical_and(proj_test_coord[:, 0] > 0, proj_test_coord[:, 1] > 0)
        )

        return ref_mask, test_mask

    def get_inlier_matches(self, ref_coord, test_coord, putative_matches, gt_homo, scaling=1.):
        p_ref_coord = np.float32([ref_coord[m.queryIdx] for m in putative_matches]) / scaling
        p_test_coord = np.float32([test_coord[m.trainIdx] for m in putative_matches]) / scaling

        proj_p_ref_coord = self.homo_trans(p_ref_coord, gt_homo)
        dist = np.sqrt(np.sum(np.square(proj_p_ref_coord - p_test_coord[:, 0:2]), axis=-1))
        inlier_mask = dist <= self.err_thld
        inlier_matches = [putative_matches[z] for z in np.nonzero(inlier_mask)[0]]
        return inlier_matches

    def get_gt_matches(self, ref_coord, test_coord, gt_homo, scaling=1.):
        ref_coord = ref_coord / scaling
        test_coord = test_coord / scaling
        proj_ref_coord = self.homo_trans(ref_coord, gt_homo)

        pt0 = np.expand_dims(proj_ref_coord, axis=1)
        pt1 = np.expand_dims(test_coord, axis=0)
        norm = np.linalg.norm(pt0 - pt1, ord=None, axis=2)
        min_dist0 = np.min(norm, axis=1)
        min_dist1 = np.min(norm, axis=0)
        gt_num0 = np.sum(min_dist0 <= self.err_thld)
        gt_num1 = np.sum(min_dist1 <= self.err_thld)
        gt_num = (gt_num0 + gt_num1) / 2
        return gt_num

    def compute_homography_accuracy(self, ref_coord, test_coord, ref_img_shape, putative_matches, gt_homo, scaling=1.):
        ref_coord = np.float32([ref_coord[m.queryIdx] for m in putative_matches]) / scaling
        test_coord = np.float32([test_coord[m.trainIdx] for m in putative_matches]) / scaling

        pred_homo, _ = cv2.findHomography(ref_coord, test_coord, cv2.RANSAC)
        if pred_homo is None:
            correctness = 0
        else:
            corners = np.array([[0, 0],
                                [ref_img_shape[1] / scaling - 1, 0],
                                [0, ref_img_shape[0] / scaling - 1],
                                [ref_img_shape[1] / scaling - 1, ref_img_shape[0] / scaling - 1]])
            real_warped_corners = self.homo_trans(corners, gt_homo)
            warped_corners = self.homo_trans(corners, pred_homo)
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            correctness = float(mean_dist <= self.err_thld)
        return correctness

    def print_stats(self, key):
        avg_stats = self.stats[key] / max(self.stats[key][0], 1)
        avg_stats = avg_stats[1:]
        print('----------%s----------' % key)
        print('avg_n_feat', int(avg_stats[0]))
        print('avg_rep', avg_stats[1])
        print('avg_precision', avg_stats[2])
        print('avg_matching_score', avg_stats[3])
        print('avg_recall', avg_stats[4])
        print('avg_MMA', avg_stats[5])
        print('avg_homography_accuracy', avg_stats[6])