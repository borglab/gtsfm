"""
YFCC dataset loader.

Authors: Ayush Baid
"""

import glob
import os
from typing import Dict

import gtsam
import numpy as np

import utils.io as io_utils
from loader.loader_base import LoaderBase


def build_extrinsic_matrix(R, t):
    """Build extrinsic matrix
    Args:
        R (array): Rotation matrix of shape (3,3)
        t (array): Translation vector of shape (3,)
    Returns:
        array: extrinsic matrix

    Ref: https://github.com/chrischoy/open-ucn
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_essential_matrix(T0, T1):
    """Compute essential matrix
    Args:
        T0 (array): extrinsic matrix
        T1 (array): extrinsic matrix

    Returns:
        array: essential matrix

    Ref: https://github.com/chrischoy/open-ucn
    """
    rel_rotation = T1[:3, :3] @ (T0[:3, :3].T)
    rel_translation = T1[:3, 3] - np.matmul(rel_rotation, T0[:3, 3])

    E_mine = skew_symmetric(rel_translation) @ rel_rotation

    return E_mine


def skew_symmetric(t):
    """Compute skew symmetric matrix of vector t
    Args:
        t (np.ndarray): vector of shape (3,)
    Returns:
        M (np.ndarray): skew-symmetrix matrix of shape (3, 3)

        Ref: https://github.com/chrischoy/open-ucn
    """
    M = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return M


def scale_intrinsic(instrinsic_matrix, scale):
    final_entry = instrinsic_matrix[2, 2]

    instrinsic_matrix = instrinsic_matrix*scale

    instrinsic_matrix[2, 2] = final_entry

    return instrinsic_matrix


def generate_fundamental_matrix(K1, K2, R1, R2, t1, t2):
    """Parse serialiazed calibration
    Args:
    calib (np.ndarray): serialized calibration
    Returns:
    dict: parsed calibration

    Ref: https://github.com/chrischoy/open-ucn

    """

    essential_mat = compute_essential_matrix(
        build_extrinsic_matrix(R1, t1),
        build_extrinsic_matrix(R2, t2),
    )

    fundamental_mat = np.linalg.inv(K2.T) @ essential_mat @ np.linalg.inv(K1)

    return fundamental_mat


class YFCCLoader(LoaderBase):
    def __init__(self, base_dir, scene_tag, top_k=5, scale=0.6):

        # create the path for the raw and processed directories

        self.processed_dir = os.path.join(base_dir, 'processed')
        self.base_dir = base_dir
        self.scene_tag = scene_tag

        self.scale = scale

        # for each scene, limit the number of pairs which we return
        processed_files = sorted(glob.glob(
            os.path.join(self.processed_dir, self.scene_tag + '_*.npz')
        ))[:top_k]

        self.data = {}
        for x in processed_files:
            self.data.update(self.__load_processed_data(x))

        self.image_paths = set()
        for (path0, path1) in self.data.keys():
            self.image_paths.add(path0)
            self.image_paths.add(path1)

        # convert to list of paths
        self.image_paths = sorted(list(self.image_paths))

        idx_to_path_map = dict(
            zip(self.image_paths, range(len(self.image_paths)))
        )

        # re-map the data dictionary to indices insteads of paths
        self.data = {
            (idx_to_path_map.get(p1), idx_to_path_map.get(p2)): v for (p1, p2), v in self.data.items()
        }

    def __load_processed_data(self, file_path):
        data = np.load(file_path)

        info = {
            (data['img_path0'].item(), data['img_path1'].item()): (self.__parse_calib_data(data['calib0']), self.__parse_calib_data(data['calib1']))
        }

        return info

    def __parse_calib_data(self, calib) -> Dict:
        parsed_data = {}
        instrinsic = calib[:9].reshape((3, 3))
        instrinsic[0, 2] = calib[21]/2
        instrinsic[1, 2] = calib[22]/2
        parsed_data["K"] = scale_intrinsic(instrinsic, self.scale)
        parsed_data["R"] = calib[9:18].reshape((3, 3))
        parsed_data["t"] = calib[18:21].reshape(3)
        # parsed_data["imsize"] = calib[21:23].reshape(2) # scale it

        return parsed_data

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, index):
        return io_utils.load_image(
            os.path.join(self.base_dir, self.image_paths[index]),
            self.scale
        )

    def validate_pair(self, idx1, idx2):
        return (idx1, idx2) in self.data

    def get_fundamental_matrix(self, idx1, idx2):
        data_idx1, data_idx2 = self.data[(idx1, idx2)]
        return generate_fundamental_matrix(data_idx1['K'],
                                           data_idx2['K'],
                                           data_idx1['R'],
                                           data_idx2['R'],
                                           data_idx1['t'],
                                           data_idx2['t'],
                                           )

    def is_homography(self):
        return False

    def get_camera_instrinsics(self, idx1, idx2):
        data_idx1, data_idx2 = self.data[(idx1, idx2)]
        return data_idx1['K'], data_idx2['K']

    def get_relative_rotation(self, idx1: int, idx2: int) -> gtsam.Rot3:
        data_idx1, data_idx2 = self.data[(idx1, idx2)]
        relative_rotation = gtsam.Rot3(
            data_idx1['R']).between(gtsam.Rot3(data_idx2['R']))

        return relative_rotation

    def get_relative_pose(self, idx1: int, idx2: int) -> gtsam.Pose3:
        data_idx1, data_idx2 = self.data[(idx1, idx2)]
        return gtsam.Pose3(gtsam.Rot3(data_idx1['R']), gtsam.Point3(data_idx1['t'])).between(
            gtsam.Pose3(gtsam.Rot3(
                data_idx2['R']), gtsam.Point3(data_idx2['t']))
        )
