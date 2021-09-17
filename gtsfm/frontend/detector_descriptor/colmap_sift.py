"""Hacky detector descriptor which imports colmap's features.

Authors: Ayush Baid
"""
import sqlite3
from typing import Tuple

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


COLMAP_DB_PATH = "/home/ayush/colmap_comparison/db.db"

KEYPOINTS_TABLE_NAME = "keypoints"
DESCRIPTORS_TABLE_NAME = "descriptors"
IMAGES_TABLE_NAME = "images"


class ColmapSIFT(DetectorDescriptorBase):
    def __init__(self):
        super().__init__()

        con = sqlite3.connect("/home/ayush/colmap_comparison/db.db")
        cur = con.cursor()

        image_id_to_name_dict = {}
        for (img_idx, img_name) in cur.execute("SELECT image_id, name FROM {}".format(IMAGES_TABLE_NAME)):
            image_id_to_name_dict[img_idx] = img_name

        self.keypoints_dict = {}
        for (img_idx, num_rows, num_cols, binary_blob) in cur.execute("SELECT * from {}".format(KEYPOINTS_TABLE_NAME)):
            print(img_idx, num_rows, num_cols)
            np_array = np.frombuffer(binary_blob, dtype=np.float32).reshape(num_rows, num_cols)
            keypoints = Keypoints(coordinates=np_array[: self.max_keypoints, :2])

            self.keypoints_dict[image_id_to_name_dict[img_idx]] = keypoints

        self.descriptors_dict = {}
        for (img_idx, num_rows, _, binary_blob) in cur.execute("SELECT * from {}".format(DESCRIPTORS_TABLE_NAME)):
            print(img_idx, num_rows)
            descriptors = np.frombuffer(binary_blob, dtype=np.uint8).reshape(num_rows, 128)
            self.descriptors_dict[image_id_to_name_dict[img_idx]] = descriptors[: self.max_keypoints]

        con.close()

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        keypoints = self.keypoints_dict[image.file_name]
        descriptors = self.descriptors_dict[image.file_name]

        assert np.all(keypoints.get_x_coordinates() >= 0)
        assert np.all(keypoints.get_x_coordinates() <= image.width)
        assert np.all(keypoints.get_y_coordinates() >= 0)
        assert np.all(keypoints.get_y_coordinates() <= image.height)

        return keypoints, descriptors
