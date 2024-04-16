"""Correspondence generator which reads from a Colmap db.

References: 
- Colmap github
- Pycolmap github


Authors: Ayush Baid
"""

import typing as T

import numpy as np
import pycolmap
from dask.distributed import Future
from distributed import Client

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase

logger = logger_utils.get_logger()


class ColmapCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Load correspondences from Colmap DB."""

    def __init__(self, database_path: str) -> None:
        self._db = pycolmap.Database(database_path)
        logger.info(
            "Loaded colmap db with %d images, %d keypoints, and %d verified pairs",
            self._db.num_images,
            self._db.num_keypoints,
            self._db.num_verified_image_pairs,
        )

    def _read_keypoints_from_image(self, image: pycolmap.Image) -> Keypoints:
        py_keypoints: T.List[np.ndarray] = image.points2D
        return Keypoints(coordinates=np.array(py_keypoints, dtype=np.float32), scales=None, responses=None)

    def _read_images(self, images: T.List[Image]) -> T.Tuple[T.List[int], T.List[Keypoints]]:
        file_names = [image.file_name for image in images if image.file_name is not None]
        if len(file_names) != len(images):
            raise ValueError("All images should be associated with a file name for ColmapCorrespondenceGenerator")
        pycolmap_images: T.List[pycolmap.Image] = [self._db.read_image_with_name(file_name) for file_name in file_names]

        keypoints: T.List[Keypoints] = [self._read_keypoints_from_image(image) for image in pycolmap_images]
        gtsfm_id_to_pycolmap_id: T.List[int] = [image.image_id for image in pycolmap_images]

        return gtsfm_id_to_pycolmap_id, keypoints

    def _read_matches(
        self, image_pairs: T.List[T.Tuple[int, int]], gtsfm_id_to_pycolmap_id: T.List[int]
    ) -> T.Dict[T.Tuple[int, int], np.ndarray]:
        corr_idxs: T.Dict[T.Tuple[int, int], np.ndarray] = {}
        for i1, i2 in image_pairs:
            colmap_i1 = gtsfm_id_to_pycolmap_id[i1]
            colmap_i2 = gtsfm_id_to_pycolmap_id[i2]

            two_view_geometry = self._db.read_two_view_geometry(colmap_i1, colmap_i2)

            # Only read matches if we have an essential or a fundamental matrix
            if two_view_geometry.config != 2 and two_view_geometry.config != 3:
                continue

            # Note(Ayush): the matches we are loading are actually post verification
            corr_idxs[(i1, i2)] = np.array(two_view_geometry.inlier_matches, dtype=np.int32)

        return corr_idxs

    def generate_correspondences(
        self, client: Client, images: T.List[Future], image_pairs: T.List[T.Tuple[int, int]]
    ) -> T.Tuple[T.List[Keypoints], T.Dict[T.Tuple[int, int], np.ndarray]]:
        images_actual = client.gather(images)

        gtsfm_id_to_pycolmap_id, keypoints = self._read_images(images_actual)
        corr_idxs = self._read_matches(image_pairs, gtsfm_id_to_pycolmap_id)

        return keypoints, corr_idxs
