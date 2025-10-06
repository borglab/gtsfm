"""Correspondence generator which reads from a Colmap db.

References:
- Colmap github
- Pycolmap github


Authors: Ayush Baid
"""

import sqlite3
from typing import Dict, List, Tuple

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
        """Initialize the correspondence generator with the Colmap DB.

        Args:
            database_path: path of the Colmap DB.
        """
        self._pycolmap_db = pycolmap.Database(database_path)
        # Note(Ayush): using SQLite3 to load keypoints because PyColmap does not expose bindings.
        raw_db = sqlite3.connect(database_path)
        self._keypoints_dict: Dict[int, np.ndarray] = {
            image_id: np.frombuffer(data, dtype=np.float32).reshape(rows, -1)
            for image_id, rows, data in raw_db.execute("SELECT image_id, rows, data FROM keypoints")
        }
        raw_db.close()

        logger.info(
            "Loaded colmap db with %d images, %d keypoints, and %d verified pairs",
            self._pycolmap_db.num_images,
            self._pycolmap_db.num_keypoints,
            self._pycolmap_db.num_verified_image_pairs,
        )

    def _read_keypoints(self, image: Image) -> Keypoints:
        """
        Read keypoints from a pycolmap.Image object.

        Args:
            image (Image): Input image object.

        Returns:
            Keypoints: Keypoints with their coordinates, scales, and responses.
        """
        pycolmap_image = self._pycolmap_db.read_image(image.file_name)
        image_id = pycolmap_image.image_id

        if image_id not in self._keypoints_dict:
            return Keypoints(coordinates=np.array([], dtype=np.float32), scales=None, responses=None)

        coordinates = self._keypoints_dict[image_id][:, :2]
        camera = self._pycolmap_db.read_camera(pycolmap_image.camera_id)

        # Colmap extracts features in the downscaled image
        # but scales keypoints back to the original dimensions before storing in the database.
        if image.width != camera.width or image.height != camera.height:
            scale = np.array([image.width / camera.width, image.height / camera.height])
            scaled_coordinates = coordinates * scale
            return Keypoints(coordinates=scaled_coordinates, scales=None, responses=None)

        return Keypoints(coordinates=coordinates, scales=None, responses=None)

    def _read_image_ids_and_keypoints(self, images: List[Image]) -> Tuple[List[int], List[Keypoints]]:
        """
        Read image IDs and keypoints for the images.

        Args:
            images (List[Image]): List of input image objects.

        Returns:
            Tuple[List[int], List[Keypoints]]: A tuple containing the list of image IDs and the corresponding keypoints.

        Raises:
            ValueError: If any image lacks a file name.
        """
        file_names = [image.file_name for image in images if image.file_name is not None]

        if len(file_names) != len(images):
            raise ValueError("All images should be associated with a file name for ColmapCorrespondenceGenerator.")

        pycolmap_images = [self._pycolmap_db.read_image(file_name) for file_name in file_names]

        keypoints: List[Keypoints] = [self._read_keypoints(image) for image in images]
        gtsfm_id_to_pycolmap_id: List[int] = [image.image_id for image in pycolmap_images]

        return gtsfm_id_to_pycolmap_id, keypoints

    def _read_matches(
        self, image_pairs: List[Tuple[int, int]], gtsfm_id_to_pycolmap_id: List[int]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Read matches for image pairs."""
        corr_idxs: Dict[Tuple[int, int], np.ndarray] = {}
        for i1, i2 in image_pairs:
            colmap_i1 = gtsfm_id_to_pycolmap_id[i1]
            colmap_i2 = gtsfm_id_to_pycolmap_id[i2]

            two_view_geometry = self._pycolmap_db.read_two_view_geometry(colmap_i1, colmap_i2)

            # Only read matches if we have an essential or a fundamental matrix
            if two_view_geometry.config != 2 and two_view_geometry.config != 3:
                continue

            # Note(Ayush): the matches we are loading are actually post verification
            corr_idxs[(i1, i2)] = np.array(two_view_geometry.inlier_matches, dtype=np.int32)

        return corr_idxs

    def generate_correspondences(
        self, client: Client, images: List[Future], image_pairs: List[Tuple[int, int]]
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            image_pairs: Indices of the pairs of images to estimate two-view pose and correspondences.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """
        # Note: we will end up reading verified correspondences from the colmap DB.
        images_actual = client.gather(images)

        gtsfm_id_to_pycolmap_id, keypoints = self._read_image_ids_and_keypoints(images_actual)
        corr_idxs = self._read_matches(image_pairs, gtsfm_id_to_pycolmap_id)

        return keypoints, corr_idxs
