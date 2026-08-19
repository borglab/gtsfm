"""Correspondence generator which reads from a Colmap db.

References:
- Colmap github
- Pycolmap github


Authors: Ayush Baid
"""

import os
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
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


class ColmapCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Load correspondences from Colmap DB."""

    # The matches read from `two_view_geometries` are already geometrically verified, so the verified
    # pipeline can use them directly and skip its Dask two-view estimation + gather.
    produces_verified_correspondences: bool = True

    def __init__(self, database_path: str) -> None:
        """Initialize the correspondence generator with the Colmap DB.

        Args:
            database_path: path of the Colmap DB.
        """
        self._database_path = database_path
        self._open_db()

    def _open_db(self) -> None:
        """Open the pycolmap db handle and load all keypoints (on construction and after unpickle)."""
        self._pycolmap_db = pycolmap.Database(self._database_path)
        # Note(Ayush): using SQLite3 to load keypoints because PyColmap does not expose bindings.
        raw_db = sqlite3.connect(self._database_path)
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

    def __getstate__(self) -> dict:
        """Drop the unpicklable pycolmap.Database (and the large, re-derivable keypoints cache) so this
        generator can be embedded in a Dask task graph. The colmap-db frontend runs in the MAIN process
        (reuse_global_correspondences), so a worker copy never touches the db; _ensure_db re-opens it
        lazily if a method is ever actually called on a worker."""
        state = self.__dict__.copy()
        state["_pycolmap_db"] = None
        state["_keypoints_dict"] = None
        state["_image_by_basename"] = None
        return state

    def _ensure_db(self) -> None:
        """Re-open the db if this instance was unpickled (e.g. shipped to a Dask worker) with no handle."""
        if getattr(self, "_pycolmap_db", None) is None:
            self._open_db()

    def _image_by_name(self, file_name: str) -> "pycolmap.Image":
        """DB image lookup tolerating path-prefixed db names (mirrors ColmapDBRetriever).

        DBs built with --image_path above the images dir store names like
        'Scene/images/x.jpg' while loaders serve basenames; exact read_image()
        then yields an invalid image_id (uint32 -1) that crashes pycolmap reads.
        """
        if getattr(self, "_image_by_basename", None) is None:
            self._image_by_basename = {
                os.path.basename(img.name): img for img in self._pycolmap_db.read_all_images()
            }
        img = self._image_by_basename.get(os.path.basename(file_name))
        if img is None:
            raise ValueError(f"Image {file_name!r} not found in COLMAP database (by basename).")
        return img

    def _read_keypoints(self, image: Image) -> Keypoints:
        """
        Read keypoints from a pycolmap.Image object.

        Args:
            image (Image): Input image object.

        Returns:
            Keypoints: Keypoints with their coordinates, scales, and responses.
        """
        pycolmap_image = self._image_by_name(image.file_name)
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

        pycolmap_images = [self._image_by_name(file_name) for file_name in file_names]

        keypoints: List[Keypoints] = [self._read_keypoints(image) for image in images]
        gtsfm_id_to_pycolmap_id: List[int] = [image.image_id for image in pycolmap_images]

        return gtsfm_id_to_pycolmap_id, keypoints

    def _read_matches(
        self, image_pairs: VisibilityGraph, gtsfm_id_to_pycolmap_id: List[int]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Read matches for image pairs."""
        corr_idxs: Dict[Tuple[int, int], np.ndarray] = {}
        for i1, i2 in image_pairs:
            colmap_i1 = gtsfm_id_to_pycolmap_id[i1]
            colmap_i2 = gtsfm_id_to_pycolmap_id[i2]

            two_view_geometry = self._pycolmap_db.read_two_view_geometry(colmap_i1, colmap_i2)

            # Accept E (2), F (3), and homography-type (4-6) verifications — matches the
            # ColmapDBRetriever's admission so same-graph runs consume GLOMAP's full diet.
            if two_view_geometry.config not in (2, 3, 4, 5, 6):
                continue

            # Note(Ayush): the matches we are loading are actually post verification
            corr_idxs[(i1, i2)] = np.array(two_view_geometry.inlier_matches, dtype=np.int32)

        return corr_idxs

    def generate_correspondences(
        self, client: Client, images: List[Future], visibility_graph: VisibilityGraph
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            visibility_graph: The visibility graph defining which image pairs to process.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """
        self._ensure_db()  # re-open the db if this generator was unpickled onto a worker
        # Note: we will end up reading verified correspondences from the colmap DB.
        images_actual = client.gather(images)

        gtsfm_id_to_pycolmap_id, keypoints = self._read_image_ids_and_keypoints(images_actual)
        corr_idxs = self._read_matches(visibility_graph, gtsfm_id_to_pycolmap_id)

        return keypoints, corr_idxs
