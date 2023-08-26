"""Cacher for any direct image matcher, which caches the output on disk in the top level folder `cache`.

This class provides the caching functionality to a GTSFM image-based matcher. To use this cacher, initialize
it with the image matcher obj you want to apply the cache on.

Authors: John Lambert
"""
from pathlib import Path
from typing import Optional, Tuple

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase

logger = logger_utils.get_logger()

CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "cache"


class ImageMatcherCacher(ImageMatcherBase):
    """Cacher for matcher output on disk, keyed on the input.

    Unlike the MatcherCacher, the ImageMatcherCacher accepts 2 images as input.
    """

    def __init__(self, matcher_obj: ImageMatcherBase) -> None:
        super().__init__()
        self._matcher = matcher_obj
        self._matcher_obj_key = type(self._matcher).__name__

    def __repr__(self) -> str:
        return f"""
        ImageMatcherCacher:
           {self._matcher}
        """

    def _get_cache_path(self, cache_key: str) -> Path:
        """Gets the file path to the cache bz2 file from the cache key."""
        return CACHE_ROOT_PATH / "image_matcher" / "{}.pbz2".format(cache_key)

    def _generate_cache_key(self, image_i1: Image, image_i2: Image) -> str:
        """Generates the cache key from the two input images."""
        input_key_i1 = cache_utils.generate_hash_for_image(image_i1)
        input_key_i2 = cache_utils.generate_hash_for_image(image_i2)

        return "{}_{}_{}".format(self._matcher_obj_key, input_key_i1, input_key_i2)

    def _load_result_from_cache(self, image_i1: Image, image_i2: Image) -> Optional[Tuple[Keypoints, Keypoints]]:
        """Load cached result, if it exists. The cached result will be a 2D numpy array with 2 columns."""
        cache_path = self._get_cache_path(cache_key=self._generate_cache_key(image_i1=image_i1, image_i2=image_i2))
        cached_data = io_utils.read_from_bz2_file(cache_path)
        if cached_data is None:
            return None
        return cached_data["keypoints_i1"], cached_data["keypoints_i2"]

    def _save_result_to_cache(
        self, image_i1: Image, image_i2: Image, keypoints_i1: Keypoints, keypoints_i2: Keypoints
    ) -> None:
        """Save the results (corresponding keypoints) to the cache."""
        cache_path = self._get_cache_path(cache_key=self._generate_cache_key(image_i1=image_i1, image_i2=image_i2))
        data = {"keypoints_i1": keypoints_i1, "keypoints_i2": keypoints_i2}
        io_utils.write_to_bz2_file(data, cache_path)

    def match(
        self,
        image_i1: Image,
        image_i2: Image,
    ) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        If the results are in the cache, they are fetched and returned. Otherwise, the `match()` of the
        underlying object's API is called and the results are cached.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
        """
        cached_data = self._load_result_from_cache(
            image_i1=image_i1,
            image_i2=image_i2,
        )

        if cached_data is not None:
            return cached_data

        keypoints_i1, keypoints_i2 = self._matcher.match(
            image_i1=image_i1,
            image_i2=image_i2,
        )

        self._save_result_to_cache(
            image_i1=image_i1, image_i2=image_i2, keypoints_i1=keypoints_i1, keypoints_i2=keypoints_i2
        )
        return keypoints_i1, keypoints_i2
