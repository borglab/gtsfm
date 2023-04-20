"""Class for holding an image and its associated data.

Authors: Ayush Baid
"""

from typing import Any, Dict, NamedTuple, Optional

import numpy as np
from gtsam import Cal3Bundler

from gtsfm.common.sensor_width_database import SensorWidthDatabase


class Image(NamedTuple):
    """Holds the image, associated exif data, and original image file name."""

    value_array: np.ndarray
    exif_data: Optional[Dict[str, Any]] = None
    sensor_width_db = SensorWidthDatabase()
    file_name: Optional[str] = None
    mask: Optional[np.ndarray] = None

    @property
    def height(self) -> int:
        """
        The height of the image (i.e. number of pixels in the vertical direction).
        """
        return self.value_array.shape[0]

    @property
    def width(self) -> int:
        """
        The width of the image (i.e. number of pixels in the horizontal direction).
        """
        return self.value_array.shape[1]

    @property
    def shape(self) -> tuple:
        """
        The shape of the image (H, W, C).
        """
        return self.value_array.shape

    def get_intrinsics_from_exif(self) -> Optional[Cal3Bundler]:
        """Constructs the camera intrinsics from exif tag.

        Equation: focal_px=max(w_px,h_px)∗focal_mm / ccdw_mm

        Ref:
        - https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
        - https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
        - https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from # noqa: E501

        Returns:
            intrinsics matrix (3x3).
        """

        if self.exif_data is None or len(self.exif_data) == 0:
            return None

        focal_length_mm = self.exif_data.get("FocalLength")

        if focal_length_mm is None:
            return None

        try:
            sensor_width_mm = Image.sensor_width_db.lookup(
                self.exif_data.get("Make"),
                self.exif_data.get("Model"),
            )
        except (AttributeError, LookupError):
            # TODO(yanwei.du): Set a prior focal length, e.g. 1.5 * max(self.width, self.height).
            return None

        img_w_px = self.width
        img_h_px = self.height
        focal_length_px = max(img_h_px, img_w_px) * focal_length_mm / sensor_width_mm

        center_x = img_w_px / 2
        center_y = img_h_px / 2

        return Cal3Bundler(
            fx=float(focal_length_px),
            k1=0.0,
            k2=0.0,
            u0=float(center_x),
            v0=float(center_y),
        )

    def extract_patch(self, center_x: float, center_y: float, patch_size: int) -> "Image":
        """Extracts a square patch from the image.

        Note: appropriate padding is done if patch is out of bounds.

        Args:
            center_x: horizontal coordinate of the patch center.
            center_y: vertical coordinate of the patch center.
            patch_size: edge length of the patch.

        Returns:
            Image: extracted patch.
        """

        center_x = int(round(center_x))
        center_y = int(round(center_y))

        if center_x < 0 or center_x >= self.width:
            raise ValueError("patch center should be in the image")

        if center_y < 0 or center_y >= self.height:
            raise ValueError("patch center should be in the image")

        # pad the whole image to take care of boundary conditions
        len_left = patch_size // 2  # 20 -> 10, 21 -> 10
        len_right = (patch_size - 1) // 2  # 20 -> 9, 21 -> 10

        # apply computed padding on the spatial dimensions and zero padding on channel dimensions
        padded_value_array = np.pad(
            array=self.value_array,
            pad_width=((len_left, len_right), (len_left, len_right), (0, 0)),
        )

        # extract the values in the patch
        # Note: the padding amount and pad_size//2 cancel each other out, so
        # center index becomes the far left edge of patch in padded image
        patch_values = padded_value_array[center_y : center_y + patch_size, center_x : center_x + patch_size]

        return Image(value_array=patch_values, exif_data=None)
