"""Class for holding an image and its associated data.

Authors: Ayush Baid
"""

from typing import NamedTuple, Optional, Tuple, Dict

import numpy as np
from gtsam import Cal3Bundler

from utils.sensor_width_database import SensorWidthDatabase


class Image(NamedTuple):
    value_array: np.ndarray
    exif_data: Dict = None
    """Holds the image and associated exif data."""

    sensor_width_db = SensorWidthDatabase()

    @property
    def height(self) -> int:
        """
        The height of the image (i.e. number of pixels in the vertical
        direction).
        """
        return self.value_array.shape[0]

    @property
    def width(self) -> int:
        """
        The width of the image (i.e. number of pixels in the horizontal
        direction).
        """
        return self.value_array.shape[1]

    def get_intrinsics_from_exif(self) -> Optional[Cal3Bundler]:
        """Constructs the camera intrinsics from exif tag.

        Equation: focal_px=max(w_px,h_px)∗focal_mm / ccdw_mm

        Ref: 
        - https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
        - https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
        - https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from

        Returns:
            intrinsics matrix (3x3).
        """

        if self.exif_data is None or len(self.exif_data) == 0:
            return None

        focal_length_mm = self.exif_data.get('FocalLength')

        sensor_width_mm = Image.sensor_width_db.lookup(
            self.exif_data.get('Make'),
            self.exif_data.get('Model'),
        )

        img_w_px, img_h_px = self.value_array.shape[:2]
        focal_length_px = max(img_h_px, img_w_px) * \
            focal_length_mm/sensor_width_mm

        center_x = img_w_px/2
        center_y = img_h_px/2

        return Cal3Bundler(
            fx=float(focal_length_px),
            k1=0.0,
            k2=0.0, u0=float(center_x),
            v0=float(center_y))
