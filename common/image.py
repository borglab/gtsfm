"""Class for holding an image and its associated data.

Authors: Ayush Baid
"""

from typing import Optional, Tuple

import numpy as np

from utils.sensor_width_database import SensorWidthDatabase


class Image:
    """Holds the image and associated exif data."""

    sensor_width_db = SensorWidthDatabase()

    def __init__(self, image_array: np.ndarray, exif_data=None) -> None:
        self.image_array = image_array
        self.exif_data = exif_data

    @property
    def shape(self) -> Tuple[int, int]:
        """
        The shape of the image, with the horizontal direction (x coordinate) being the first entry

        Returns:
            Tuple[int, int]: shape of the image
        """
        return self.image_array.shape[1::-1]

    def get_intrinsics_from_exif(self) -> Optional[np.ndarray]:
        """Constructs the camera intrinsics from exif tag.

        Equation: focal_px=max(w_px,h_px)∗focal_mm / ccdw_mm

        Ref: 
        - https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
        - https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
        - https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from

        Returns:
            np.ndarray: intrinsics matrix (3x3).
        """

        if self.exif_data is None:
            return None

        focal_length_mm = self.exif_data.get('FocalLength')

        sensor_width_mm = Image.sensor_width_db.lookup(
            self.exif_data.get('Make'),
            self.exif_data.get('Model'),
        )

        img_w_px, img_h_px = self.image_array.shape[:2]
        focal_length_px = max(img_h_px, img_w_px) * \
            focal_length_mm/sensor_width_mm

        center_x = img_w_px.shape[0]/2
        center_y = img_h_px.shape[1]/2

        return np.array([
            [focal_length_px, 0, center_x],
            [0, focal_length_px, center_y],
            [0, 0, 1]
        ])
