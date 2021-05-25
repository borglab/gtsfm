import numpy as np
from gtsam import SfmTrack

from gtsfm.common.image import Image
import gtsfm.utils.images as image_utils


def test_get_average_point_color():
    """ Ensure 3d point color is computed as mean of RGB per 2d measurement."""
    # random point; 2d measurements below are dummy locations (not actual projection)
    triangulated_pt = np.array([1, 2, 1])
    track_3d = SfmTrack(triangulated_pt)

    # in camera 0
    track_3d.add_measurement(idx=0, m=np.array([130, 80]))
    # in camera 1
    track_3d.add_measurement(idx=1, m=np.array([10, 60]))

    img0 = np.zeros((100, 200, 3), dtype=np.uint8)
    img0[80, 130] = np.array([40, 50, 60])

    img1 = np.zeros((100, 200, 3), dtype=np.uint8)
    img1[60, 10] = np.array([60, 70, 80])

    images = {0: Image(img0), 1: Image(img1)}

    r, g, b = image_utils.get_average_point_color(track_3d, images)
    assert r == 50
    assert g == 60
    assert b == 70


def test_get_downsample_factor_per_axis() -> None:
    """Ensure that max resolution constraint is met, when downsampling image.

    Resize a 700x1500 image, so that the shorter image side is at most 600 px.
    """
    img = Image(np.zeros((700, 1500, 3), dtype=np.uint8))
    max_resolution = 600
    downsample_u, downsample_v, new_h, new_w = image_utils.get_downsample_factor_per_axis(img, max_resolution)

    # 6/7 will not give a clean integer division
    assert downsample_u == 1.167
    assert downsample_v == 1.166
    assert new_h == 600
    assert new_w == 1285
