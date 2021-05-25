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


def test_get_downsampling_factor_per_axis_leaveintact() -> None:
    """Ensure that image is left intact, when shorter side is smaller than max_resolution."""
    img_h = 700
    img_w = 1500
    img = Image(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    max_resolution = 800
    scale_u, scale_v, new_h, new_w = image_utils.get_downsampling_factor_per_axis(img_h, img_w, max_resolution)

    assert scale_u == 1.0
    assert scale_v == 1.0
    assert new_h == 700
    assert new_w == 1500


def test_get_rescaling_factor_per_axis_upsample() -> None:
    """Ensure that max resolution constraint is met, when upsampling image.

    Resize a 700x1500 image, so that the shorter image side is EXACTLY 800 px.
    """
    img_h = 700
    img_w = 1500
    img = Image(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    max_resolution = 800
    scale_u, scale_v, new_h, new_w = image_utils.get_rescaling_factor_per_axis(img_h, img_w, max_resolution)
    
    # 6/7 will not give a clean integer division
    assert np.isclose(scale_u, 1.1427, atol=4)
    assert np.isclose(scale_v, 1.1429, atol=4)
    assert new_h == 800
    assert new_w == 1714


def test_get_downsampling_factor_per_axis() -> None:
    """Ensure that max resolution constraint is met, when downsampling image.

    Resize a 700x1500 image, so that the shorter image side is AT MOST 600 px.
    Image is in landscape mode.
    """
    img_h = 700
    img_w = 1500
    img = Image(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    max_resolution = 600
    scale_u, scale_v, new_h, new_w = image_utils.get_downsampling_factor_per_axis(img_h, img_w, max_resolution)

    # Note that 600 / 700 = 0.85714
    # 1500 * 0.85714 = 1285.7, which we round up to 1286.
    assert np.isclose(scale_u, 0.8573, atol=4)
    assert np.isclose(scale_v, 0.8571, atol=4)
    assert new_h == 600
    assert new_w == 1286


def test_get_rescaling_factor_per_axis_downsample() -> None:
    """Ensure that max resolution constraint is met, when downsampling image.

    Resize a 700x1500 image, so that the shorter image side is EXACTLY 600 px.
    Image is in andscape mode.
    """
    img_h = 700
    img_w = 1500
    img = Image(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    max_resolution = 600
    scale_u, scale_v, new_h, new_w = image_utils.get_rescaling_factor_per_axis(img_h, img_w, max_resolution)

    # Note that 600 / 700 = 0.85714
    # 1500 * 0.85714 = 1285.7, which we round up to 1286.
    assert np.isclose(scale_u, 0.8573, atol=4)
    assert np.isclose(scale_v, 0.8571, atol=4)
    assert new_h == 600
    assert new_w == 1286




def test_get_downsampling_factor_per_axis_portrait() -> None:
    """Ensure that max resolution constraint is met, when downsampling image.

    Resize a 700x1500 image, so that the shorter image side is AT MOST 600 px.
    Image is in portrait mode.
    """
    img_h = 1500 
    img_w = 700
    img = Image(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    max_resolution = 600
    scale_u, scale_v, new_h, new_w = image_utils.get_downsampling_factor_per_axis(img_h, img_w, max_resolution)

    # Note that 600 / 700 = 0.85714
    # 1500 * 0.85714 = 1285.7, which we round up to 1286.
    assert np.isclose(scale_u, 0.8571, atol=4)
    assert np.isclose(scale_v, 0.8573, atol=4)
    assert new_h == 1286
    assert new_w == 600


def test_get_rescaling_factor_per_axis_downsample_portrait() -> None:
    """Ensure that max resolution constraint is met, when downsampling image.

    Resize a 700x1500 image, so that the shorter image side is EXACTLY 600 px.
    Image is in portrait mode.
    """
    img_h = 1500
    img_w = 700
    img = Image(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    max_resolution = 600
    scale_u, scale_v, new_h, new_w = image_utils.get_rescaling_factor_per_axis(img_h, img_w, max_resolution)

    # Note that 600 / 700 = 0.85714
    # 1500 * 0.85714 = 1285.7, which we round up to 1286.
    assert np.isclose(scale_v, 0.8571, atol=4)
    assert np.isclose(scale_u, 0.8573, atol=4)
    assert new_h == 1286
    assert new_w == 600
