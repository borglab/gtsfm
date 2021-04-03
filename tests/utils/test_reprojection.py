import numpy as np
from gtsam import Cal3Bundler, Rot3, PinholeCameraCal3Bundler, Pose3, SfmTrack

from gtsfm.common.image import Image
from gtsfm.common.sfm_track import SfmMeasurement
import gtsfm.utils.reprojection as reproj_utils


def test_compute_track_reprojection_errors():
    """Ensure that reprojection error is computed properly within a track.

    # For camera 0:
    # [13] = [10,0,3]   [1,0,0 | 0]  [1]
    # [24] = [0,10,4] * [0,1,0 | 0] *[2]
    #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
    #                                [1]

    # For camera 1:
    # [-7] = [10,0,3]   [1,0,0 |-2]  [1]
    # [44] = [0,10,4] * [0,1,0 | 2] *[2]
    #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
    #                                [1]
    """
    wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), np.zeros((3, 1)))
    wTi1 = Pose3(Rot3.RzRyRx(0, 0, 0), np.array([2, -2, 0]))

    f = 10
    k1 = 0
    k2 = 0
    u0 = 3
    v0 = 4

    K0 = Cal3Bundler(f, k1, k2, u0, v0)
    K1 = Cal3Bundler(f, k1, k2, u0, v0)

    track_camera_dict = {0: PinholeCameraCal3Bundler(wTi0, K0), 1: PinholeCameraCal3Bundler(wTi1, K1)}

    triangulated_pt = np.array([1, 2, 1])
    track_3d = SfmTrack(triangulated_pt)

    # in camera 0
    track_3d.add_measurement(idx=0, m=np.array([13, 24]))
    # in camera 1
    track_3d.add_measurement(idx=1, m=np.array([-8, 43]))  # should be (-7,44), 1 px error in each dim

    errors, avg_track_reproj_error = reproj_utils.compute_track_reprojection_errors(track_camera_dict, track_3d)

    expected_errors = np.array([0, np.sqrt(2)])
    np.testing.assert_allclose(errors, expected_errors)
    assert avg_track_reproj_error == np.sqrt(2) / 2


def test_compute_point_reprojection_errors():
    """Ensure a hypothesized 3d point is projected correctly and compared w/ 2 measurements.
    # For camera 0:
    # [13] = [10,0,3]   [1,0,0 | 0]  [1]
    # [24] = [0,10,4] * [0,1,0 | 0] *[2]
    #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
    #                                [1]

    # For camera 1:
    # [-7] = [10,0,3]   [1,0,0 |-2]  [1]
    # [44] = [0,10,4] * [0,1,0 | 2] *[2]
    #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
    #                                [1]
    """
    wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), np.zeros((3, 1)))
    wTi1 = Pose3(Rot3.RzRyRx(0, 0, 0), np.array([2, -2, 0]))

    f = 10
    k1 = 0
    k2 = 0
    u0 = 3
    v0 = 4

    K0 = Cal3Bundler(f, k1, k2, u0, v0)
    K1 = Cal3Bundler(f, k1, k2, u0, v0)

    track_camera_dict = {0: PinholeCameraCal3Bundler(wTi0, K0), 1: PinholeCameraCal3Bundler(wTi1, K1)}
    point3d = np.array([1, 2, 1])
    measurements = [SfmMeasurement(i=1, uv=np.array([-8, 43])), SfmMeasurement(i=0, uv=np.array([13, 24]))]

    errors, avg_track_reproj_error = reproj_utils.compute_point_reprojection_errors(
        track_camera_dict, point3d, measurements
    )
    expected_errors = np.array([np.sqrt(2), 0])
    np.testing.assert_allclose(errors, expected_errors)
    assert avg_track_reproj_error == np.sqrt(2) / 2


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

    r, g, b = reproj_utils.get_average_point_color(track_3d, images)
    assert r == 50
    assert g == 60
    assert b == 70
