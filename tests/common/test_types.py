"""Unit tests for types.py

Authors: Travis Driver
"""

import pytest
import gtsam

from gtsfm.common.types import (
    get_camera_class_for_calibration,
    get_camera_set_class_for_calibration,
    get_prior_factor_for_calibration,
    get_sfm_factor_for_calibration,
)


# Define the expected types for the calibration objects
CALIBRATION_OBJECTS = [
    (
        gtsam.Cal3Bundler(), 
        gtsam.PinholeCameraCal3Bundler, 
        gtsam.CameraSetCal3Bundler, gtsam.
        PriorFactorCal3Bundler, 
        gtsam.GeneralSFMFactor2Cal3Bundler,
    ),
    (
        gtsam.Cal3_S2(), 
        gtsam.PinholeCameraCal3_S2, 
        gtsam.CameraSetCal3_S2, 
        gtsam.PriorFactorCal3_S2, 
        gtsam.GeneralSFMFactor2Cal3_S2,
    ),
    (
        gtsam.Cal3DS2(), 
        gtsam.PinholeCameraCal3DS2, 
        gtsam.CameraSetCal3DS2, 
        gtsam.PriorFactorCal3DS2, 
        gtsam.GeneralSFMFactor2Cal3DS2,
    ),
    (
        gtsam.Cal3Fisheye(), 
        gtsam.PinholeCameraCal3Fisheye, 
        gtsam.CameraSetCal3Fisheye, 
        gtsam.PriorFactorCal3Fisheye, 
        gtsam.GeneralSFMFactor2Cal3Fisheye,
    ),
]

# Create specific parameter lists for each test
CAMERA_CLASS_PARAMS = [(c[0], c[1]) for c in CALIBRATION_OBJECTS]
CAMERA_SET_PARAMS = [(c[0], c[2]) for c in CALIBRATION_OBJECTS]
PRIOR_FACTOR_PARAMS = [(c[0], c[3]) for c in CALIBRATION_OBJECTS]
SFM_FACTOR_PARAMS = [(c[0], c[4]) for c in CALIBRATION_OBJECTS]

# List of invalid inputs remains the same
INVALID_INPUTS = [None, "a_string", 123, object(), gtsam.Pose2()]


class TestGtsamHelpers:
    """Test suite for GTSAM helper functions."""

    @pytest.mark.parametrize("calibration_obj, expected_class", CAMERA_CLASS_PARAMS)
    def test_get_camera_class_for_calibration_success(self, calibration_obj, expected_class):
        # The function signature is now clean and only contains used arguments
        assert get_camera_class_for_calibration(calibration_obj) is expected_class

    @pytest.mark.parametrize("calibration_obj, expected_class", CAMERA_SET_PARAMS)
    def test_get_camera_set_class_for_calibration_success(self, calibration_obj, expected_class):
        assert get_camera_set_class_for_calibration(calibration_obj) is expected_class

    @pytest.mark.parametrize("calibration_obj, expected_class", PRIOR_FACTOR_PARAMS)
    def test_get_prior_factor_for_calibration_success(self, calibration_obj, expected_class):
        assert get_prior_factor_for_calibration(calibration_obj) is expected_class

    @pytest.mark.parametrize("calibration_obj, expected_class", SFM_FACTOR_PARAMS)
    def test_get_sfm_factor_for_calibration_success(self, calibration_obj, expected_class):
        assert get_sfm_factor_for_calibration(calibration_obj) is expected_class

    # --- The failure tests can be grouped together ---
    @pytest.mark.parametrize(
        "helper_function",
        [
            get_camera_class_for_calibration,
            get_camera_set_class_for_calibration,
            get_prior_factor_for_calibration,
            get_sfm_factor_for_calibration,
        ],
    )
    @pytest.mark.parametrize("invalid_input", INVALID_INPUTS)
    def test_all_helpers_failure(self, helper_function, invalid_input):
        """Tests that all helper functions raise a ValueError for unsupported input types."""
        with pytest.raises(ValueError, match="Unsupported calibration type"):
            helper_function(invalid_input)
