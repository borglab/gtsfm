"""PyTheia match verification.

Author: Travis Driver
"""
from typing import Optional, Tuple, List

import pytheia as pt
import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3, Pose3

import gtsfm.common.types as gtsfm_types
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.common.keypoints import Keypoints

RANSAC_SUCCESS_PROB = 0.999999
RANSAC_MAX_ITERS = 1000000


class PyTheiaVerifier(VerifierBase):
    def __theia_keypoints(
        self,
        keypoints: Keypoints,
    ) -> List[pt.matching.KeypointsAndDescriptors]:
        """Builds KeypointsAndDescriptors object to be used by Theia.

        Note: Descriptors are left empty, as they are not used during geometric verification.
        """
        keypoints_theia = pt.matching.KeypointsAndDescriptors()
        keypoints_theia.keypoints = [
            pt.matching.Keypoint(kp[0], kp[1], pt.matching.Keypoint.KeypointType(0)) for kp in keypoints.coordinates
        ]
        keypoints_theia.descriptors = np.zeros((keypoints.coordinates.shape[0], 128))

        return keypoints_theia

    def __theia_intrinsics(self, intrinsics: Optional[gtsfm_types.CALIBRATION_TYPE]) -> pt.sfm.CameraIntrinsicsPrior:
        assert isinstance(intrinsics, Cal3Bundler)

        prior = pt.sfm.CameraIntrinsicsPrior()
        prior.focal_length.value = [intrinsics.fx()]
        prior.aspect_ratio.value = [intrinsics.fy() / intrinsics.fx()]
        prior.principal_point.value = [intrinsics.px(), intrinsics.py()]
        prior.radial_distortion.value = [intrinsics.k1(), intrinsics.k2(), 0, 0]
        prior.tangential_distortion.value = [0, 0]
        prior.skew.value = [0]
        # TODO (travisdriver): unfix this
        # prior.image_width = int(760)
        # prior.image_height = int(1135)

        # 'PINHOLE_RADIAL_TANGENTIAL', 'DIVISION_UNDISTORTION', 'DOUBLE_SPHERE', 'FOV', 'EXTENDED_UNIFIED', 'FISHEYE
        prior.camera_intrinsics_model_type = "PINHOLE_RADIAL_TANGENTIAL"

        return prior

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
        """Performs verification of correspondences between two images to recover the relative pose and indices of
        verified correspondences.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
            Inlier ratio of w.r.t. the estimated model, i.e. the #final RANSAC inliers/ #putatives.
        """
        # Build intrinsic priors.
        prior1 = self.__theia_intrinsics(camera_intrinsics_i1)
        prior2 = self.__theia_intrinsics(camera_intrinsics_i2)

        # Convert keypoint to Theia format.
        kpts_theia_i1 = self.__theia_keypoints(keypoints_i1)
        kpts_theia_i2 = self.__theia_keypoints(keypoints_i2)

        # Build IndexedFeatureMatch list for Theia.
        # Note: distance set arbitrarily to 0 as it is not used during geometric verification.
        indexed_matches = [pt.matching.IndexedFeatureMatch(idx1, idx2, 0.0) for idx1, idx2 in match_indices]

        # Set geometric verification options.
        options = pt.sfm.TwoViewMatchGeometricVerificationOptions()
        options.guided_matching = True  # epipolar-guided matching
        options.min_num_inlier_matches = 0
        options.bundle_adjustment = False
        options.estimate_twoview_info_options.max_sampson_error_pixels = self._estimation_threshold_px
        options.estimate_twoview_info_options.expected_ransac_confidence = RANSAC_SUCCESS_PROB
        options.estimate_twoview_info_options.max_ransac_iterations = RANSAC_MAX_ITERS
        options.estimate_twoview_info_options.use_lo = True
        options.estimate_twoview_info_options.ransac_type = pt.sfm.RansacType(0)

        # Verify!
        success, two_view_info, verified_matches = pt.sfm.VerifyMatches(
            options, prior1, prior2, kpts_theia_i1, kpts_theia_i2, indexed_matches
        )
        if not success:
            return self._failure_result

        # Unpack results.
        i1Ri2_angleaxis = two_view_info.rotation_2
        i1Ri2_rot_angle = np.linalg.norm(i1Ri2_angleaxis)
        i1Ri2 = Rot3.AxisAngle(i1Ri2_angleaxis, i1Ri2_rot_angle)
        i1ti2 = two_view_info.position_2

        i1Ti2 = Pose3(i1Ri2.inverse(), i1ti2)
        i2Ti1 = i1Ti2.inverse()
        i2Ui1 = Unit3(i2Ti1.translation())

        verified_match_indices = np.array([[m.feature1_ind, m.feature2_ind] for m in verified_matches])
        inlier_ratio_est_model = verified_match_indices.shape[0] / match_indices.shape[0]  # can be >1

        return i1Ri2, i2Ui1, verified_match_indices, inlier_ratio_est_model
