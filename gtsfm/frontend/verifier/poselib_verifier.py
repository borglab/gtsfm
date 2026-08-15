"""PoseLib-based verifier using 5-point E-matrix with LO-RANSAC and local BA.

PoseLib's estimate_relative_pose provides:
- 5-point essential matrix solver (more constrained than 8-point F-matrix)
- Locally Optimized RANSAC (refines hypotheses on inlier sets)
- Local bundle adjustment within RANSAC loop
- Integrated pose estimation (no separate F→E→pose decomposition chain)

Inlier re-scoring is done separately in the multi-view optimizer (GLOMAP style).

Reference: PoseLib (https://github.com/PoseLib/PoseLib)
"""

from typing import Optional, Tuple

import numpy as np
import poselib
from gtsam import Cal3_S2, Cal3Bundler, Cal3DS2, Cal3Fisheye, Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()


# pycolmap two-view ConfigurationType values we emit (see gric_verifier.ConfigurationType).
_UNCALIBRATED_CONFIG = 3  # F-model -> plumbed to the closed-form Fetzer
_PLANAR_CONFIG = 6  # PLANAR_OR_PANORAMIC -> excluded from Fetzer (degenerate F)
_MAX_H_INLIER_RATIO = 0.8  # COLMAP EstimateTwoViewGeometry model selection threshold


class PoseLibVerifier(VerifierBase):
    """Verifier using PoseLib's estimate_relative_pose with 5-point solver + LO-RANSAC + local BA.

    Returns PoseLib's raw inliers.
    """

    def __init__(
        self,
        estimation_threshold_px: float = 2.0,
        max_iterations: int = 100000,
        confidence: float = 0.999999,
        estimate_calibration_geometry: bool = False,
    ) -> None:
        super().__init__(use_intrinsics_in_verification=True, estimation_threshold_px=estimation_threshold_px)
        self._max_iterations = max_iterations
        self._confidence = confidence
        # When True, also estimate F (for Fetzer) + config (planar gate) directly from PoseLib.
        self._estimate_calibration_geometry = estimate_calibration_geometry

    def __repr__(self) -> str:
        # Only extend the repr when the flag is on, so other configs' two-view caches stay valid.
        return super().__repr__() + ("_calibgeom" if self._estimate_calibration_geometry else "")

    def _cal3bundler_to_poselib_camera(self, K: CALIBRATION_TYPE, width: int, height: int) -> poselib.Camera:
        """Convert GTSAM calibration to PoseLib Camera."""
        if isinstance(K, Cal3Bundler):
            fx = K.fx()
            k1 = K.k1()
            k2 = K.k2()
            cx = K.px()
            cy = K.py()
            if abs(k1) < 1e-10 and abs(k2) < 1e-10:
                return poselib.Camera("PINHOLE", [fx, fx, cx, cy], width, height)
            if abs(k2) < 1e-10:
                return poselib.Camera("SIMPLE_RADIAL", [fx, cx, cy, k1], width, height)
            return poselib.Camera("RADIAL", [fx, cx, cy, k1, k2], width, height)

        if isinstance(K, Cal3_S2):
            return poselib.Camera("PINHOLE", [K.fx(), K.fy(), K.px(), K.py()], width, height)

        if isinstance(K, Cal3DS2):
            return poselib.Camera(
                "OPENCV",
                [K.fx(), K.fy(), K.px(), K.py(), K.k1(), K.k2(), K.p1(), K.p2()],
                width,
                height,
            )

        if isinstance(K, Cal3Fisheye):
            return poselib.Camera(
                "OPENCV_FISHEYE",
                [K.fx(), K.fy(), K.px(), K.py(), K.k1(), K.k2(), K.k3(), K.k4()],
                width,
                height,
            )

        raise ValueError(f"Unsupported GTSAM calibration type for PoseLib: {type(K)}")

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: CALIBRATION_TYPE,
        camera_intrinsics_i2: CALIBRATION_TYPE,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float, Optional[np.ndarray], Optional[int]]:
        """Estimate relative pose using PoseLib's 5-point solver with LO-RANSAC."""
        if match_indices.shape[0] < self._min_matches:
            return self._failure_result

        pts1 = keypoints_i1.coordinates[match_indices[:, 0]]
        pts2 = keypoints_i2.coordinates[match_indices[:, 1]]

        K1 = camera_intrinsics_i1.K()
        K2 = camera_intrinsics_i2.K()
        w1 = int(2 * K1[0, 2])
        h1 = int(2 * K1[1, 2])
        w2 = int(2 * K2[0, 2])
        h2 = int(2 * K2[1, 2])

        cam1 = self._cal3bundler_to_poselib_camera(camera_intrinsics_i1, w1, h1)
        cam2 = self._cal3bundler_to_poselib_camera(camera_intrinsics_i2, w2, h2)

        ransac_opt = {
            "max_reproj_error": self._estimation_threshold_px,
            "max_iterations": self._max_iterations,
            "min_iterations": 100,
            "success_prob": self._confidence,
        }
        bundle_opt = {"max_iterations": 25}

        try:
            pose, info = poselib.estimate_relative_pose(
                pts1,
                pts2,
                cam1,
                cam2,
                ransac_opt=ransac_opt,
                bundle_opt=bundle_opt,
            )
        except Exception as e:
            logger.debug("PoseLib estimate_relative_pose failed: %s", e)
            return self._failure_result

        inlier_mask = np.array(info.get("inliers", []), dtype=bool)
        if len(inlier_mask) == 0 or inlier_mask.sum() < self._min_matches:
            return self._failure_result

        R = np.array(pose.R)
        t = np.array(pose.t)

        if np.linalg.norm(t) < 1e-12:
            return self._failure_result

        i2Ri1 = Rot3(R)
        i2Ui1 = Unit3(t)

        inlier_idxs = np.where(inlier_mask)[0]
        v_corr_idxs = match_indices[inlier_idxs]
        inlier_ratio = float(inlier_mask.sum()) / len(inlier_mask)

        if not self._estimate_calibration_geometry:
            return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio, None, None

        # Focal-INDEPENDENT F (for the closed-form Fetzer) + planar/uncalibrated config, both from
        # PoseLib. F is estimated directly from the 2D correspondences (NOT K2^-T E K1^-1 from the
        # calibrated essential, which would be focal-dependent -> circular Fetzer residual).
        i2Fi1, config = self._estimate_fundamental_and_config(pts1, pts2)
        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio, i2Fi1, config

    def _estimate_fundamental_and_config(
        self, pts1: np.ndarray, pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """PoseLib estimate_fundamental (focal-independent F) + H/F-inlier-ratio config classification.

        Mirrors COLMAP's EstimateTwoViewGeometry model selection: PLANAR if the homography explains
        nearly as many correspondences as the fundamental matrix (n_H > 0.8 * n_F), else UNCALIBRATED.
        Returns (None, None) on failure so the pair is kept as an edge but excluded from Fetzer.
        """
        thr = self._estimation_threshold_px
        try:
            F, f_info = poselib.estimate_fundamental(pts1, pts2, {"max_epipolar_error": thr}, {})
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("PoseLib estimate_fundamental failed: %s", e)
            return None, None
        F = np.asarray(F, dtype=np.float64)
        if F.shape != (3, 3) or not np.all(np.isfinite(F)) or np.allclose(F, 0.0):
            return None, None
        n_F = int(f_info.get("num_inliers", 0))
        try:
            _, h_info = poselib.estimate_homography(pts1, pts2, {"max_reproj_error": thr}, {})
            n_H = int(h_info.get("num_inliers", 0))
        except Exception:  # pragma: no cover - defensive
            n_H = 0
        config = _PLANAR_CONFIG if n_H > _MAX_H_INLIER_RATIO * max(n_F, 1) else _UNCALIBRATED_CONFIG
        return F, config
