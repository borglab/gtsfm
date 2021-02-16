"""
Library functionality for aligning two trajectories by fitting a SIM(3) transformation

Zichao Zhang, Davide Scaramuzza: A Tutorial on Quantitative Trajectory Evaluation for
Visual(-Inertial) Odometry, IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

Code adapted from https://github.com/uzh-rpg/rpg_trajectory_evaluation
"""
from typing import List, Optional, Tuple

import numpy as np
from gtsam import Rot3, Pose3

# from gtsfm.utils.logger import get_logger

# logger = get_logger()

# def align_poses(input_list: List[Pose3], ref_list: List[Pose3]) -> List[Pose3]:
#     """Align by similarity transformation.

#     We calculate s, R, t so that:
#         gt = R * s * est + t

#     We force SIM(3) alignment rather than SE(3) alignment.
#     We assume the two trajectories are of the exact same length.
#     Ref: rpg_trajectory_evaluation/src/rpg_trajectory_evaluation/align_utils.py

#     Args:
#         input_list: input poses which need to be aligned, suppose w1Ti in world-1 frame for all frames i.
#         ref_list: reference poses which are target for alignment, suppose w2Ti_ in world-2 frame for all frames i.
#             We set the reference as the ground truth.

#     Returns:
#         transformed poses which have the same origin and scale as reference (now living in world-2 frame)
#     """
#     p_est = np.array([w1Ti.translation() for w1Ti in input_list])
#     p_gt = np.array([w2Ti.translation() for w2Ti in ref_list])

#     assert p_est.shape[1] == 3
#     assert p_gt.shape[1] == 3

#     n_to_align = p_est.shape[0]
#     assert p_gt.shape[0] == n_to_align
#     assert n_to_align >= 2, "SIM(3) alignment uses at least 2 frames"

#     scale, w2Rw1, w2tw1 = align_umeyama(p_gt, p_est)  # note the order

#     aligned_input_list = []
#     for i in range(n_to_align):
#         w1Ti = input_list[i]
#         p_est_aligned = scale * w2Rw1 @ w1Ti.translation() + w2tw1
#         R_est_aligned = w2Rw1 @ w1Ti.rotation().matrix()
#         aligned_input_list += [Pose3( Rot3(R_est_aligned), p_est_aligned)]

#     logger.info("Trajectory alignment complete.")

#     return aligned_input_list


def align_umeyama(
    model: np.ndarray, data: np.ndarray, known_scale: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.
    model = s * R * data + t

    Ref: rpg_trajectory_evaluation/src/rpg_trajectory_evaluation/align_trajectory.py

    Args:
        model: Array of shape (N,3) representing first trajectory as ground truth
        data: Array of shape (N,3), representing second trajectory

    Returns:
        s: float scalar representing scale factor
        R: rotation matrix of shape (3,3)
        t: translation vector of shape (3,1)
    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = model.shape[0]

    # correlation
    C = 1.0 / n * np.dot(model_zerocentered.T, data_zerocentered)
    # squared l2-norm
    sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, Vt_svd = np.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = Vt_svd.T

    S = np.eye(3)
    if np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    R = U_svd @ S @ V_svd.T

    if known_scale:
        s = 1
    else:
        s = 1.0 / sigma2 * np.trace(D_svd @ S)

    t = mu_M - s * np.dot(R, mu_D)

    return s, R, t


# import numpy as np
# from gtsam import Cal3_S2, Point3, Pose3, Rot3, Unit3
# from gtsam.examples import SFMdata
# from scipy.spatial.transform import Rotation



# def test_compare_global_poses_scaled_squares():
#     """Make sure a big and small square can be aligned.

#     The u's represent a big square (10x10), and v's represents a small square (4x4).
#     """
#     R0 = Rotation.from_euler("z", 0, degrees=True).as_matrix()
#     R90 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
#     R180 = Rotation.from_euler("z", 180, degrees=True).as_matrix()
#     R270 = Rotation.from_euler("z", 270, degrees=True).as_matrix()

#     wTu0 = Pose3(Rot3(R0), np.array([2, 3, 0]))
#     wTu1 = Pose3(Rot3(R90), np.array([12, 3, 0]))
#     wTu2 = Pose3(Rot3(R180), np.array([12, 13, 0]))
#     wTu3 = Pose3(Rot3(R270), np.array([2, 13, 0]))

#     wTi_list = [wTu0, wTu1, wTu2, wTu3]

#     wTv0 = Pose3(Rot3(R0), np.array([4, 3, 0]))
#     wTv1 = Pose3(Rot3(R90), np.array([8, 3, 0]))
#     wTv2 = Pose3(Rot3(R180), np.array([8, 7, 0]))
#     wTv3 = Pose3(Rot3(R270), np.array([4, 7, 0]))

#     wTi_list_ = [wTv0, wTv1, wTv2, wTv3]

#     import pdb
#     pdb.set_trace()

#     pose_graphs_equal = compare_global_poses(
#         wTi_list, wTi_list_
#     )
#     assert pose_graphs_equal





# def compare_global_poses(
#     wTi_list: List[Optional[Pose3]],
#     wTi_list_: List[Optional[Pose3]],
#     rot_err_thresh: float = 1e-2,
#     trans_err_thresh: float = 1e-1,
# ) -> bool:
#     """Helper function to compare two lists of global Pose3, considering the
#     origin and scale ambiguous.

#     Notes:
#     1. The input lists have the poses in the same order, and can contain None entries.
#     2. To resolve global origin ambiguity, we will fix one image index as origin in both the inputs and transform both
#        the lists to the new origins.
#     3. As there is a scale ambiguity, we use the median scaling factor to resolve the ambiguity.

#     Args:
#         wTi_list: 1st list of poses.
#         wTi_list_: 2nd list of poses.
#         rot_err_thresh (optional): error threshold for rotations. Defaults to 1e-3.
#         trans_err_thresh (optional): relative error threshold for translation. Defaults to 1e-1.

#     Returns:
#         results of the comparison.
#     """

#     # check the length of the input lists
#     if len(wTi_list) != len(wTi_list_):
#         return False

#     # check the presense of valid Pose3 objects in the same location
#     wTi_valid = [i for (i, wTi) in enumerate(wTi_list) if wTi is not None]
#     wTi_valid_ = [i for (i, wTi) in enumerate(wTi_list_) if wTi is not None]
#     if wTi_valid != wTi_valid_:
#         return False

#     if len(wTi_valid) <= 1:
#         # we need >= two entries going forward for meaningful comparisons
#         return False

#     # align the remaining poses
#     wTi_list = [wTi_list[i] for i in wTi_valid]
#     wTi_list_ = [wTi_list_[i] for i in wTi_valid_]

#     wTi_list = align_poses(wTi_list, ref_list=wTi_list_)

#     import pdb
#     pdb.set_trace()

#     for (wTi, wTi_) in zip(wTi_list, wTi_list_):
#         equal = wTi.rotation().equals(wTi_.rotation(), rot_err_thresh)

#     rot_errors = [
#         wTi.rotation().equals(wTi_.rotation(), rot_err_thresh)
#         for (wTi, wTi_) in zip(wTi_list, wTi_list_)
#     ]

#     trans_errors = [
#         np.allclose(wTi.translation(), wTi_.translation(), rtol=trans_err_thresh)
#         for (wTi, wTi_) in zip(wTi_list, wTi_list_)
#     ]

#     import pdb
#     pdb.set_trace()


#     return all(errors)


# if __name__ == '__main__':

#     test_compare_global_poses_scaled_squares()
