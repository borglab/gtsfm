""" """

from gtsam import Rot3
import numpy as np

from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
import gtsfm.utils.io as io_utils
from gtsfm.loader.colmap_loader import ColmapLoader


def main() -> None:

    images_dir = "/Users/johnlambert/Downloads/gerrard-hall-100/images"
    colmap_files_dirpath = "/Users/johnlambert/Downloads/gerrard-hall-100/colmap-3.7-sparse-txt-2023-07-27"

    loader = ColmapLoader(
        colmap_files_dirpath=colmap_files_dirpath,
        images_dir=images_dir,
        max_resolution=760,
    )
    gt_wTi_list = loader.get_gt_poses()

    #num_images = 128
    #fpath = "/Users/johnlambert/Documents/2023_10_19_0013_spanning_tree_view_graph_estimator/20231019_041341/20231019_041341__south-building-128__results__num_matched5__maxframelookahead10__760p__unified_superglue/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"

    num_images = 98 # 100
    fpath = "/Users/johnlambert/Documents/2023_10_19_0013_spanning_tree_view_graph_estimator/20231019_041341/20231019_041341__gerrard-hall-100__results__num_matched5__maxframelookahead10__760p__unified_sift/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    # fpath = "/Users/johnlambert/Downloads/gtsfm_2023_07_08/gtsfm/door_results_2023_10_18/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    data = io_utils.read_json_file(fpath)

    i2Ri1_dict = {}
    two_view_reports_dict = {}
    frontend_uncertainty_dict = {}

    for entry in data:
        i1 = entry["i1"]
        i2 = entry["i2"]
        rot_dict = entry["rotation"]
        qw = rot_dict["qw"]
        qx = rot_dict["qx"]
        qy = rot_dict["qy"]
        qz = rot_dict["qz"]

        i2Ri1 = Rot3(float(qw), float(qx), float(qy), float(qz))
        i2Ri1_dict[(i1, i2)] = i2Ri1

        two_view_reports_dict[(i1, i2)] = TwoViewEstimationReport(
            v_corr_idxs=None,
            num_inliers_est_model=entry["num_inliers_est_model"],
            R_error_deg=entry["rotation_angular_error"],
            U_error_deg=entry["translation_angular_error"],
        )

        #uncertainty = 1 / two_view_reports_dict[(i1, i2)].num_inliers_est_model
        uncertainty = np.exp(1 / (two_view_reports_dict[(i1, i2)].num_inliers_est_model / 50))
        # 
        # uncertainty = two_view_reports_dict[(i1, i2)].R_error_deg ** 3
        # # 
        print(f"Inliers {two_view_reports_dict[(i1, i2)].num_inliers_est_model} Uncertainty: ", uncertainty)
        # if uncertainty < 0.05:

        #     uncertainty = max(uncertainty, 0.05)
        #     print(f"\tClamped to {uncertainty}")

        assert uncertainty > 0.01
        # assert uncertainty < 200

        frontend_uncertainty_dict[(i1,i2)] = uncertainty
        


    rot_avg_module = ShonanRotationAveraging()
    wRis, metrics = rot_avg_module._run_rotation_averaging_base(
        num_images,
        i2Ri1_dict=i2Ri1_dict,
        i1Ti2_priors={},
        frontend_uncertainty_dict=frontend_uncertainty_dict,
        wTi_gt=gt_wTi_list,
    )
    metrics_dict = metrics.get_metrics_as_dict()
    #print("Metrics:", )

    print(metrics_dict['rotation_averaging_metrics']['num_rotations_computed'])
    print(metrics_dict['rotation_averaging_metrics']['rotation_angle_error_deg']['summary'])



if __name__ == "__main__":
    main()