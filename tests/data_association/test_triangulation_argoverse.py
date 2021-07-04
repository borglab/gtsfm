
import copy
from pathlib import Path

import imageio
import numpy as np

from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.calibration import (
    CameraConfig,
    get_calibration_config,
    point_cloud_to_homogeneous,
    project_lidar_to_img
)
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3

from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3

from gtsfm.common.sfm_track import SfmTrack2d, SfmMeasurement
from gtsfm.data_association.point3d_initializer import TriangulationParam, Point3dInitializer


def main(dataset_dir: str, log_id: str) -> None:
    """ """
    camera_name = "ring_front_center"

    dl = SimpleArgoverseTrackingDataLoader(data_dir=dataset_dir, labels_dir=dataset_dir)

    all_cam_fpaths = dl.get_ordered_log_cam_fpaths(log_id, camera_name)
    all_cam_timestamps = [ int(Path(fpath).stem.split('_')[-1]) for fpath in all_cam_fpaths]

    num_images = 5

    # Take a N-frame continuous image sequence
    cam_timestamps = all_cam_timestamps[:num_images]
    city_SE3_egovehicle_list = [dl.get_city_SE3_egovehicle(log_id, t) for t in cam_timestamps]

    initial_timestamp = cam_timestamps[0]

    log_calib_data = dl.get_log_calibration_data(log_id)
    cam_config = dl.get_log_camera_config(log_id, camera_name)
    K = cam_config.intrinsic

    calib_cal3bundler = Cal3Bundler(fx=K[0,0], k1=0, k2=0, u0=K[0,2], v0=K[1,2])

    ply_fpath = dl.get_closest_lidar_fpath(log_id, initial_timestamp)
    lidar_egofr = load_ply(ply_fpath)
    initial_lidar_timestamp = Path(ply_fpath).stem.split("_")[-1]
    initial_lidar_timestamp = int(initial_lidar_timestamp)
    lidar_cityfr = city_SE3_egovehicle_list[0].transform_point_cloud(lidar_egofr)

    extrinsic = cam_config.extrinsic
    camera_SE3_egovehicle = SE3(rotation=extrinsic[:3,:3], translation=extrinsic[:3,3])
    egovehicle_SE3_camera = camera_SE3_egovehicle.inverse()

    track_camera_dict = {}
    for i in range(num_images):
        wTe = city_SE3_egovehicle_list[i]
        wTi = wTe.compose(egovehicle_SE3_camera)
        track_camera_dict[i] = PinholeCameraCal3Bundler(
            Pose3(wTi.transform_matrix),
            calib_cal3bundler
        ) 

    triangulator = Point3dInitializer(track_camera_dict=track_camera_dict, mode=TriangulationParam.NO_RANSAC, reproj_error_thresh=100)

    import pdb; pdb.set_trace()

    # Find a LiDAR point in the 0th frame
    for j, pt_city in enumerate(lidar_cityfr):
        if j % 100 == 0:
            print(f"On LiDAR point {j}")

        measurements = []
        for i, cam_timestamp in enumerate(cam_timestamps):

            egovehicle_SE3_city = city_SE3_egovehicle_list[i].inverse()
            pt_ego = egovehicle_SE3_city.transform_point_cloud(pt_city.reshape(1,3))

            # Project it into front center camera, for all 50 frames, check if it's visible / cheirality
            points_h = point_cloud_to_homogeneous(pt_ego).T

            uv, uv_cam, valid_pts_bool = project_lidar_to_img(
                points_h,
                copy.deepcopy(log_calib_data),
                camera_name,
                return_camera_config=False,
                remove_nan=False,
            )

            # (uv, uv_cam, valid_pts_bool, K,) = project_lidar_to_img_motion_compensated(
            #     points_h,  # these are recorded at lidar_time
            #     copy.deepcopy(log_calib_data),
            #     camera_name,
            #     cam_timestamp,
            #     lidar_timestamp,
            #     dataset_dir,
            #     log_id,
            #     return_K=True,
            # )
            print(uv, valid_pts_bool)
            uv = uv.squeeze()
            valid_pts_bool = valid_pts_bool.squeeze()
            if valid_pts_bool.sum() == 0:
                continue

            measurements.append(SfmMeasurement(i=i, uv=uv))

        if len(measurements) < 2:
            continue

        import pdb; pdb.set_trace()

        track_2d = SfmTrack2d(measurements=measurements)

        # Triangulate it using GT poses
        track_3d, avg_track_reproj_error, is_cheirality_failure = triangulator.triangulate(track_2d)

        # Compare point-point error
        err = np.linalg.norm(pt_city - track_3d.point3())
        print(f"Point error was {err:.2f}")

        # Triangulate it using GT poses, and then run bundle-adjustment on track point only
        # Compare point-point error


if __name__ == "__main__":
    """
    Data available at https://s3.amazonaws.com/argoai-argoverse/tracking_sample_v1.1.tar.gz
    """
    dataset_dir = "/Users/johnlambert/Downloads/argoverse-tracking_sample_3/sample"
    log_id = "c6911883-1843-3727-8eaa-41dc8cda8993"
    main(dataset_dir, log_id)


