#!/usr/bin/env python3

# Created by Beda Berner
# beda.berner@hilti.com

# Modified for using both fastlio and gtsfm TUM files
# If you already have matplotlib, scipy and numpy, this will need an additional 
# pip install evo==1.13.5

# Instructions to run
# ./eval_hilti.py <path_to_tum1> <path_to_reference_tum> <optional_path_to_tum2>

from evo.core import metrics
from evo.core import lie_algebra as lie
from evo.tools.plot import *
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt
import sys
from evo.tools.settings import SETTINGS

SETTINGS.plot_usetex = False
from evo.core import sync
from evo.tools import file_interface
import copy
from scipy.spatial.transform import Rotation as R
from evo.core.trajectory import PoseTrajectory3D

import gtsfm.utils.io as io_utils
from gtsam import Pose3
import yaml
import argparse

def traj_custom(ax: plt.Axes, plot_mode: PlotMode, traj: trajectory.PosePath3D,
                style: str = '-', color: str = 'black', label: str = "",
                alpha: float = 1.0, markers: str = 'None') -> None:
    """
    plot a path/trajectory based on xyz coordinates into an axis
    :param ax: the matplotlib axis
    :param plot_mode: PlotMode
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param style: matplotlib line style
    :param color: matplotlib color
    :param label: label (for legend)
    :param alpha: alpha value for transparency
    """
    x_idx, y_idx, z_idx = plot_mode_to_idx(plot_mode)
    x = traj.positions_xyz[:, x_idx]
    y = traj.positions_xyz[:, y_idx]
    if plot_mode == PlotMode.xyz:
        z = traj.positions_xyz[:, z_idx]
        ax.plot(x, y, z, linestyle=style, color=color, label=label, alpha=alpha, marker=markers)
        if SETTINGS.plot_xyz_realistic:
            set_aspect_equal_3d(ax)
    else:
        ax.plot(x, y, linestyle=style, color=color, label=label, alpha=alpha, marker=markers)
    if label:
        ax.legend(frameon=True)


def trajectories_custom(fig: plt.Figure, trajectories: typing.Union[
    trajectory.PosePath3D, typing.Sequence[trajectory.PosePath3D],
    typing.Dict[str, trajectory.PosePath3D]], plot_mode=PlotMode.xy,
                        title: str = "", subplot_arg: int = 111) -> None:
    """
    high-level function for plotting multiple trajectories
    :param fig: matplotlib figure
    :param trajectories: instance or container of PosePath3D or derived
    - if it's a dictionary, the keys (names) will be used as labels
    :param plot_mode: e.g. plot.PlotMode.xy
    :param title: optional plot title
    :param subplot_arg: optional matplotlib subplot ID if used as subplot
    """
    ax = prepare_axis(fig, plot_mode, subplot_arg)
    cmap_colors = None
    if SETTINGS.plot_multi_cmap.lower() != "none" and isinstance(
            trajectories, collections.Iterable):
        cmap = getattr(cm, SETTINGS.plot_multi_cmap)
        cmap_colors = iter(cmap(np.linspace(0, 1, len(trajectories))))

    # helper function
    def draw(t, name=""):
        if cmap_colors is None:
            color = next(ax._get_lines.prop_cycler)['color']
        else:
            color = next(cmap_colors)
        if SETTINGS.plot_usetex:
            name = name.replace("_", "\\_")
        if name != "evaluated estimate points":
            traj_custom(ax, plot_mode, t, '-', color, name)
        else:
            traj_custom(ax, plot_mode, t, 'dotted', color, name, markers='o')

    if isinstance(trajectories, trajectory.PosePath3D):
        draw(trajectories)
    elif isinstance(trajectories, dict):
        for name, t in trajectories.items():
            draw(t, name)
    else:
        for t in trajectories:
            draw(t)


def evaluate_trajectory(est_file, ref_file, prefix):
    traj_ref = file_interface.read_tum_trajectory_file(ref_file)

    apply_pole_tip_calibration = True
    # apply poletip calibration
    if apply_pole_tip_calibration == True:
        calibration_type = ref_file.split('_')[-1].lower()
        if calibration_type == 'pole.txt':
            T_imu_ref = np.array([[  0.176566,   -0.984288, 0.00121622, -0.00938425],
                                  [ -0.984256,    -0.17655, 0.00837907,  -0.0148401],
                                  [-0.0080327, -0.00267653,  -0.999964,     1.66722],
                                  [         0,           0,          0,         1.0]])
        elif calibration_type == 'prism.txt':
            T_imu_ref = np.array([[  0.176566,   -0.984288, 0.00121622, -0.00594496],
                                  [ -0.984256,    -0.17655, 0.00837907, -0.00721288],
                                  [-0.0080327, -0.00267653,  -0.999964,    0.272943],
                                  [         0,           0,          0,         1.0]])
        elif calibration_type == 'imu.txt':
            T_imu_ref = np.identity(4)
        else:
            print("reference file has non supported calibration type. Please don't change the reference file names.")
            exit()
        data = np.genfromtxt(est_file, delimiter=' ', skip_header=False)

        for i in range(data.shape[0]):
            rot_mat = R.from_quat([data[i, 4:8]]).as_matrix().reshape([3, 3]) # from_quat(), xyzw
            transl = data[i, 1:4].reshape([3, 1])
            homogeneous_transform = np.vstack([np.hstack([rot_mat, transl]), np.array([0, 0, 0, 1])])
            result = homogeneous_transform @ T_imu_ref
            data[i, 1:4] = result[0:3,3].reshape([1, 3])
            data[i, 4:] = R.from_matrix(result[0:3,0:3]).as_quat()

        stamps = data[:, 0]  # n x 1
        xyz = data[:, 1:4]  # n x 3
        quat_wxyz = data[:, [7,4,5,6]] # n x 4
        traj_est = PoseTrajectory3D(xyz, quat_wxyz, stamps) # PoseTrajectory3D(), wxyz

    else:
        traj_est = file_interface.read_tum_trajectory_file(est_file)

    # determine if a dense or sparse reference file is used
    if traj_ref.num_poses > 100:
        dense_trajectory = True
    else:
        dense_trajectory = False

    # timesync the reference and estimate trajectories
    max_diff = 1
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est, max_diff)

    # align the trajectories
    traj_est_aligned = copy.deepcopy(traj_est_sync)
    umeyama_parameters = traj_est_aligned.align(traj_ref_sync, correct_scale=False, correct_only_scale=False)
    traj_est_aligned_complete = copy.deepcopy(traj_est)
    traj_est_aligned_complete.scale(umeyama_parameters[2])
    traj_est_aligned_complete.transform(lie.se3(umeyama_parameters[0], umeyama_parameters[1]))

    # calculate the metrics
    data = (traj_ref_sync, traj_est_aligned)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)

    ape_stats = ape_metric.get_all_statistics()
    for i in ape_stats:
        print("APE {} = {}.".format(i, ape_stats[i]))

    # plot the trajectories
    fig = plt.figure()
    if dense_trajectory:
        traj_by_label = {
            "estimate": traj_est_aligned,
            "reference": traj_ref_sync
        }
        trajectories(fig, traj_by_label, plot.PlotMode.xyz)

    else:
        traj_by_label = {
            "estimate": traj_est_aligned_complete,
            "evaluated estimate points": traj_est_aligned,
            "reference": traj_ref_sync
        }
        trajectories_custom(fig, traj_by_label, plot.PlotMode.xyz)


    plt.savefig(prefix+"_trajectory.png")

    seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est_sync.timestamps]
    fig = plt.figure()

    # plot the error over time
    if dense_trajectory:
        plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
                          statistics={s: v for s, v in ape_stats.items() if s != "sse"},
                          name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
    else:
        plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
                          statistics={s: v for s, v in ape_stats.items() if s != "sse"},
                          name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)", marker='o',
                          linestyle='dotted')
    plt.savefig(prefix+"_APE_error.png")

def read_timestamps_from_tum_file(file_path):
    """Reads the timestamp information from a tum file."""
    data = np.genfromtxt(file_path, delimiter=' ', skip_header=False)
    return data[:, 0]


def write_poses_to_tum_file(poses, file_path, timestamps):
    """Writes GTSfM pose estimates to a TUM file. 
    Assumes that the index of poses corresponds to index of timestamps.
    """
    REFERENCE_CAMERA_IDX = 2
    NUM_CAMERAS_IN_RIG = 5
    with open(file_path, 'w') as f:
        for i, pose in enumerate(poses):
            if pose is None:
                continue
            if i % NUM_CAMERAS_IN_RIG != REFERENCE_CAMERA_IDX:
                continue
            qw, qx, qy, qz = pose.rotation().quaternion()
            tx, ty, tz = pose.translation()
            ts = timestamps[i]
            f.write(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")


def get_body_poses(world_T_camera_poses, camera_T_body):
    """Converts pose estimates of cam0 to that of the body frame (IMU)."""
    world_T_body_poses = []
    for world_T_camera in world_T_camera_poses:
        if world_T_camera is None:
            world_T_body_poses.append(None)
            continue
        world_T_body_poses.append(world_T_camera * camera_T_body)
    return world_T_body_poses


if __name__ == "__main__":

    # check if files where provided in the command line argument
    parser = argparse.ArgumentParser(description="Evaluation arguments")

    parser.add_argument(
        "--fastlio_tum",
        type=str,
        default=None,
        required=True,
        help="Path to fastlio_odom.txt poses",
    )
    parser.add_argument(
        "--gtsfm_images_txt",
        type=str,
        default=None,
        required=True,
        help="Path to gtsfm images.txt pose-BA output",
    )
    parser.add_argument(
        "--cam2_calib_yaml",
        type=str,
        default=None,
        required=True,
        help="Path to Hilti's camera-2 calibration YAML.",
    )
    parser.add_argument(
        "--gt_tum",
        type=str,
        default=None,
        required=True,
        help="Path to GT tum file from Hilti"
    )
    args = parser.parse_args()
    
    GTSFM_TUM_PATH = "gtsfm_imu_poses.txt"
    gtsfm_poses, _ = io_utils.read_images_txt(args.gtsfm_images_txt)
    with open(args.cam2_calib_yaml, "r") as file:
        calibration_data = yaml.safe_load(file)

    cam_T_imu = Pose3(calibration_data["cam0"]["T_cam_imu"])
    body_poses = get_body_poses(gtsfm_poses, cam_T_imu)
    timestamps = read_timestamps_from_tum_file(args.fastlio_tum)

    write_poses_to_tum_file(body_poses, GTSFM_TUM_PATH, timestamps)

    print("FASTLIO poses evaluation results:")
    evaluate_trajectory(args.fastlio_tum, args.gt_tum, "fastlio")
    print("\n\nGTSfM poses evaluation results:")
    evaluate_trajectory(GTSFM_TUM_PATH, args.gt_tum, "gtsfm")

