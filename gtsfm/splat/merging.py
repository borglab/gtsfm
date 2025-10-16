"""
Merging Gaussian Splats from different partitions into one coordinate frame

Authors: Harneet Singh Khanuja
"""

import os
from pathlib import Path

import gtsam
import numpy as np
import torch
from plyfile import PlyData

from gtsfm.splat.rendering import save_splats
from gtsfm.utils import splat


def _get_splats(vertex):
    """Reads the splats from a ply file"""
    opacities = np.array(vertex["opacity"])[:, np.newaxis]
    means = np.stack([np.array(vertex["x"]), np.array(vertex["y"]), np.array(vertex["z"])], axis=1)
    scales = np.stack([np.array(vertex["scale_0"]), np.array(vertex["scale_1"]), np.array(vertex["scale_2"])], axis=1)
    quats = np.stack(
        [np.array(vertex["rot_0"]), np.array(vertex["rot_1"]), np.array(vertex["rot_2"]), np.array(vertex["rot_3"])],
        axis=1,
    )
    sh0 = np.stack([np.array(vertex["f_dc_0"]), np.array(vertex["f_dc_1"]), np.array(vertex["f_dc_2"])], axis=1)[
        :, np.newaxis, :
    ]
    K = 15
    shN_list = []
    for i in range(K * 3):
        shN_list.append(np.array(vertex[f"f_rest_{i}"]))
    shN = np.stack(shN_list, axis=1)
    shN = shN.reshape(means.shape[0], K, 3)
    splats = {
        "means": torch.from_numpy(means).float(),
        "opacities": torch.from_numpy(opacities).float(),
        "scales": torch.from_numpy(scales).float(),
        "quats": torch.from_numpy(quats).float(),
        "sh0": torch.from_numpy(sh0).float(),
        "shN": torch.from_numpy(shN).float(),
    }

    return splats


# inputs will need to be changed
transformations = {1: gtsam.Similarity3(), 2: gtsam.Similarity3(), 3: gtsam.Similarity3(), 4: gtsam.Similarity3()}
results_path = Path("/home/hkhanuja3/testing_gtsfm/gtsfm/results/")

transformed_gaussians = []
for idx, partition_folder in enumerate(sorted(os.listdir(results_path))):
    ply_file = Path(partition_folder) / "gs_output" / "gaussian_splats.ply"
    gaussian_partition = PlyData.read(ply_file)
    partition_splats = _get_splats(gaussian_partition["vertex"])
    transformation = transformations[idx]
    transformed_gaussians.append(splat.transform_gaussian_splats(partition_splats, transformation))

all_gaussian_splats = {}
for key in transformed_gaussians[0].keys():
    all_gaussian_splats[key] = torch.cat([t_gs[key].unsqueeze(0) for t_gs in transformed_gaussians], dim=0)

save_splats(Path(results_path), all_gaussian_splats)
