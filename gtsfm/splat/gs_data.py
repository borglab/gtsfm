"""Class that implements an interface from GtsfmData to Gaussian Splatting data.

Authors: Harneet Singh Khanuja
"""
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.utils import images as image_utils
from gtsfm.utils.splat import auto_orient_and_center_poses, _undistort_image

logger = logger_utils.get_logger()


class GaussianSplattingData(Dataset):
    """Converts the data format from GtsfmData to Gaussian Splatting input"""

    def __init__(self, images: Dict[int, Image], sfm_result: GtsfmData) -> None:
        """Cache images and GtsfmData for Gaussian Splatting training, including camera poses and tracks.

        Args:
            images: input images (H, W, C) to GTSFM
            sfm_result: sparse multiview reconstruction result
        """

        # Cache sfm result
        self._sfm_result = sfm_result

        self.orientation_method = "none"
        self.center_method = "poses"
        self.scale_factor = 1.0
        self.auto_scale_poses = True

        valid_camera_idxs = sorted(self._sfm_result.get_valid_camera_indices())
        self._gaussiansplatting_idx_to_camera_idx = {gs_i: i for gs_i, i in enumerate(valid_camera_idxs)}
        #   the number of images with estimated poses，not the number of images provided to GTSFM
        self._num_valid_cameras = len(valid_camera_idxs)

        self._images = [images[i] for gs_i, i in self._gaussiansplatting_idx_to_camera_idx.items()]
        
        # Get actual image dimensions from the Image objects
        self.actual_img_dims = [(images[i].height, images[i].width) for gs_i, i in self._gaussiansplatting_idx_to_camera_idx.items()]
        
        self._intrinsics = [self._sfm_result.get_camera(i).calibration().K() for gs_i, i in self._gaussiansplatting_idx_to_camera_idx.items()]
        
        self._dataparser_outputs = self._generate_dataparser_outputs_from_gtsfm_data(self._sfm_result, self._images)

        self._camtoworlds = self._dataparser_outputs['cameras']['camera_to_worlds'].numpy()
        
        self._scene_scale = self._dataparser_outputs['dataparser_scale']
        self.transform_matrix = self._dataparser_outputs['transform_matrix']
        self.points = self._dataparser_outputs['point_cloud']
        self.point_colors = self._dataparser_outputs['rgb']
        
        if self.points is not None:
            logger.info("Applying the same orientation and scaling to 3D points...")
            points_torch = torch.from_numpy(self.points)
            transform_matrix_torch = self.transform_matrix.float()
            points_homogeneous = F.pad(points_torch, (0, 1), "constant", 1.0)
            transform_4x4 = torch.cat([transform_matrix_torch, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=transform_matrix_torch.device)], axis=0)
            points_transformed = (transform_4x4 @ points_homogeneous.T).T
            points_transformed[:, :3] *= self._scene_scale
            self.points = points_transformed[:, :3].numpy()
            logger.info("3D points transformed successfully.")
        


    def _generate_dataparser_outputs_from_gtsfm_data(self, gtsfm_data: GtsfmData, images: List[Image]) -> Dict:
        """Processes an in-memory GtsfmData object to generate the required outputs
        Args:
            gtsfm_data: sparse multiview reconstruction result
            images: input images (H, W, C) to GTSFM
        Returns:
            """
        
        wTi_list = [gtsfm_data.get_camera(i).pose() for gs_i, i in self._gaussiansplatting_idx_to_camera_idx.items()]
        image_filenames = [images[i].file_name for gs_i, i in self._gaussiansplatting_idx_to_camera_idx.items()]
        calibrations = [gtsfm_data.get_camera(i).calibration() for gs_i, i in self._gaussiansplatting_idx_to_camera_idx.items()]
   
        tracks = gtsfm_data.get_tracks()
        point_cloud = np.array([track.point3() for track in tracks], dtype=np.float32)
   
        colors = []
        for track in tracks:
            r, g, b = image_utils.get_average_point_color(track, images)
            colors.append([r, g, b])

        rgb = np.array(colors, dtype=np.float32) /255.0
        
        return self._process_scene_data(wTi_list, calibrations, point_cloud, rgb)
   
   
    def _process_scene_data(self, wTi_list, calibrations, point_cloud, rgb) -> Dict:
        """Processing logic for data from any source
        Args:
            wTi_list: camera poses camera-to-worls
            calibrations: intrinsics for all valid images
            point_cloud: sfm points in the 3D space
            rgb: colors associated with sfm points

        Returns: 
            Dictionary containing
                "cameras": dictionary with camera intrinsics, poses, and distortion parameters
                "dataparser_scale": scaling factor to reduce the scene to a unit scale
                "transform_matrix": transformation matrix from orienting and centering the poses
                "point_cloud": sfm points in the 3D space
                "rgb": colors associated with sfm points
            """
        poses = np.stack([wTi.matrix() for wTi in wTi_list])
        poses = torch.from_numpy(poses.astype(np.float32))
    
        poses, transform_matrix = auto_orient_and_center_poses(
            poses,
            method=self.orientation_method,
            center_method=self.center_method,
        )

        # Perform auto-scaling
        scale_factor = 1.0
        if self.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.scale_factor

        poses[:, :3, 3] *= scale_factor
    
        # Intrinsics from calibration
        fx_list, fy_list, cx_list, cy_list = [], [], [], []
        height_list, width_list = [], []
        distortion_params_list = []

        for i in range(len(calibrations)):
            fx = float(calibrations[i].fx())
            fy = float(calibrations[i].fy())
            cx = float(calibrations[i].px())
            cy = float(calibrations[i].py())
            
   
            distortion_params = torch.tensor(
                [
                    calibrations[i].k1(),
                    calibrations[i].k2(),
                    0.0, 0.0, 0.0, 0.0,
                ],
                dtype=torch.float32,
            )
   
            height, width = self.actual_img_dims[i]

            fx_list.append(fx)
            fy_list.append(fy)
            cx_list.append(cx)
            cy_list.append(cy)
            height_list.append(int(height))
            width_list.append(int(width))
            distortion_params_list.append(distortion_params)
   
        cameras = {
            'fx': torch.tensor(fx_list, dtype=torch.float32),
            'fy': torch.tensor(fy_list, dtype=torch.float32),
            'cx': torch.tensor(cx_list, dtype=torch.float32),
            'cy': torch.tensor(cy_list, dtype=torch.float32),
            'height': torch.tensor(height_list, dtype=torch.int32),
            'width': torch.tensor(width_list, dtype=torch.int32),
            'camera_to_worlds': poses[:, :3, :4],
            'distortion_params': torch.stack(distortion_params_list, dim=0)
        }

        return {
            "cameras": cameras,
            "dataparser_scale": scale_factor,
            "transform_matrix": transform_matrix,
            "point_cloud": point_cloud,
            "rgb": rgb
        }

    def __len__(self) -> int:
        """
        Returns:
            the length of the dataset.
        """
        return self._num_valid_cameras

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get data to Gaussian splatting

        Args:
            index: the reference image ID of the train data

        Returns:
            Dictionary containing:
                "K": camera intrinsic matrix for a particular image
                "camtoworld": camera-to-world matrix for a particular image
                "image": the 2D image
                "image_id": index of the image
        """      
        image = self._images[index].value_array  

        K = self._intrinsics[index]
        distortion_params = self._dataparser_outputs['cameras']['distortion_params'][index].numpy()

        # Undistort the image and update intrinsics
        if np.any(distortion_params):
            K, image = _undistort_image(distortion_params, image, K)
        
        c2w = self._dataparser_outputs['cameras']['camera_to_worlds'][index].numpy()
        
        image = image.astype(np.float32) / 255.0

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(c2w).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": torch.tensor(index, dtype=torch.int32),
        }
        return data
