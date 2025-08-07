import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gtsfm.utils.io import read_scene_data_from_colmap_format


# mimics the rotation_matrix_between function from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py
def get_rotation_matrix_from_two_vectors(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Get the rotation matrix that rotates vec1 to vec2.
    """
    a = vec1 / torch.linalg.norm(vec1)
    b = vec2 / torch.linalg.norm(vec2)
    v = torch.linalg.cross(a, b)

    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0])
        v = torch.linalg.cross(a, x)

    v = v / torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return torch.eye(3) + torch.sin(theta) * skew_sym_mat + (1 - torch.cos(theta)) * (skew_sym_mat @ skew_sym_mat)


# this function is taken from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py and has reduced functionality
def auto_orient_and_center_poses(
    poses: torch.Tensor,
    method: Literal["up", "none"] = "none",
    center_method: Literal["poses", "none"] = "poses",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orients and centers the poses.
    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.
    """
    origins = poses[..., :3, 3]
    mean_origin = torch.mean(origins, dim=0)
    
    translation = torch.zeros_like(mean_origin)
    if center_method == "poses":
        translation = mean_origin
    elif center_method == "none":
        pass
    else:
        raise ValueError(f"Unknown center method: {center_method}")

    R = torch.eye(3, device=poses.device)
    if method == "up":
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        R = get_rotation_matrix_from_two_vectors(up, torch.tensor([0.0, 1.0, 0.0], device=up.device))
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown orientation method: {method}")
    
    transform = torch.cat([R, R @ -translation[..., None]], dim=-1)
    poses_new = transform.to(poses.device) @ poses

    return poses_new, transform

# follows _undistort_image function from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/utils/dataloaders.py
def _undistort_image(
    distortion_params: np.ndarray, image: np.ndarray, K: np.ndarray
):
    assert distortion_params[3] == 0, (
        "We don't support the 4th Brown parameter for image undistortion, Only k1, k2, k3, p1, p2 can be non-zero."
    )
    # we rearrange the distortion parameters because OpenCV expects the order (k1, k2, p1, p2, k3)
    # see https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # The last two zeros are for k4 and k5, which are not used in this context.
    distortion_params_cv = np.array(
        [
            distortion_params[0],
            distortion_params[1],
            distortion_params[4],
            distortion_params[5],
            distortion_params[2],
            0, # k4 is not used
            0, # k5 is not used
            0, # k6 is not used
        ]
    )
    
    # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
    # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
    K_shifted = K.copy()
    K_shifted[0, 2] -= 0.5
    K_shifted[1, 2] -= 0.5
    
    if np.any(distortion_params_cv):
        newK, roi = cv2.getOptimalNewCameraMatrix(K_shifted, distortion_params_cv, (image.shape[1], image.shape[0]), 0)
        image = cv2.undistort(image, K_shifted, distortion_params_cv, None, newK)
    else:
        newK = K_shifted
        roi = 0, 0, image.shape[1], image.shape[0]
        
    # crop the image and update the intrinsics accordingly
    x, y, w, h = roi
    image = image[y : y + h, x : x + w]
    newK[0, 2] -= x
    newK[1, 2] -= y

    newK[0, 2] += 0.5
    newK[1, 2] += 0.5
    K = newK

    return K, image

@dataclass
class DataParserConfig:
    """Simplified config for our custom parser, mimicking Nerfstudio's behavior."""
    data: Path = Path()
    scale_factor: float = 1.0
    orientation_method: Literal["up","none"] = "none"
    center_method: Literal["poses", "none"] = "poses"
    auto_scale_poses: bool = True

@dataclass
class SimpleCameras:
    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    height: torch.Tensor
    width: torch.Tensor
    camera_to_worlds: torch.Tensor
    distortion_params: torch.Tensor

    def __post_init__(self):
        num_cameras = self.camera_to_worlds.shape[0]
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if not isinstance(val, torch.Tensor):
                setattr(self, f, torch.tensor(val))
            if len(getattr(self, f).shape) == 0:
                setattr(self, f, getattr(self,f).unsqueeze(0).repeat(num_cameras))

    def get_intrinsics_matrices(self) -> torch.Tensor:
        leading_shape = self.fx.shape 
        K = torch.zeros((*leading_shape, 3, 3), dtype=torch.float32)
        K[..., 0, 0] = self.fx.squeeze(-1)
        K[..., 1, 1] = self.fy.squeeze(-1)
        K[..., 0, 2] = self.cx.squeeze(-1)
        K[..., 1, 2] = self.cy.squeeze(-1)
        K[..., 2, 2] = 1.0
        return K

    def rescale_output_resolution(self, scaling_factor: float):
        self.fx *= scaling_factor
        self.fy *= scaling_factor
        self.cx *= scaling_factor
        self.cy *= scaling_factor
        self.height = (self.height * scaling_factor).int()
        self.width = (self.width * scaling_factor).int()

    def __len__(self):
        return self.camera_to_worlds.shape[0]

class DataParser:
    """Parser for cameras, image and points3D files.
    """
    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        normalize: bool = True,
        test_every: int = 6,
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.test_every = test_every
        if test_every == -1:
            self.test_every = 1e9
        
        
        self.config = DataParserConfig(
            data=self.data_dir,
            auto_scale_poses=normalize,
        )

        self.dataparser_outputs = self._generate_dataparser_outputs()
        
        self.image_names = self.dataparser_outputs['image_filenames']
        self.image_paths = [self.images_dir / f for f in self.image_names]
        self.camtoworlds = self.dataparser_outputs['cameras'].camera_to_worlds.numpy()
        
        # currently works for single camera model for simplicity
        self.camera_ids = [1] * len(self.image_names)
        self.Ks_dict = {1: self.dataparser_outputs['cameras'].get_intrinsics_matrices()[0].numpy()}
        self.imsize_dict = {1: (int(self.dataparser_outputs['cameras'].width[0]), int(self.dataparser_outputs['cameras'].height[0]))}
        self.params_dict = {1: self.dataparser_outputs['cameras'].distortion_params[0].numpy()}
        
        self.scene_scale = self.dataparser_outputs['dataparser_scale']
        self.transform_matrix = self.dataparser_outputs['transform_matrix']
        self.points = self.dataparser_outputs['point_cloud']
        self.point_colors = self.dataparser_outputs['rgb']
        
        if self.points is not None:
            print("Applying the same orientation and scaling to 3D points...")
            points_torch = torch.from_numpy(self.points)
            transform_matrix_torch = self.transform_matrix.float()
            points_homogeneous = F.pad(points_torch, (0, 1), "constant", 1.0)
            transform_4x4 = torch.cat([transform_matrix_torch, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=transform_matrix_torch.device)], axis=0)
            points_transformed = (transform_4x4 @ points_homogeneous.T).T
            points_transformed[:, :3] *= self.scene_scale
            self.points = points_transformed[:, :3].numpy()
            print("3D points transformed successfully.")
        
        
    def _generate_dataparser_outputs(self) -> Dict:
        print(self.data_dir, self.config.data)
        wTi_list, image_filenames, calibrations, point_cloud, rgb, img_dims = read_scene_data_from_colmap_format(str(self.data_dir))
        if point_cloud is not None:
            point_cloud = point_cloud.astype(np.float32)
            rgb = rgb.astype(np.float32)/255.0
        
        print(calibrations) 
        poses = np.stack([wTi.matrix() for wTi in wTi_list])
        poses = torch.from_numpy(poses.astype(np.float32))

        # Perform auto-orientation and centering
        poses, transform_matrix = auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Perform auto-scaling
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor
        # print('Image Filenames:', image_filenames)
        print('Calibrations:',calibrations[0], calibrations[0].fx(), calibrations[0].px(), calibrations[0].py(), calibrations[0].k1(), calibrations[0].k2())
        print('Image dims:',img_dims)
        # Intrinsics
        fx = float(calibrations[0].fx())
        fy = float(calibrations[0].fy())
        cx = float(calibrations[0].px())
        cy = float(calibrations[0].py())
        h_gtsfm = int(img_dims[0][0])
        w_gtsfm = int(img_dims[0][1])

        distortion_params = torch.tensor(
            [
                calibrations[0].k1(),
                calibrations[0].k2(),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )
        # print('Instrinsics:', fx, fy, cx, cy, height, width)
        # print('Distortion params:', distortion_params)

        first_image_path = self.images_dir / image_filenames[0]
        actual_image = cv2.imread(str(first_image_path))
        h_actual, w_actual, _ = actual_image.shape

        if h_gtsfm != h_actual or w_gtsfm != w_actual:
            print("Intrinsics are being scaled to match image resolution.")
            scale_h = h_actual / h_gtsfm
            scale_w = w_actual / w_gtsfm
            print('Scaling values:', scale_h, scale_w)
            fx *= scale_w
            fy *= scale_h
            cx *= scale_w
            cy *= scale_h
        
        height = h_actual
        width = w_actual

        # Creating a simplified Cameras object to hold and rescale data
        cameras = SimpleCameras(
            fx=fx, fy=fy, cx=cx, cy=cy,
            height=height, width=width,
            camera_to_worlds=poses[:, :3, :4],
            distortion_params=distortion_params
        )

        return {
            "image_filenames": image_filenames,
            "cameras": cameras,
            "dataparser_scale": scale_factor,
            "transform_matrix": transform_matrix,
            "point_cloud": point_cloud,
            "rgb": rgb
        }

class Dataset:
    def __init__(
        self,
        parser: DataParser,
        split: str = "train",
    ):
        self.parser = parser
        self.split = split
        
        indices = np.arange(1, len(self.parser.image_names)+1)
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0] - 1
        else:
            self.indices = indices[indices % self.parser.test_every == 0] - 1
        
        print(f"Created {split} dataset with {len(self.indices)} images.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        
        image_path = self.parser.image_paths[index]
        try:
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            if image.shape[2] == 4:
                image = image[..., :3]
        except FileNotFoundError:
            print(f"FATAL: Image not found at {image_path}")
            return {}
            
        K = self.parser.dataparser_outputs['cameras'].get_intrinsics_matrices()[index].numpy()
        distortion_params = self.parser.dataparser_outputs['cameras'].distortion_params.numpy()

        # Undistort the image and update intrinsics
        if np.any(distortion_params):
            K, image = _undistort_image(distortion_params, image, K)
        
        c2w = self.parser.dataparser_outputs['cameras'].camera_to_worlds[index].numpy()
        
        image = image.astype(np.float32) / 255.0

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(c2w).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": torch.tensor(index, dtype=torch.int32),
            "image_name": str(self.parser.image_names[index]),
        }
        return data
