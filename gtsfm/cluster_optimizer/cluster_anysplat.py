"""Feed forward Gaussian Splatting using the AnySplat model.
https://github.com/InternRobotics/AnySplat
Authors: Harneet Singh Khanuja
"""

import sys
from pathlib import Path
from typing import Any, List, Sequence

import cv2
import gtsam
import numpy as np
import torch
import torchvision  # type: ignore
from dask import delayed  # type: ignore
from dask.delayed import Delayed
from gtsam import Point3, Pose3, Rot3  # type: ignore

import gtsfm.common.types as gtsfm_types
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterOptimizerBase
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils import logger as logger_utils
from gtsfm.utils.anysplat import AnySplatReconstructionResult

HERE_PATH = Path(__file__).parent
ANYSPLAT_REPO_PATH = HERE_PATH.parent.parent / "thirdparty" / "AnySplat"

if ANYSPLAT_REPO_PATH.exists():
    # workaround for sibling import
    sys.path.insert(0, str(ANYSPLAT_REPO_PATH))
elif not ANYSPLAT_REPO_PATH.exists():
    raise ImportError(
        f"AnySplat is not initialized, could not find: {ANYSPLAT_REPO_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )


from src.misc.image_io import save_interpolated_video  # type: ignore
from src.model.model.anysplat import AnySplat  # type: ignore
from src.model.ply_export import export_ply  # type: ignore

logger = logger_utils.get_logger()


def _calibration_from_intrinsic(matrix: np.ndarray) -> gtsam.Cal3_S2:
    """Map a 3x3 intrinsic matrix to the corresponding GTSAM calibration type."""
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)


def _pose_from_extrinsic(matrix: np.ndarray) -> Pose3:
    """Convert a AnySplat extrinsic matrix to a Pose3."""
    if matrix.shape == (4, 4):
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
    elif matrix.shape == (3, 4):
        rotation = matrix[:, :3]
        translation = matrix[:, 3]
    else:
        raise ValueError(f"Unexpected extrinsic shape {matrix.shape}")
    return Pose3(Rot3(rotation), Point3(*translation))  # type: ignore


class ClusterAnySplat(ClusterOptimizerBase):
    """Class for AnySplat (feed forward GS implementation)"""

    def __init__(self):
        """."""
        super().__init__()
        self._model = None

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""
        return UiMetadata(
            display_name="AnySplat",
            input_products=("Key Images",),
            output_products=("Gaussian Splats", "Interpolated Video"),
            parent_plate="Cluster Optimizer",
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the AnySplat model to avoid unnecessary initialization."""
        if self._model is None:
            logger.info("⏳ Loading AnySplat model weights...")
            self._model = AnySplat.from_pretrained("lhjiang/anysplat")

    def __repr__(self) -> str:
        """Provide a readable summary of the optimizer configuration."""
        components = ["Model Name = AnySplat", "Input image preprocessed to (448,448)"]
        return "Feed Forward Gaussian Splatting(\n  " + ",\n  ".join(str(c) for c in components) + "\n)"

    def _aggregate_anysplat_metrics(
        self,
        gaussian_count: int,
        voxel_count: int,
    ) -> GtsfmMetricsGroup:
        """Capture simple runtime metrics for the front-end."""

        return GtsfmMetricsGroup(
            "anysplat_runtime_metrics",
            [
                GtsfmMetric("total_pixel wise_gaussians", gaussian_count),
                GtsfmMetric("total_voxels", voxel_count),
                GtsfmMetric("Percent remaining after voxelize", voxel_count / gaussian_count),
            ],
        )

    def _generate_splats(
        self, images: List[Image], keys: list, image_names: Sequence[str]
    ) -> AnySplatReconstructionResult:
        """
        Apply AnySplat feed forward network to generate Gaussian splats.
        Args:
            images: List of all images.
        """

        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._model.to(device)
        processed_images = self._preprocess_images(images, device)
        _, _, _, height, width = processed_images.shape
        splats, pred_context_pose = model.inference((processed_images + 1) * 0.5)

        # Move results to CPU to avoid Dask serialization errors.
        for attr in ["means", "scales", "rotations", "harmonics", "opacities"]:
            setattr(splats, attr, getattr(splats, attr).cpu())
        pred_context_pose["extrinsic"] = pred_context_pose["extrinsic"].cpu()
        pred_context_pose["intrinsic"] = pred_context_pose["intrinsic"].cpu()
        decoder = model.decoder.cpu()

        gtsfm_data = GtsfmData(number_images=len(keys))
        # image_names_str = [str(name) for name in image_names] if image_names is not None else None
        for idx, key in enumerate(keys):
            intrinsic = pred_context_pose["intrinsic"][0][idx].numpy()
            calibration = _calibration_from_intrinsic(intrinsic)
            camera_cls = gtsfm_types.get_camera_class_for_calibration(calibration)

            extrinsic = pred_context_pose["extrinsic"][0][idx].numpy()

            pose = _pose_from_extrinsic(extrinsic)
            gtsfm_data.add_camera(key, camera_cls(pose, calibration))  # type: ignore
            # gtsfm_data.set_image_info(
            #     key,
            #     name=image_names_str[idx] if image_names_str is not None else None,
            #     shape=(height, width),
            # )
        return AnySplatReconstructionResult(gtsfm_data, splats, pred_context_pose, height, width, decoder)

    def _preprocess_images(self, images: List[Image], device):
        """
        Converts the data format from GtsfmData to AnySplat input
        """
        images_list = []
        for i, img in enumerate(images):
            height, width = img.shape[:2]
            img_array = img.value_array
            if width > height:
                new_height = 448
                new_width = int(width * (new_height / height))
            else:
                new_width = 448
                new_height = int(height * (new_width / width))
            img_array = cv2.resize(img_array, (new_width, new_height))

            # Center crop
            left = (new_width - 448) // 2
            top = (new_height - 448) // 2
            right = left + 448
            bottom = top + 448
            img_array = img_array[top:bottom, left:right]  # cropping
            img_tensor = torchvision.transforms.ToTensor()(img_array) * 2.0 - 1.0  # [-1, 1]
            images_list.append(img_tensor.to(device))
        images_tensor = torch.stack(images_list, dim=0).unsqueeze(0)
        return images_tensor.to(device)

    def _generate_interpolated_video(
        self,
        decoder_model: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        splats: Any,
        save_gs_files_path: str,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA not available, cannot generate interpolated video on CPU.")
            return

        # Move data to GPU
        decoder_model = decoder_model.to(device)
        extrinsics = extrinsics.to(device)
        intrinsics = intrinsics.to(device)
        for attr in ["means", "scales", "rotations", "harmonics", "opacities"]:
            setattr(splats, attr, getattr(splats, attr).to(device))

        b = 1  # AnySplat convention
        save_interpolated_video(
            extrinsics,
            intrinsics,
            b,
            height,
            width,
            splats,
            save_gs_files_path,
            decoder_model,
        )

    def _save_splats(self, splats, save_gs_files_path: Path) -> None:
        export_ply(
            splats.means[0],
            splats.scales[0],
            splats.rotations[0],
            splats.harmonics[0],
            splats.opacities[0],
            save_gs_files_path / "gaussian_splats.ply",
        )

    def create_computation_graph(
        self,
        num_images: int,
        one_view_data_dict,
        output_paths,
        loader,
        output_root: Path,
        visibility_graph,
        image_futures,
    ) -> tuple[list[Delayed], list[Delayed]]:
        """Create a Dask computation graph to process a cluster.

        Returns:
            - List of Delayed I/O tasks to be computed
            - List of Delayed metrics to be computed
        """

        del num_images, one_view_data_dict, image_futures, output_root  # unused in AnySplat pipeline

        io_tasks: List[Delayed] = []
        metrics_tasks: List[Delayed] = []

        keys = sorted(visibility_graph_keys(visibility_graph))

        image_filenames = loader.image_filenames()
        image_names = [str(image_filenames[idx]) for idx in keys]

        def _pack_images_for_anysplat(*images: Image) -> list[Image]:
            """Collect variadic image inputs into an ordered list."""

            return list(images)

        logger.info("Keys for AnySplat GS computation: %s", keys)
        selected_images = loader.get_key_images_as_delayed_map(keys)

        images = delayed(_pack_images_for_anysplat)(*selected_images.values())

        result_graph = delayed(self._generate_splats)(images, keys, image_names)

        total_gaussians = len(keys) * result_graph.height * result_graph.width  # type: ignore
        total_voxels = result_graph.splats.means.shape[1]  # type: ignore

        metrics_tasks.append(delayed(self._aggregate_anysplat_metrics)(total_gaussians, total_voxels))
        with self._output_annotation():
            io_tasks.append(
                delayed(self._generate_interpolated_video)(
                    result_graph.decoder,
                    result_graph.pred_context_pose["extrinsic"],  # type: ignore
                    result_graph.pred_context_pose["intrinsic"],  # type: ignore
                    result_graph.height,
                    result_graph.width,
                    result_graph.splats,
                    str(output_paths.results),
                )
            )
            io_tasks.append(delayed(self._save_splats)(result_graph.splats, output_paths.results))

        return io_tasks, metrics_tasks
