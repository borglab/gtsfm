"""Run Pi3 reconstruction per cluster using a saved cluster_tree.pkl.

Writes COLMAP text outputs under:
  <output_root>/results/.../<model_name>
matching the cluster tree directory structure.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image
from torchvision import transforms

import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from gtsfm.common.outputs import prepare_output_paths
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.utils.tree import PreOrderIter, Tree
_PI3_PROJECT_ROOT = Path("/nethome/xzhang979/nvme/gtsfm/thirdparty/Pi3")
if not _PI3_PROJECT_ROOT.is_dir():
    _PI3_PROJECT_ROOT = Path(__file__).resolve().parents[3] / "thirdparty" / "Pi3"
if str(_PI3_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PI3_PROJECT_ROOT))

from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pi3 on clusters from a saved ClusterTree.")
    parser.add_argument("--cluster_tree_path", type=str, required=True, help="Path to cluster_tree.pkl")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset root (used for loader).")
    parser.add_argument("--images_root", type=str, default=None, help="Root directory for images.")
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Base output directory (results will be in <output_root>/results/...).",
    )
    parser.add_argument("--config_name", type=str, default="vggt", help="Config in gtsfm/configs for loader.")
    parser.add_argument("--max_resolution", type=int, default=None, help="Optional loader max resolution override.")
    parser.add_argument("--model_name", type=str, default="pi3", help="Per-cluster output model folder name.")
    parser.add_argument("--min_images", type=int, default=2, help="Skip clusters with fewer images.")
    parser.add_argument("--run_leaf", action="store_true", default=True, help="Run on leaf clusters.")
    parser.add_argument("--run_parent", action="store_true", default=True, help="Run on non-leaf clusters.")
    parser.add_argument("--run_root", action="store_true", default=True, help="Run on root cluster.")
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        default=True,
        help="Recompute even if output already exists.",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--focal_length", type=float, default=None, help="Optional SIMPLE_PINHOLE focal length.")
    parser.add_argument(
        "--pixel_limit",
        type=int,
        default=255000,
        help="Target max image pixels after resizing (same behavior as Pi3 examples).",
    )
    return parser.parse_args()


def _build_loader(config_name: str, dataset_dir: str, images_dir: str | None, max_resolution: int | None):
    overrides: list[str] = [f"loader.dataset_dir={dataset_dir}"]
    if images_dir is not None:
        overrides.append(f"loader.images_dir={images_dir}")
    if max_resolution is not None:
        overrides.append(f"loader.max_resolution={max_resolution}")

    with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return instantiate(cfg.loader)


def _load_cluster_tree(cluster_tree_path: str):
    with open(cluster_tree_path, "rb") as f:
        return pickle.load(f)


def _resolve_image_paths(image_names: Sequence[str], images_root: str | None) -> list[str]:
    resolved_paths: list[str] = []
    for name in image_names:
        if os.path.isabs(name):
            resolved_paths.append(name)
        else:
            if images_root is None:
                raise ValueError("images_root is required when image filenames are relative.")
            resolved_paths.append(os.path.join(images_root, name))
    return resolved_paths


def _iter_clusters_with_paths(cluster_tree) -> Iterable[tuple[tuple[int, ...], Sequence[tuple[int, int]], bool]]:
    path_tree: Tree[tuple[tuple[int, ...], Sequence[tuple[int, int]]]] = cluster_tree.map_with_path(
        lambda path, visibility_graph: (path, visibility_graph)
    )
    for node in PreOrderIter(path_tree):
        path, visibility_graph = node.value
        yield path, visibility_graph, node.is_leaf()


def _should_run_cluster(path: tuple[int, ...], is_leaf: bool, args: argparse.Namespace) -> bool:
    if path == () and not args.run_root:
        return False
    if is_leaf and args.run_leaf:
        return True
    if (not is_leaf) and args.run_parent:
        return True
    return False


def _load_images_from_paths(
    image_paths: Sequence[str], pixel_limit: int, device: torch.device
) -> tuple[torch.Tensor, tuple[int, int]]:
    sources: list[Image.Image] = []
    for image_path in image_paths:
        sources.append(Image.open(image_path).convert("RGB"))

    if not sources:
        raise ValueError("No images loaded for cluster.")

    first_img = sources[0]
    w_orig, h_orig = first_img.size
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1.0
    w_target, h_target = w_orig * scale, h_orig * scale
    k, m = round(w_target / 14), round(h_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / max(1, m) > w_target / max(1.0, h_target):
            k -= 1
        else:
            m -= 1
    target_w = max(1, k) * 14
    target_h = max(1, m) * 14

    to_tensor = transforms.ToTensor()
    tensors: list[torch.Tensor] = []
    for img in sources:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        tensors.append(to_tensor(resized))
    imgs = torch.stack(tensors, dim=0).to(device)
    return imgs, (target_w, target_h)


def _write_colmap_text(
    colmap_dir: str,
    image_names: Sequence[str],
    camera_poses_wTc: np.ndarray,
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    image_size: tuple[int, int],
    focal_length: Optional[float],
) -> None:
    if len(image_names) != camera_poses_wTc.shape[0]:
        raise ValueError("Number of image names must match number of camera poses.")
    os.makedirs(colmap_dir, exist_ok=True)

    width, height = image_size
    if focal_length is None:
        focal_length = 0.5 * (width + height)

    cameras = {}

    images = {}
    for idx, name in enumerate(image_names, start=1):
        cameras[idx] = colmap_io.Camera(
            id=idx,
            model="SIMPLE_PINHOLE",
            width=int(width),
            height=int(height),
            params=np.array([float(focal_length), float(width) / 2.0, float(height) / 2.0]),
        )
        wTc = camera_poses_wTc[idx - 1]
        r_wc = wTc[:3, :3]
        t_wc = wTc[:3, 3]
        r_cw = r_wc.T
        t_cw = -r_cw @ t_wc
        qvec = colmap_io.rotmat2qvec(r_cw)
        images[idx] = colmap_io.Image(
            id=idx,
            qvec=qvec,
            tvec=t_cw,
            camera_id=idx,
            name=name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )

    points3d = {}
    for idx, (xyz, rgb) in enumerate(zip(points_xyz, points_rgb), start=1):
        points3d[idx] = colmap_io.Point3D(
            id=idx,
            xyz=xyz,
            rgb=rgb,
            error=0.0,
            image_ids=np.zeros((0,), dtype=np.int32),
            point2D_idxs=np.zeros((0,), dtype=np.int32),
        )

    colmap_io.write_model(cameras, images, points3d, path=colmap_dir, ext=".txt")


def _setup_model(device: torch.device, ckpt: str | None) -> Pi3:
    if ckpt is not None:
        model = Pi3().to(device).eval()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    return model


def _run_pi3_on_cluster(
    model: Pi3,
    device: torch.device,
    image_paths: Sequence[str],
    image_names: Sequence[str],
    output_dir: Path,
    focal_length: float | None,
    pixel_limit: int,
) -> None:
    imgs, (width, height) = _load_images_from_paths(image_paths, pixel_limit=pixel_limit, device=device)
    if imgs.shape[0] == 0:
        raise ValueError("No images available after preprocessing.")

    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device=device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs[None])
    else:
        with torch.no_grad():
            res = model(imgs[None])

    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    points_xyz = res["points"][0][masks].cpu().numpy()
    points_rgb = (imgs.permute(0, 2, 3, 1)[masks].cpu().numpy() * 255.0).round().astype(np.uint8)

    _write_colmap_text(
        colmap_dir=str(output_dir),
        image_names=image_names,
        camera_poses_wTc=res["camera_poses"][0].cpu().numpy(),
        points_xyz=points_xyz,
        points_rgb=points_rgb,
        image_size=(width, height),
        focal_length=focal_length,
    )


def main() -> None:
    args = _parse_args()
    cluster_tree = _load_cluster_tree(args.cluster_tree_path)

    loader = _build_loader(args.config_name, args.dataset_dir, args.images_root, args.max_resolution)
    image_names = loader.image_filenames()
    images_root = args.images_root
    if images_root is None and hasattr(loader, "_images_dir"):
        images_root = getattr(loader, "_images_dir")
    image_paths = _resolve_image_paths(image_names, images_root)

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = _setup_model(device, args.ckpt)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "pi3_cluster.log"

    def log(message: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
        print(message)

    for path, visibility_graph, is_leaf in _iter_clusters_with_paths(cluster_tree):
        if not _should_run_cluster(path, is_leaf, args):
            continue

        image_indices = sorted(visibility_graph_keys(visibility_graph))
        if len(image_indices) < args.min_images:
            log(f"Skipping {path}: only {len(image_indices)} images.")
            continue
        if max(image_indices) >= len(image_names):
            log(f"Skipping {path}: image index out of range (max={max(image_indices)}).")
            continue

        cluster_image_paths = [image_paths[idx] for idx in image_indices]
        missing_paths = [p for p in cluster_image_paths if not os.path.exists(p)]
        if missing_paths:
            log(f"Skipping {path}: missing {len(missing_paths)} images.")
            continue

        cluster_image_names = [os.path.basename(image_names[idx]) for idx in image_indices]
        output_paths = prepare_output_paths(output_root, path)
        output_dir = output_paths.results / args.model_name
        if args.skip_existing and (output_dir / "cameras.txt").exists():
            log(f"Skipping {path}: output already exists at {output_dir}.")
            continue

        try:
            log(f"Running Pi3 for {path} -> {output_dir}")
            _run_pi3_on_cluster(
                model=model,
                device=device,
                image_paths=cluster_image_paths,
                image_names=cluster_image_names,
                output_dir=output_dir,
                focal_length=args.focal_length,
                pixel_limit=args.pixel_limit,
            )
        except Exception as exc:
            log(f"Failed {path}: {exc!r}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
