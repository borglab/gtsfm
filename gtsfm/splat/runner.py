"""Train Gaussian splats from an existing Studio COLMAP result."""

import argparse
from pathlib import Path

from visualization.reconstruction import splat_sources


def load_reconstruction(results_root: Path, relative: str):
    """Load registered photos at the resolution used by the saved cameras and tracks."""
    from gtsfm.common.gtsfm_data import GtsfmData
    from gtsfm.utils.images import resize_image
    from gtsfm.utils.io import load_image

    scene, sources = splat_sources(results_root, relative)
    reconstruction = GtsfmData.read_colmap(str(scene))
    images = {}
    for index in reconstruction.get_valid_camera_indices():
        info = reconstruction.get_image_info(index)
        image = load_image(sources[info.name])
        height, width = info.shape[:2]
        if (image.height, image.width) != (height, width):
            image = resize_image(image, height, width)
        images[index] = image
    return images, reconstruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--config_name", default="base_gs")
    parser.add_argument("--max_steps", type=int, default=7000)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max_steps must be positive")

    import torch
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    from gtsfm.splat.rendering import save_splats

    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian splatting requires a CUDA-capable NVIDIA GPU")
    images, reconstruction = load_reconstruction(args.results_root, args.scene)
    print(f"Splatifying {args.scene}: {len(images)} cameras, {reconstruction.number_tracks()} points", flush=True)
    with initialize_config_module(config_module="gtsfm.configs.gaussian_splatting", version_base=None):
        cfg = compose(config_name=args.config_name)
    cfg.gaussian_splatting_optimizer.cfg.max_steps = args.max_steps
    optimizer = instantiate(cfg.gaussian_splatting_optimizer)
    splats, _ = optimizer.splatify(images, reconstruction)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_splats(args.output_root, splats)
    print(f"Saved Gaussian splats to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
