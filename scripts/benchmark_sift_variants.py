"""Compare SIFT variants using one reconstruction configuration, without VLFeat.

The ``rootsift_legacy`` variant calls Ayush's existing standalone descriptor.
The ``rootsift`` control applies his normalization to the original descriptors,
retaining the detector's orientation and octave information.

Run with the repository on PYTHONPATH. Output includes commands, versions, logs,
config snapshots, and the runner's reconstruction metrics. COLMAP is used only
as a comparison feature extractor; all variants use the same OpenCV verifier.
"""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.rootsift import RootSIFTDescriptor
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class LegacyRootSIFTDetectorDescriptor(SIFTDetectorDescriptor):
    """Combine OpenCV SIFT keypoints with Ayush's existing standalone RootSIFT."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Recompute descriptors through the existing standalone implementation.

        Args:
            image: Input image.

        Returns:
            Selected keypoints and their RootSIFT descriptors.
        """
        keypoints, _ = super().detect_and_describe(image)
        # NumPy 2 promotes the legacy epsilon expression to float64; OpenCV matching requires float32.
        return keypoints, RootSIFTDescriptor().describe(image, keypoints).astype(np.float32)


class NormalizedRootSIFTDetectorDescriptor(SIFTDetectorDescriptor):
    """Apply Ayush's normalization while preserving OpenCV's original descriptors."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """L1-normalize and square-root the baseline descriptors.

        Args:
            image: Input image.

        Returns:
            Unchanged keypoints and normalized descriptors.
        """
        keypoints, descriptors = super().detect_and_describe(image)
        # Same operations and epsilon as RootSIFTDescriptor.describe().
        l1_norms = np.sum(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (l1_norms + np.finfo(float).eps)
        return keypoints, np.sqrt(descriptors).astype(np.float32)


VARIANTS = {
    "opencv": "gtsfm.frontend.detector_descriptor.sift.SIFTDetectorDescriptor",
    "colmap": "gtsfm.frontend.detector_descriptor.colmap_sift.ColmapSIFTDetectorDescriptor",
    "rootsift_legacy": "scripts.benchmark_sift_variants.LegacyRootSIFTDetectorDescriptor",
    "rootsift": "scripts.benchmark_sift_variants.NormalizedRootSIFTDetectorDescriptor",
}


def summarize(output: Path) -> dict:
    """Collect comparable BA metrics and the additional post-merge metrics.

    Args:
        output: Directory containing per-method runner output.

    Returns:
        Mapping from completed method names to their metrics.
    """
    results = {}
    filenames = (
        "bundle_adjustment_metrics",
        "merging_metrics_retriangulated",
        "retriever_metrics",
        "verifier_summary_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT",
        "total_summary_metrics",
    )
    for method in VARIANTS:
        metrics_dir = output / method / "results" / "metrics"
        if not (metrics_dir / "total_summary_metrics.json").exists():
            continue
        results[method] = {}
        for name in filenames:
            path = metrics_dir / f"{name}.json"
            if path.exists():
                results[method][name] = json.loads(path.read_text())[name]
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    """Run the requested variants sequentially and record execution metadata."""
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=repo / "tests/data/set1_lund_door")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--loader", choices=("olsson", "colmap"), default="olsson")
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=VARIANTS, default=["opencv", "colmap"])
    parser.add_argument("--max-resolution", type=int, default=1296)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    packages = ("numpy", "opencv-python", "pycolmap", "gtsam", "torch", "scipy", "dask", "distributed")
    metadata = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "python": sys.version,
        "versions": {package: importlib.metadata.version(package) for package in packages},
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = dict(os.environ, PYTHONPATH=str(repo), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    failures = []
    for method in args.methods:
        destination = output / method
        if destination.exists():
            raise FileExistsError(f"Use a fresh output directory: {destination}")
        command = [
            sys.executable,
            str(repo / "gtsfm/runner.py"),
            "--loader",
            args.loader,
            "--dataset_dir",
            str(args.dataset_dir.resolve()),
            "--config_name",
            "unified",
            "--share_intrinsics",
            "--num_workers",
            "1",
            "--threads_per_worker",
            "1",
            "--num_matched",
            "5",
            "--max_frame_lookahead",
            "10",
            "--worker_memory_limit",
            "16GB",
            "--output_root",
            str(destination),
            "--max_resolution",
            str(args.max_resolution),
            "cluster_optimizer.correspondence_generator.detector_descriptor.detector_descriptor_obj._target_="
            + VARIANTS[method],
            "cluster_optimizer.correspondence_generator.matcher.matcher_obj._target_="
            "gtsfm.frontend.matcher.twoway_matcher.TwoWayMatcher",
            "~cluster_optimizer.correspondence_generator.matcher.matcher_obj.features",
            "+cluster_optimizer.correspondence_generator.matcher.matcher_obj.ratio_test_threshold=0.8",
            "cluster_optimizer.dense_multiview_optimizer=null",
            "cluster_optimizer.gaussian_splatting_optimizer=null",
        ]
        if args.images_dir is not None:
            command.extend(["--images_dir", str(args.images_dir.resolve())])
        record = {"command": command, "method": method}
        record_path = output / f"{method}.execution.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Starting {method}: {output / (method + '.log')}", flush=True)
        start = time.perf_counter()
        with (output / f"{method}.log").open("w") as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, wall_seconds=time.perf_counter() - start)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Finished {method}: {record}", flush=True)
        if result.returncode:
            failures.append(method)
        else:
            summarize(output)
    if failures:
        raise RuntimeError(f"Failed benchmark variants: {', '.join(failures)}; see their execution records and logs")


if __name__ == "__main__":
    main()
