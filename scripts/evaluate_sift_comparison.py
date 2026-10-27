"""Evaluate exported SIFT benchmark poses against a fixed set of reference images.

Missing reconstructed cameras count as failures. Filenames establish camera
correspondence, independent of COLMAP image IDs or export ordering. Relative
pose errors are invariant to the reconstruction's global similarity transform.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pycolmap
from gtsam import Pose3, Rot3

from gtsfm.utils.metrics import (
    get_all_relative_rotations_translations,
    get_relative_rotation_angles,
    get_relative_translation_angles,
    pose_auc,
)


def read_poses(model: Path) -> dict[str, Pose3]:
    """Read camera-to-world poses keyed by image filename.

    Args:
        model: Directory containing a COLMAP-format reconstruction.

    Returns:
        Image filenames mapped to camera-to-world poses.
    """
    reconstruction = pycolmap.Reconstruction(str(model))
    return {
        image.name: Pose3(Rot3(image.cam_from_world.rotation.matrix()), image.cam_from_world.translation).inverse()
        for image in reconstruction.images.values()
    }


def evaluate(reference: dict[str, Pose3], estimated: dict[str, Pose3]) -> dict:
    """Compute relative pose AUC over every reference-image pair.

    Args:
        reference: Fixed reference image set and poses.
        estimated: Reconstructed poses, possibly missing reference images.

    Returns:
        AUC percentages, registration coverage, and per-pair errors.
    """
    names = sorted(reference)
    if len(names) < 2:
        raise ValueError("At least two reference images are needed")
    gt = {i: reference[name] for i, name in enumerate(names)}
    poses = {i: estimated.get(name) for i, name in enumerate(names)}
    rotations, translations = get_all_relative_rotations_translations(gt)
    r_errors = np.array(get_relative_rotation_angles(rotations, poses, include_none=True))
    t_errors = np.array(get_relative_translation_angles(translations, poses, include_none=True))
    errors = np.maximum(r_errors, t_errors)
    errors = np.nan_to_num(errors, nan=np.inf)
    thresholds = (1.0, 2.5, 5.0, 10.0, 20.0)
    auc = pose_auc(errors, thresholds)
    return {
        "reference_images": len(names),
        "registered_images": sum(name in estimated for name in names),
        "missing_images": [name for name in names if name not in estimated],
        "evaluated_pairs": len(errors),
        "missing_pairs": int(np.isinf(errors).sum()),
        "auc_percent": {str(t): round(float(a) * 100, 4) for t, a in zip(thresholds, auc)},
        "pair_errors": [
            {"images": [names[i], names[j]], "pose_error_deg": float(error) if np.isfinite(error) else None}
            for (i, j), error in zip(itertools.combinations(range(len(names)), 2), errors)
        ],
    }


def main() -> None:
    """Evaluate completed benchmark exports and save their full-reference metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    reference = read_poses(args.reference_model)
    output = {}
    for method in ("opencv", "colmap", "rootsift_legacy", "rootsift"):
        results = args.output_root / method / "results"
        if not (results / "metrics/total_summary_metrics.json").exists():
            continue
        output[method] = {}
        for stage in ("ba_output", "merged_retriangulated"):
            output[method][stage] = evaluate(reference, read_poses(results / stage))
            summary = {key: value for key, value in output[method][stage].items() if key != "pair_errors"}
            print(method, stage, summary)
    (args.output_root / "full-reference-metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
