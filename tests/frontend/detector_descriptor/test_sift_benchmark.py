"""Validate RootSIFT experiment adapters against the existing implementation."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gtsfm.frontend.descriptor.rootsift import RootSIFTDescriptor
from gtsfm.frontend.descriptor.sift import SIFTDescriptor
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
from gtsfm.loader.olsson_loader import OlssonLoader
from scripts.benchmark_sift_variants import LegacyRootSIFTDetectorDescriptor, NormalizedRootSIFTDetectorDescriptor


def test_root_variants_preserve_keypoints_and_match_reference() -> None:
    """Check reference equivalence and exercise the native matcher on both variants."""
    dataset = Path(__file__).resolve().parents[2] / "data/set1_lund_door"
    image = OlssonLoader(str(dataset), max_resolution=320).get_image(0)
    keypoints, descriptors = SIFTDetectorDescriptor(max_keypoints=100).detect_and_describe(image)
    assert len(keypoints) > 0

    for adapter in (NormalizedRootSIFTDetectorDescriptor, LegacyRootSIFTDetectorDescriptor):
        actual_keypoints, actual = adapter(max_keypoints=100).detect_and_describe(image)
        assert actual_keypoints == keypoints
        if adapter is NormalizedRootSIFTDetectorDescriptor:
            # Reference normalizer consumes the *same* original descriptors.
            with patch.object(SIFTDescriptor, "describe", return_value=descriptors):
                expected = RootSIFTDescriptor().describe(image, keypoints)
        else:
            expected = RootSIFTDescriptor().describe(image, keypoints)
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
        assert actual.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
        # This fails without the float32 conversion under NumPy 2.
        matches = TwoWayMatcher(ratio_test_threshold=0.8).match(
            keypoints, keypoints, actual, actual, image.shape, image.shape
        )
        assert len(matches) > 0
        np.testing.assert_array_equal(matches[:, 0], matches[:, 1])


def test_benchmark_continues_after_failed_variant(tmp_path: Path) -> None:
    """A failed reconstruction must not prevent later extractors from being tested."""
    from scripts.benchmark_sift_variants import main

    args = ["benchmark", "--output-root", str(tmp_path), "--methods", "opencv", "colmap"]
    outcomes = [subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)]
    with (
        patch("sys.argv", args),
        patch("subprocess.check_output", return_value="test-commit"),
        patch("importlib.metadata.version", return_value="test-version"),
        patch("subprocess.run", side_effect=outcomes) as run,
    ):
        with pytest.raises(RuntimeError, match="opencv"):
            main()
    assert run.call_count == 2
    assert json.loads((tmp_path / "opencv.execution.json").read_text())["returncode"] == 1
    assert json.loads((tmp_path / "colmap.execution.json").read_text())["returncode"] == 0


@pytest.mark.parametrize("separate_intrinsics", [False, True])
def test_benchmark_forwards_dataset_settings(tmp_path: Path, separate_intrinsics: bool) -> None:
    """AstroVision, calibration sharing, lookahead, and bounded rotation reach the runner."""
    from scripts.benchmark_sift_variants import main

    args = [
        "benchmark",
        "--output-root",
        str(tmp_path),
        "--methods",
        "opencv",
        "--loader",
        "astrovision",
        "--max-frame-lookahead",
        "15",
        "--uncertified-rotation",
    ]
    if separate_intrinsics:
        args.append("--separate-intrinsics")
    with (
        patch("sys.argv", args),
        patch("subprocess.check_output", return_value="test-commit"),
        patch("importlib.metadata.version", return_value="test-version"),
        patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
    ):
        main()
    command = run.call_args.args[0]
    assert command[command.index("--loader") + 1] == "astrovision"
    assert command[command.index("--max_frame_lookahead") + 1] == "15"
    assert ("--share_intrinsics" in command) != separate_intrinsics
    assert any("UncertifiedRotationAveraging" in argument for argument in command)


def test_benchmark_reuses_only_identical_completed_runs(tmp_path: Path) -> None:
    """Resuming skips identical successful runs and refuses mismatched settings."""
    from scripts.benchmark_sift_variants import main

    args = ["benchmark", "--output-root", str(tmp_path), "--methods", "opencv"]
    with (
        patch("sys.argv", args),
        patch("subprocess.check_output", return_value="test-commit"),
        patch("importlib.metadata.version", return_value="test-version"),
        patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
    ):
        main()
        (tmp_path / "opencv").mkdir()
        run.reset_mock()
        main()
        run.assert_not_called()
        args.append("--separate-intrinsics")
        with pytest.raises(FileExistsError):
            main()
        run.assert_not_called()
