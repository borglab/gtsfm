"""Unit tests to ensure CI benchmark artifact fnames are generated on-the-fly appropriately.

Authors: John Lambert
"""

from pathlib import Path

import gtsfm.evaluation.visualize_benchmark_comparison as dashboard_utils

SAMPLE_BENCHMARK_YAML_FPATH = Path(__file__).parent.parent / "data" / "sample_ci_benchmark" / "benchmark.yml"


def test_generate_artifact_fnames_from_workflow() -> None:
    """Verify that dynamically generated artifact names are correct, based on a dummy workflow YAML file."""
    zip_fnames = dashboard_utils.generate_artifact_fnames_from_workflow(workflow_yaml_fpath=SAMPLE_BENCHMARK_YAML_FPATH)

    expected_zip_fnames = [
        "deep_front_end-2011205_rc3-20-png-wget-astronet-1024-true.zip",
        "deep_front_end-door-12-12-JPG-test_data-olsson-loader-1296-true.zip",
        "deep_front_end-notre-dame-20-20-jpg-gdrive-colmap-loader-760-false.zip",
        "deep_front_end-skydio-8-8-jpg-gdrive-colmap-loader-760-true.zip",
        "deep_front_end-skydio-32-32-jpg-gdrive-colmap-loader-760-true.zip",
        "sift_front_end-2011205_rc3-65-png-wget-astronet-1024-true.zip",
        "sift_front_end-door-12-12-JPG-test_data-olsson-loader-1296-true.zip",
        "sift_front_end-palace-fine-arts-281-25-jpg-wget-olsson-loader-320-true.zip",
        "sift_front_end-skydio-8-8-jpg-gdrive-colmap-loader-760-true.zip",
        "sift_front_end-skydio-32-32-jpg-gdrive-colmap-loader-760-true.zip",
        "deep_front_end-skydio-501-15-jpg-wget-colmap-loader-760-true.zip",
    ]
    assert set(zip_fnames) == set(expected_zip_fnames)
