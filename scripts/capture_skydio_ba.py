"""Capture immutable BA inputs before diagnosing the Skydio CI failure.

Diagnostic branch only: the solver and its parameters are unchanged. A bad result
is saved and reported before Dask attempts to transfer an unpickleable object.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from distributed import Client, Worker, WorkerPlugin

from gtsfm.runner import GtsfmRunner


def scene_summary(scene: Any) -> dict[str, Any]:
    """Inspect numeric fields and the native pickle round trip without modifying a scene."""
    bad = []
    for i, camera in scene.cameras().items():
        calibration = camera.calibration()
        fields = {
            "pose": camera.pose().matrix(),
            "calibration": calibration.vector(),
            "K": calibration.K(),
        }
        for name, value in fields.items():
            if not np.isfinite(value).all():
                bad.append({"camera": i, "field": name, "value": repr(value.tolist())})
    empty_tracks = 0
    for j, track in enumerate(scene.get_tracks()):
        empty_tracks += int(track.numberMeasurements() == 0)
        for name, value in {
            "point": np.asarray(track.point3()),
            "rgb": np.asarray([track.r, track.g, track.b]),
        }.items():
            if not np.isfinite(value).all():
                bad.append({"track": j, "field": name, "value": repr(value.tolist())})
    try:
        pickle.loads(pickle.dumps(scene, protocol=5))
        roundtrip = {"passed": True}
    except Exception as error:
        roundtrip = {"passed": False, "type": type(error).__name__, "message": str(error)}
    return {
        "cameras": len(scene.cameras()),
        "tracks": scene.number_tracks(),
        "empty_tracks": empty_tracks,
        "nonfinite_field_count": len(bad),
        "bad_fields": bad[:40],
        "pickle_roundtrip": roundtrip,
    }


class CaptureBAPlugin(WorkerPlugin):
    """Install an observational capture around multi-view BA in each native Dask worker."""

    def __init__(self, directory: str) -> None:
        """Choose the shared local capture directory."""
        self.directory = directory

    def setup(self, worker: Worker) -> None:
        """Wrap multi-view BA, retaining the exact graph, inputs and optimizer state."""
        from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer

        original = BundleAdjustmentOptimizer._BundleAdjustmentOptimizer__optimize_and_recover
        root = Path(self.directory)
        root.mkdir(parents=True, exist_ok=True)

        def capture(optimizer: Any, initial_data: Any, graph: Any, ordering_type: str) -> Any:
            """Record one BA invocation and fail explicitly on an invalid output."""
            if len(initial_data.cameras()) <= 2:
                return original(optimizer, initial_data, graph, ordering_type)
            prefix = root / f"ba-{os.getpid()}-{time.time_ns()}"
            payload = {
                "initial_data": initial_data,
                "graph": graph,
                "optimizer_state": vars(optimizer).copy(),
                "ordering_type": ordering_type,
            }
            with prefix.with_suffix(".pickle").open("wb") as stream:
                pickle.dump(payload, stream, protocol=5)
            metadata = {"input": scene_summary(initial_data), "graph_factors": graph.size()}
            try:
                result = original(optimizer, initial_data, graph, ordering_type)
                metadata["output"] = scene_summary(result[0])
                metadata["final_error"] = repr(result[2])
            except Exception as error:
                metadata["exception"] = {"type": type(error).__name__, "message": str(error)}
                prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
                raise
            prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
            output = metadata["output"]
            if output["nonfinite_field_count"] or not output["pickle_roundtrip"]["passed"]:
                print("GTSFM_BA_INVALID_RESULT=" + json.dumps(metadata), flush=True)
                raise RuntimeError(f"Captured invalid BA result: {prefix.name}")
            print("GTSFM_BA_CAPTURE=" + prefix.name, flush=True)
            return result

        BundleAdjustmentOptimizer._BundleAdjustmentOptimizer__optimize_and_recover = capture


class CaptureRunner(GtsfmRunner):
    """Run the existing pipeline with diagnostic worker instrumentation."""

    def _create_dask_client(self) -> Client:
        """Create the original client and install the capture plugin before any tasks."""
        client = super()._create_dask_client()
        client.register_plugin(CaptureBAPlugin(os.environ["GTSFM_BA_CAPTURE_DIR"]), name="capture-ba")
        return client


if __name__ == "__main__":
    CaptureRunner(override_args=sys.argv[1:]).run()
