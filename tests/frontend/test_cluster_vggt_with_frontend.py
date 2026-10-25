"""Worker concurrency and local preset regression tests for frontend-based VGGT geometry."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import dask
import hydra
import numpy as np
import pytest
import torch
from dask.delayed import delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, SfmTrack
from omegaconf import OmegaConf

import gtsfm.cluster_optimizer.cluster_vggt as pipeline
import gtsfm.cluster_optimizer.cluster_vggt_with_frontend as frontend
from gtsfm.cluster_optimizer.cluster_optimizer_cacher import ClusterOptimizerCacher
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.frontend.geometry_transformer import GeometryTransformerOutput


@pytest.mark.parametrize("cached", [False, True])
def test_frontend_graph_colors_exports_and_returned_tracks_without_changing_geometry(tmp_path, monkeypatch, cached):
    scene = GtsfmData(8)
    for idx in (2, 7):
        scene.add_camera(idx, PinholeCameraCal3Bundler(Pose3(), Cal3Bundler(10, 0, 0, 1, 1)))
    track = SfmTrack(Point3(0, 0, 5))
    track.addMeasurement(2, np.array([1.0, 1.0]))
    track.addMeasurement(7, np.array([2.0, 1.0]))
    scene.add_track(track)
    pixels_a = np.zeros((4, 5, 3), dtype=np.uint8)
    pixels_b = pixels_a.copy()
    pixels_a[1, 1] = [240, 20, 40]
    pixels_b[1, 2] = [20, 160, 80]
    images = {2: Image(pixels_a, file_name="two.jpeg"), 7: Image(pixels_b, file_name="seven.jpeg")}
    context = SimpleNamespace(
        visibility_graph={(2, 7)},
        num_images=8,
        global_v_corr_idxs_dict={},
        global_keypoints=[],
        loader=SimpleNamespace(image_filenames=lambda: [f"{i}.jpeg" for i in range(8)]),
        output_paths=SimpleNamespace(results=tmp_path, metrics=tmp_path / "metrics"),
        label="C",
        one_view_data_dict={i: SimpleNamespace(camera_gt=None) for i in range(8)},
        get_delayed_image_map=lambda: delayed(lambda: images)(),
    )
    optimizer = frontend.ClusterVGGTWithFrontend(
        correspondence_generator=Mock(), two_view_estimator=Mock(), use_model_cache=False
    )
    if cached:
        cacher = ClusterOptimizerCacher(optimizer)
        monkeypatch.setattr(cacher, "_load_result_from_cache", lambda _: scene)
        monkeypatch.setattr(
            optimizer, "create_computation_graph", Mock(side_effect=AssertionError("Cache hit ran geometry"))
        )
        graph = cacher.create_computation_graph(context)
    else:
        monkeypatch.setattr(frontend, "get_2d_tracks", lambda *_: [])
        monkeypatch.setattr(frontend, "_load_vggt_inputs", lambda *a, **kw: (None, None))
        monkeypatch.setattr(frontend, "_get_image_shapes", lambda *_: {})
        monkeypatch.setattr(frontend, "_run_vggt_geometry", lambda *a, **kw: None)
        monkeypatch.setattr(frontend, "_build_gtsfm_data_from_vggt_depth", lambda *a, **kw: scene)
        monkeypatch.setattr(frontend, "_run_cluster_ba", lambda *a, **kw: (scene, scene))
        graph = optimizer.create_computation_graph(context)
    result, *_ = dask.compute(graph.sfm_result, *graph.io_tasks, scheduler="synchronous")
    colored = result.get_track(0)
    assert (colored.r, colored.g, colored.b) == (130, 90, 60)
    np.testing.assert_array_equal(colored.point3(), track.point3())
    for k in range(2):
        assert colored.measurement(k)[0] == track.measurement(k)[0]
        np.testing.assert_array_equal(colored.measurement(k)[1], track.measurement(k)[1])
    assert result.get_image_info(2).name == "two.jpeg"
    assert result.get_image_info(7).shape == (4, 5)
    assert (scene.get_track(0).r, scene.get_track(0).g, scene.get_track(0).b) == (0, 0, 0)
    for subdir in (["vggt"] if cached else ["vggt", "vggt_pre_ba"]):
        rows = [
            line.split()
            for line in (tmp_path / subdir / "points3D.txt").read_text().splitlines()
            if not line.startswith("#")
        ]
        assert rows[0][4:7] == ["130", "90", "60"]


def _output(images):
    n, _, h, w = images.shape
    return GeometryTransformerOutput(
        device=torch.device("cuda"),  # Exercise cleanup with CPU tensors; no GPU required.
        dtype=torch.float32,
        images=images,
        extrinsic=torch.eye(4)[None, :3].expand(n, -1, -1),
        intrinsic=torch.eye(3)[None].expand(n, -1, -1),
        depth_map=torch.ones(n, h, w),
        depth_confidence=torch.ones(n, h, w),
        dense_points=torch.zeros(n, h, w, 3),
    )


def _run(transformer, cache_key=None):
    return frontend._run_vggt_geometry(
        torch.zeros(2, 3, 16, 16),
        torch.tensor([[0, 0, 16, 16, 16, 16]] * 2),
        transformer=transformer,
        image_indices=(3, 7),
        model_cache_key=cache_key,
    )


@pytest.mark.parametrize("cache_key", ["frontend-model", None])
def test_geometry_is_exclusive_through_model_loading_transfer_and_cleanup(monkeypatch, cache_key):
    monkeypatch.setattr(pipeline, "_VGGT_MODEL_CACHE", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    start = Barrier(16)
    events = []
    cached_model = object()

    def load(**kwargs):
        time.sleep(0.01)  # Cold model loading releases the GIL.
        return cached_model

    loader = Mock(side_effect=load)
    monkeypatch.setattr(pipeline, "load_model", loader)

    def predict(images, *, model, **kwargs):
        assert model is (cached_model if cache_key else None)
        events.append("predict")
        time.sleep(0.01)
        return _output(images)

    camera_from_matrices = frontend.torch_utils.camera_from_matrices

    def camera(*args, **kwargs):
        events.append("transfer")
        time.sleep(0.001)
        return camera_from_matrices(*args, **kwargs)

    monkeypatch.setattr(frontend.torch_utils, "camera_from_matrices", camera)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("cleanup"))

    def run(_):
        start.wait(timeout=10)
        return _run(SimpleNamespace(predict=predict), cache_key)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(run, range(16)))
    assert frontend._VGGT_PIPELINE_LOCK is pipeline._VGGT_PIPELINE_LOCK
    assert events == ["predict", "transfer", "transfer", "cleanup"] * 16
    assert loader.call_count == (1 if cache_key else 0)
    for result in results:
        assert set(result.cameras) == {3, 7}
        assert result.dense_points.shape == (2, 16, 16, 3)
        np.testing.assert_array_equal(result.depth_confidence, np.ones((2, 16, 16)))


@pytest.mark.parametrize("failure_stage", ["load", "predict", "transfer"])
def test_geometry_failure_releases_shared_guard(monkeypatch, failure_stage):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    resolver = Mock(return_value=None)
    predict = Mock(side_effect=_output)
    transformer = SimpleNamespace(predict=lambda images, **kwargs: predict(images))
    camera = Mock(return_value=None)
    {"load": resolver, "predict": predict, "transfer": camera}[failure_stage].side_effect = RuntimeError("failed")
    monkeypatch.setattr(frontend, "_resolve_vggt_model", resolver)
    monkeypatch.setattr(frontend.torch_utils, "camera_from_matrices", camera)
    with pytest.raises(RuntimeError, match="failed"):
        _run(transformer)
    assert pipeline._VGGT_PIPELINE_LOCK.acquire(blocking=False)
    pipeline._VGGT_PIPELINE_LOCK.release()


def test_16gb_preset_only_changes_cluster_sizes():
    with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
        base = hydra.compose(config_name="vggt_sift_frontend_megaloc_phototourism")
        local = hydra.compose(config_name="vggt_sift_frontend_megaloc_phototourism_16gb")
    assert base.graph_partitioner.max_cameras == 70
    assert local.graph_partitioner.max_cameras == 20
    assert local.graph_partitioner.min_cameras_to_partition == 12
    local.graph_partitioner = base.graph_partitioner
    assert OmegaConf.to_container(local) == OmegaConf.to_container(base)
