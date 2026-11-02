"""Focused checks for native Hydra LocalCluster constructor YAML."""

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

import gtsfm.runner as runner_module
from gtsfm.runner import GtsfmRunner


def test_native_local_cluster_yaml_is_loaded_without_manual_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "_target_: distributed.LocalCluster\n"
        "n_workers: 2\n"
        "threads_per_worker: 3\n"
        "processes: false\n"
        "memory_limit: 2GiB\n"
        f"local_directory: {tmp_path / 'scratch'}\n"
        "dashboard_address: ':0'\n",
        encoding="utf-8",
    )
    config = GtsfmRunner._load_cluster_config(str(config_path))
    assert config["_target_"] == "distributed.LocalCluster"
    assert GtsfmRunner._cluster_config_is_local(str(config_path))


def test_legacy_cluster_yaml_remains_non_local() -> None:
    assert not GtsfmRunner._cluster_config_is_local("cluster.yaml")


def test_missing_cluster_yaml_fails_before_launch() -> None:
    with pytest.raises(ValueError, match="not found"):
        GtsfmRunner._load_cluster_config("does-not-exist.yaml")


def test_local_yaml_is_instantiated_authoritatively(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "_target_: distributed.LocalCluster\nn_workers: 2\nthreads_per_worker: 3\n", encoding="utf-8"
    )
    instantiated = []
    fake_cluster = object()
    fake_client = type("FakeClient", (), {"dashboard_link": "http://127.0.0.1:1"})

    def fake_instantiate(config, **kwargs):
        assert kwargs == {"_convert_": "all"}
        instantiated.append(config)
        return fake_cluster

    monkeypatch.setattr(runner_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(runner_module, "Client", lambda cluster: fake_client())
    runner = GtsfmRunner(["--cluster_config", str(config_path), "--dask_tmpdir", "ignored"])
    client = runner._create_dask_client()

    assert client.dashboard_link.endswith(":1")
    assert len(instantiated) == 1
    assert instantiated[0]["n_workers"] == 2
    assert instantiated[0]["threads_per_worker"] == 3
    assert "memory_limit" not in instantiated[0]


@pytest.mark.parametrize(
    "model", ["sift_front_end", "vggt_sift_frontend_megaloc", "synthetic_front_end", "scene_optimizer_unit_test_config"]
)
def test_frontend_preset_reaches_cached_and_uncached_optimizer(model: str) -> None:
    """The UI/CLI selected matcher must be present in the effective optimizer config."""
    with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
        config = hydra.compose(model)
    runner = GtsfmRunner(
        ["--correspondence_generator_config_name", "sift_brute_force", "--verifier_config_name", "loransac_8pt"]
    )
    runner._set_frontend_overwrites(config)
    optimizer = config.cluster_optimizer
    if "optimizer" in optimizer:
        optimizer = optimizer.optimizer
    assert optimizer.correspondence_generator.matcher.matcher_obj.search_method == "brute_force"
    estimator = optimizer.two_view_estimator
    if "two_view_estimator_obj" in estimator:
        estimator = estimator.two_view_estimator_obj
    assert estimator.verifier.use_intrinsics_in_verification is False
    assert "search_method" in OmegaConf.to_yaml(config)


def test_frontend_preset_rejects_models_without_that_frontend() -> None:
    """An unsupported selection must fail rather than silently running another setting."""
    with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
        config = hydra.compose("vggt")
    runner = GtsfmRunner(["--correspondence_generator_config_name", "sift_brute_force"])
    with pytest.raises(ValueError, match="does not support"):
        runner._set_frontend_overwrites(config)
