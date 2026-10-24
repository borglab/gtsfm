"""Focused checks for native Hydra LocalCluster constructor YAML."""

from pathlib import Path

import pytest

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

    def fake_instantiate(config):
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
