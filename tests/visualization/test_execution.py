"""Validation and persistence checks for saved execution settings."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from visualization.execution import ExecutionConfiguration, ExecutionStore


def configuration(name="Cyber", **settings):
    return ExecutionConfiguration(name=name, settings=settings)


def test_saved_settings_survive_restart_and_edits_are_explicit(tmp_path):
    path = tmp_path / "configurations.json"
    store = ExecutionStore(path)
    store.save(configuration(ssh_host="cyber", num_workers=3, threads_per_worker=2, worker_memory_limit="6GiB"))
    loaded = ExecutionStore(path).list()[0]
    assert loaded["settings"]["num_workers"] == 3
    assert loaded["settings"]["threads_per_worker"] == 2
    loaded["settings"]["num_workers"] = 7
    assert store.list()[0]["settings"]["num_workers"] == 3
    store.save(ExecutionConfiguration.model_validate(loaded))
    assert store.list()[0]["settings"]["num_workers"] == 7
    assert path.stat().st_mode & 0o777 == 0o600
    store.delete("Cyber")
    assert store.list() == []


@pytest.mark.parametrize(
    "settings",
    [
        {"api_key": "secret"},
        {"modal_token_secret": "secret"},
        {"num_workers": 0},
        {"ssh_port": 70000},
        {"cluster_config": "hidden.yaml"},
    ],
)
def test_store_rejects_credentials_invalid_resources_and_hidden_yaml(settings):
    with pytest.raises(ValidationError):
        configuration(**settings)


def test_corrupt_store_is_reported_without_overwrite(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("broken")
    with pytest.raises(ValueError):
        ExecutionStore(path).save(configuration())
    assert path.read_text() == "broken"


def test_concurrent_saves_keep_every_named_configuration(tmp_path):
    store = ExecutionStore(tmp_path / "config.json")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: store.save(configuration(name=f"Machine {n}")), range(12)))
    assert len(store.list()) == 12
    assert not list(tmp_path.glob("*.tmp"))

