"""Focused contract checks for Studio execution profiles."""

from pathlib import Path

import pytest

from visualization.execution_profiles import ExecutionProfileStore, apply_profile, validate_profile


def test_profile_round_trip_and_authoritative_resolution(tmp_path: Path) -> None:
    store = ExecutionProfileStore(tmp_path / "profiles.json")
    saved = store.save(
        {
            "workers": 2,
            "threads_per_worker": 3,
            "host_memory_limit": "8GiB",
            "scratch_directory": "/tmp/gtsfm",
            "selected_gpus": [],
        },
        name="cpu-fast",
    )
    assert store.get("cpu-fast") == saved
    resolved, profile = apply_profile(
        {
            "execution_profile": "cpu-fast",
            "num_workers": 1,
            "threads_per_worker": 1,
            "worker_memory_limit": "32GB",
            "hardware": "cpu",
        },
        store,
    )
    assert profile == saved
    assert resolved["num_workers"] == 2
    assert resolved["threads_per_worker"] == 3
    assert resolved["worker_memory_limit"] == "8GiB"
    assert resolved["resolved_execution_profile"] == saved


@pytest.mark.parametrize(
    "payload",
    [
        {"workers": True},
        {"workers": 1.5},
        {"schema_version": 99},
        {"name": 123},
        {"selected_gpus": ["cuda:0", "cuda:0"]},
        {"selected_gpus": ["bogus"]},
    ],
)
def test_profile_rejects_ambiguous_or_unsupported_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_profile(payload, name="invalid")


def test_profile_rejects_conflicting_manual_controls(tmp_path: Path) -> None:
    store = ExecutionProfileStore(tmp_path / "profiles.json")
    store.save({"workers": 2, "threads_per_worker": 1, "host_memory_limit": "4GB"}, name="cpu")
    with pytest.raises(ValueError, match="conflicts"):
        apply_profile({"execution_profile": "cpu", "num_workers": 4}, store)


def test_profile_name_alias_cannot_override_primary_selection(tmp_path: Path) -> None:
    store = ExecutionProfileStore(tmp_path / "profiles.json")
    store.save({"workers": 1, "threads_per_worker": 1, "host_memory_limit": "1GB"}, name="one")
    store.save({"workers": 2, "threads_per_worker": 1, "host_memory_limit": "1GB"}, name="two")
    with pytest.raises(ValueError, match="must identify the same"):
        apply_profile({"execution_profile": "one", "execution_profile_name": "two"}, store)
    with pytest.raises(ValueError, match="must be a profile name"):
        apply_profile({"execution_profile": 123}, store)
    resolved, profile = apply_profile({"execution_profile_name": "one"}, store)
    assert profile is not None and profile["name"] == "one"
    assert resolved["execution_profile_name"] == "one"
