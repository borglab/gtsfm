"""Tests for the Modal workspace import path."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def test_modal_app_import_in_container_does_not_need_a_local_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.delenv("GTSFM_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("GTSFM_REMOTE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("visualization.modal_app", None)

    modal_app = importlib.import_module("visualization.modal_app")

    assert modal_app.app.name == "gtsfm-studio"
    assert modal_app.gtsfm_workspace is not None


def test_modal_app_import_locally_requires_a_source_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    monkeypatch.delenv("GTSFM_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("GTSFM_REMOTE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("visualization.modal_app", None)

    with pytest.raises(RuntimeError, match="GTSFM_SOURCE_ROOT must point to a GTSFM source checkout"):
        importlib.import_module("visualization.modal_app")


class _FakeRequest:
    def __init__(self, method: str, path: str) -> None:
        self.method = method
        self.url = type("URL", (), {"path": path})()


def test_modal_workspace_allows_unauthenticated_catalog_and_scene_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    sys.modules.pop("visualization.modal_app", None)
    from visualization.modal_app import _is_public_workspace_read

    assert _is_public_workspace_read(_FakeRequest("GET", "/api/configuration"))
    assert _is_public_workspace_read(_FakeRequest("GET", "/api/samples"))
    assert _is_public_workspace_read(_FakeRequest("GET", "/api/scenes"))
    assert _is_public_workspace_read(_FakeRequest("GET", "/data/run/points3D.txt"))
    assert not _is_public_workspace_read(_FakeRequest("POST", "/api/jobs"))
    assert not _is_public_workspace_read(_FakeRequest("GET", "/api/jobs"))
