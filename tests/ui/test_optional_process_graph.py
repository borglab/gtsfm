"""Tests for best-effort process graph export."""

import logging
from pathlib import Path

import gtsfm.scene_optimizer as scene_optimizer


def test_missing_graphviz_does_not_stop_reconstruction(monkeypatch, caplog, tmp_path: Path) -> None:
    """A missing optional Graphviz executable should not raise an exception."""
    monkeypatch.setattr(scene_optimizer.shutil, "which", lambda _program: None)

    def fail_if_called(_self, _filepath: str) -> None:
        raise AssertionError("Graph export should not be attempted without Graphviz")

    monkeypatch.setattr(scene_optimizer.ProcessGraphGenerator, "save_graph", fail_if_called)

    with caplog.at_level(logging.WARNING):
        saved = scene_optimizer._save_optional_process_graph(tmp_path / "process_graph.svg")

    assert saved is False
    assert "skipping optional process graph" in caplog.text
    assert "Reconstruction will continue" in caplog.text


def test_graphviz_export_error_does_not_stop_reconstruction(monkeypatch, caplog, tmp_path: Path) -> None:
    """An unavailable Graphviz executable discovered at export time is nonfatal."""
    monkeypatch.setattr(scene_optimizer.shutil, "which", lambda _program: "/usr/local/bin/dot")

    def raise_file_not_found(_self, _filepath: str) -> None:
        raise FileNotFoundError("dot disappeared")

    monkeypatch.setattr(scene_optimizer.ProcessGraphGenerator, "save_graph", raise_file_not_found)

    with caplog.at_level(logging.WARNING):
        saved = scene_optimizer._save_optional_process_graph(tmp_path / "process_graph.svg")

    assert saved is False
    assert "reconstruction will continue" in caplog.text
