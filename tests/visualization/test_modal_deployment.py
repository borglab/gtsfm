"""Local Modal image-definition check; no deployment or endpoint requests are performed."""

import importlib
import sys
from pathlib import Path

import pytest


def test_modal_app_definition_loads_without_a_custom_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the real Modal SDK definition locally without deploying any resources."""
    source = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("GTSFM_SOURCE_ROOT", str(source))
    monkeypatch.setenv("GTSFM_API_KEY", "local-definition-check")
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    monkeypatch.delitem(sys.modules, "visualization.modal_app", raising=False)
    definition = importlib.import_module("visualization.modal_app")
    assert definition.app.name == "gtsfm-studio"
