"""Local tests of Modal deployment orchestration; no remote resources are created."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from visualization import modal_deployment


@pytest.mark.parametrize("exit_code", [0, 1])
def test_deployment_builds_once_and_checks_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    """Build failures stay failures; successful builds must pass endpoint discovery."""
    launches = []
    discoveries = []

    def discover(*_args):
        discoveries.append(True)
        return {"endpoint": "https://example.modal.run", "api_key": "workspace-key"}

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(
            stdout=iter(["manifest unknown\n"] if exit_code else ["App deployed\n"]), wait=lambda: exit_code
        )

    monkeypatch.setattr(modal_deployment.subprocess, "Popen", launch)
    manager = modal_deployment.ModalDeploymentManager(discover)
    deployment = modal_deployment.ModalDeployment(id="test", gpu="L4", token_id="ak-test", token_secret="as-test")
    manager._run(deployment, tmp_path)
    assert len(launches) == 1
    assert str(tmp_path / "visualization" / "modal_app.py") in launches[0][0]
    if exit_code:
        assert deployment.status == "failed"
        assert not discoveries
        assert deployment.error
    else:
        assert deployment.status == "completed"
        assert discoveries == [True]
        assert deployment.endpoint == "https://example.modal.run"


def test_modal_app_definition_loads_without_a_custom_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the real Modal SDK definition locally without deploying any resources."""
    source = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("GTSFM_SOURCE_ROOT", str(source))
    monkeypatch.setenv("GTSFM_API_KEY", "local-definition-check")
    monkeypatch.setenv("GTSFM_REMOTE_API_KEY", "local-definition-check")
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    monkeypatch.delitem(sys.modules, "visualization.modal_app", raising=False)
    definition = importlib.import_module("visualization.modal_app")
    assert definition.app.name == "gtsfm-studio"
