"""Unit tests for AnySplat glue.

Authors: Frank Dellaert
"""

import unittest
from unittest import mock

from gtsfm.cluster_optimizer.cluster_anysplat import ClusterAnySplat
from gtsfm.utils import anysplat


class _DummyAnySplatModel:
    """Barebones stub used to mimic the AnySplat interface in unit tests."""

    def to(self, device):  # noqa: D401 - simple passthrough API
        self.device = device
        return self


class ClusterAnySplatTest(unittest.TestCase):
    """Exercise the minimal AnySplat cluster bootstrapping logic."""

    def test_model_loader_invoked_once(self) -> None:
        """Ensure lazy-loading uses the injected loader and caches its result."""

        calls: list[_DummyAnySplatModel] = []

        def fake_loader() -> _DummyAnySplatModel:
            model = _DummyAnySplatModel()
            calls.append(model)
            return model

        optimizer = ClusterAnySplat(model_loader=fake_loader)

        self.assertEqual(len(calls), 0)

        optimizer._ensure_model_loaded()
        self.assertEqual(len(calls), 1)

        optimizer._ensure_model_loaded()  # second call should be a no-op
        self.assertEqual(len(calls), 1)
        self.assertIs(optimizer._model, calls[0])

    def test_load_model_raises_import_error_when_dependency_missing(self) -> None:
        """Simulate missing native dependencies so the utility surfaces a clear ImportError."""

        with mock.patch.object(anysplat, "_IMPORT_ERROR", RuntimeError("torch_scatter missing")):
            with self.assertRaisesRegex(ImportError, r"anysplat.*could not be imported"):
                anysplat.load_model()


if __name__ == "__main__":
    unittest.main()
