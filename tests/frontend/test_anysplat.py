"""Unit tests for AnySplat glue.

Authors: Frank Dellaert
"""

import unittest
from pathlib import Path
from unittest import mock

from gtsfm.cluster_optimizer.cluster_anysplat import ClusterAnySplat
from gtsfm.utils import anysplat
from gtsfm.utils import torch as torch_utils


class _DummyAnySplatModel:
    """Barebones stub used to mimic the AnySplat interface in unit tests."""

    def to(self, device):  # noqa: D401 - simple passthrough API
        self.device = device
        return self


class ClusterAnySplatTest(unittest.TestCase):
    """Exercise the minimal AnySplat cluster bootstrapping logic."""

    def setUp(self) -> None:
        from gtsfm.cluster_optimizer import cluster_anysplat

        cluster_anysplat._MODEL_CACHE.clear()

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

    @mock.patch("gtsfm.utils.anysplat.load_model")
    def test_default_loader_uses_remote_repo(self, mocked_loader) -> None:
        """Ensure default loader leaves checkpoint unset so the remote repo is used."""

        mocked_loader.return_value = _DummyAnySplatModel()
        optimizer = ClusterAnySplat()

        optimizer._ensure_model_loaded()

        mocked_loader.assert_called_once()
        kwargs = mocked_loader.call_args.kwargs
        self.assertEqual(kwargs["device"], torch_utils.default_device())
        self.assertNotIn("checkpoint_path", kwargs)

    @mock.patch("gtsfm.utils.anysplat.load_model")
    def test_local_checkpoint_forwarded(self, mocked_loader) -> None:
        """Explicit checkpoints should be loaded from disk."""

        mocked_loader.return_value = _DummyAnySplatModel()
        optimizer = ClusterAnySplat(local_checkpoint=Path("/tmp/fake/model.pt"))

        optimizer._ensure_model_loaded()

        mocked_loader.assert_called_once()
        kwargs = mocked_loader.call_args.kwargs
        self.assertEqual(kwargs["device"], torch_utils.default_device())
        self.assertEqual(kwargs["checkpoint_path"], Path("/tmp/fake/model.pt"))

    @mock.patch("gtsfm.utils.anysplat.load_model")
    def test_model_cache_reused_across_instances(self, mocked_loader) -> None:
        """Verify workers reuse the cached model instead of reloading per cluster."""

        cached_model = _DummyAnySplatModel()
        mocked_loader.return_value = cached_model

        optimizer_one = ClusterAnySplat()
        optimizer_two = ClusterAnySplat()

        optimizer_one._ensure_model_loaded()
        optimizer_two._ensure_model_loaded()

        mocked_loader.assert_called_once()
        kwargs = mocked_loader.call_args.kwargs
        self.assertEqual(kwargs["device"], torch_utils.default_device())
        self.assertNotIn("checkpoint_path", kwargs)
        self.assertIs(optimizer_two._model, cached_model)
        self.assertIs(optimizer_one._model, optimizer_two._model)


if __name__ == "__main__":
    unittest.main()
