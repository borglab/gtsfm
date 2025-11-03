"""Tests for loader CLI configuration helpers."""

import argparse

import pytest

from gtsfm.loader import configuration


def test_loader_default_none_does_not_override() -> None:
    """Default CLI args should leave loader untouched."""

    parser = argparse.ArgumentParser()
    configuration.add_loader_args(parser)

    args = parser.parse_args([])

    overrides = configuration.build_loader_overrides(args)
    assert all(not override.startswith("+loader@loader") for override in overrides)


@pytest.mark.parametrize("loader_name", ["olsson", "colmap"])
def test_loader_override_applied_when_specified(loader_name: str) -> None:
    """Supplying --loader should emit the expected override."""

    parser = argparse.ArgumentParser()
    configuration.add_loader_args(parser)

    args = parser.parse_args(["--loader", loader_name])

    overrides = configuration.build_loader_overrides(args)
    assert f"+loader@loader={loader_name}" in overrides
