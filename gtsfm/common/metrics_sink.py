"""Abstractions for persisting pipeline metrics."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterable

from gtsfm.evaluation.metrics import GtsfmMetricsGroup


class MetricsSink(abc.ABC):
    """Abstract destination for `GtsfmMetricsGroup` objects."""

    @abc.abstractmethod
    def record(self, metrics_group: GtsfmMetricsGroup) -> None:
        """Persist a single metrics group."""

    def record_many(self, metrics_groups: Iterable[GtsfmMetricsGroup]) -> None:
        """Persist multiple metric groups."""
        for group in metrics_groups:
            self.record(group)


class NullMetricsSink(MetricsSink):
    """No-op metrics sink to disable metrics persistence."""

    def record(self, metrics_group: GtsfmMetricsGroup) -> None:  # pragma: no cover - trivial
        return


class FileMetricsSink(MetricsSink):
    """Persist metrics as JSON files within a directory."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    def record(self, metrics_group: GtsfmMetricsGroup) -> None:
        metrics_group.save_to_json(self._directory / f"{metrics_group.name}.json")
