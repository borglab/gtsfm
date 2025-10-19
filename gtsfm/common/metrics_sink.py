"""Abstractions for persisting pipeline metrics."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Iterable, List

import gtsfm.utils.logger as logger_utils
from gtsfm.evaluation import metrics_report
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

logger = logger_utils.get_logger()

_DEFAULT_METRIC_ORDER: Dict[str, int] = {
    "retriever_metrics": 10,
    "frontend_runtime_metrics": 20,
    "verifier_summary_VIEWGRAPH_2VIEW_REPORT": 30,
    "view_graph_estimation_metrics": 40,
    "rotation_cycle_consistency_metrics": 45,
    "rotation_averaging_metrics": 50,
    "translation_averaging_metrics": 60,
    "data_association_metrics": 70,
    "bundle_adjustment_metrics": 80,
    "total_summary_metrics": 90,
}


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
        metrics_group.save_to_json(str(self._directory / f"{metrics_group.name}.json"))
        self._write_html_report()

    def _load_recorded_groups(self) -> List[GtsfmMetricsGroup]:
        """Load all persisted metric groups from disk in pipeline order."""
        groups_by_name: Dict[str, GtsfmMetricsGroup] = {}
        for json_path in self._directory.glob("*.json"):
            try:
                group = GtsfmMetricsGroup.parse_from_json(str(json_path))
            except Exception:  # pragma: no cover - defensive guard against non-metric JSON
                continue
            groups_by_name[group.name] = group

        if not groups_by_name:
            return []

        def sort_key(name: str) -> tuple[int, str]:
            return (_DEFAULT_METRIC_ORDER.get(name, 1000), name)

        ordered_names = sorted(groups_by_name.keys(), key=sort_key)
        return [groups_by_name[name] for name in ordered_names]

    def _write_html_report(self) -> None:
        """Regenerate the consolidated HTML report after each write."""
        try:
            metric_groups = self._load_recorded_groups()
            if not metric_groups:
                return
            metrics_report.generate_metrics_report_html(
                metric_groups, str(self._directory / "gtsfm_metrics_report.html"), None
            )
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("Failed to refresh metrics report HTML in %s: %s", self._directory, exc)
