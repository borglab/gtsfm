"""Class to store metrics computed in different GTSfM modules.

Authors: Akshay Krishnan
"""
from __future__ import annotations

import json
import numpy as np
from enum import Enum
from typing import Any, Dict, List, Union

import gtsfm.utils.io as io

DATA_KEY = "full_data"
SUMMARY_KEY = "summary"


class GtsfmMetric:
    """Class to store a metric computed in a GTSfM module."""

    class PlotType(Enum):
        BAR = 1  # For scalars
        BOX = 2  # For 1D distributions
        HISTOGRAM = 3  # For 1D distributions

    def _get_plot_types_for_dim(self, dim) -> List[PlotType]:
        if dim == 0:
            return [self.PlotType.BAR]
        if dim == 1:
            return [self.PlotType.BOX, self.PlotType.HISTOGRAM]
        return []

    def _get_distribution_histogram(self, data: np.ndarray) -> Dict[int, float]:
        if data.size == 0:
            print("Requested histogram for empty data metric, returning None.")
            return None
        if isinstance(data.tolist()[0], int):
            # One bin for each integer
            bins = int(np.max(data) - np.min(data) + 1)
            discrete = True
        else:
            bins = 10
            discrete = False
        count, bins = np.histogram(data, bins=bins)
        count = count.tolist()
        bins = bins.tolist()
        bins_lower = bins[:-1]
        bins_upper = bins[1:]

        histogram = {}
        for i in range(len(count)):
            if discrete:
                key = str(int(bins_lower[i]))
            else:
                key = "%.2f-%.2f" % (bins_lower[i], bins_upper[i])
            histogram[key] = count[i]
        return histogram

    def _create_summary(self, data: np.ndarray) -> Dict[str, Any]:
        if data.ndim != 1:
            raise ValueError('Metric must be a 1D distribution to get summary.')
        summary = {
            "min": np.min(data).tolist(),
            "max": np.max(data).tolist(),
            "median": np.median(data).tolist(),
            "mean": np.mean(data).tolist(),
            "stddev": np.std(data).tolist(),
        }
        if self._plot_type == self.PlotType.BOX:
            summary.update({"quartiles": self._get_distribution_quartiles(data)})
        elif self._plot_type == self.PlotType.HISTOGRAM:
            summary.update({"histogram": self._get_distribution_histogram(data)})
        return summary

    def _get_distribution_quartiles(self, data: np.ndarray) -> Dict[int, float]:
        query = list(range(0, 101, 25))
        quartiles = np.percentile(data, query)
        output = {}
        for i, q in enumerate(query):
            output['q'+str(i)] = quartiles[i].tolist()
        return output

    def __init__(
        self,
        name: str,
        data: Optional[Union[np.array, float, List[Union[int, float]]]] = None,
        summary: Optional[Dict[str, Any]] = None,
        store_full_data: bool = True,
        plot_type: PlotType = None,
    ):
        if summary is None and data is None:
            raise ValueError("Data and summary cannot both be None.")

        self._name = name
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if data.ndim > 1:
                raise ValueError("Metrics must be scalars on 1D-distributions.")
            self._dim = data.ndim 
            plot_types_for_dim = self._get_plot_types_for_dim(self._dim)
            if plot_type is None:
                self._plot_type = plot_types_for_dim[0]
            elif plot_type in plot_types_for_dim:
                self._plot_type = plot_type
            else:
                raise ValueError("Unsupported plot type for the data dimension")

            if self._dim == 1:
                self._summary = self._create_summary(data)
            if self._dim == 0 or store_full_data:
                self._data = data
            else:
                self._data = None
        else:
            self._dim = 1
            self._summary = summary
            self._plot_type = self.PlotType.HISTOGRAM if "histogram" in summary else self.PlotType.BOX
            self._data = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> np.array:
        return self._data

    @property
    def plot_type(self):
        return self._plot_type

    @property
    def dim(self):
        return self._dim

    @property
    def summary(self):
        return self._summary

    def get_metric_as_dict(self) -> Dict[str, Any]:
        if self._dim == 0:
            return {self._name: self._data.tolist()}

        metric_dict = {SUMMARY_KEY: self.summary}
        if self._data is not None:
            metric_dict[DATA_KEY] = self._data.tolist()
        return { self._name: metric_dict }

    def save_to_json(self, json_filename):
        io.save_json_file(json_filename, self.get_metric_as_dict())

    @classmethod
    def parse_from_dict(cls, metric_dict: Dict[str, Any]) -> GtsfmMetric:
        if len(metric_dict) != 1:
            raise AttributeError("Input metric dict should have a single key-value pair.")

        metric_name = list(metric_dict.keys())[0]
        metric_value = metric_dict[metric_name]

        # 1D distribution metrics
        if isinstance(metric_value, dict):
            if not DATA_KEY in metric_value:
                if not SUMMARY_KEY in metric_value:
                    raise ValueError("Metric {metric_name} does not have summary or data.")
                return cls(metric_name, summary=metric_value[SUMMARY_KEY])
            else:    
                return cls(metric_name, metric_value[DATA_KEY])

        # Scalar metrics
        return cls(metric_name, metric_value)


class GtsfmMetricsGroup:
    """Stores GtsfmMetrics from the same module. """

    def __init__(self, name: str, metrics: List[GtsfmMetric]):
        self._name = name
        self._metrics = metrics

    @property
    def name(self):
        return self._name

    @property
    def metrics(self):
        return self._metrics

    def add_metric(self, metric: GtsfmMetric):
        self._metrics.append(metric)

    def add_metrics(self, metrics: List[GtsfmMetric]):
        self._metrics.extend(metrics)

    def extend(self, metrics_group: GtsfmMetricsGroup):
        self._metrics.extend(metrics_group.metrics)

    def get_metrics_as_dict(self) -> Dict[str, Dict[str, Any]]:
        metrics_dict = {}
        for metric in self._metrics:
            metrics_dict.update(metric.get_metric_as_dict())
        return {self._name: metrics_dict}

    def save_to_json(self, path: str):
        io.save_json_file(path, self.get_metrics_as_dict())

    @classmethod
    def parse_from_dict(cls, metrics_group_dict) -> GtsfmMetricsGroup:
        if len(metrics_group_dict) != 1:
            raise AttributeError("Metrics group dict must have a single key-value pair.")
        metrics_group_name = list(metrics_group_dict.keys())[0]
        metrics_dict = metrics_group_dict[metrics_group_name]
        gtsfm_metrics_list = []
        for metric_name, metric_value in metrics_dict.items():
            gtsfm_metrics_list.append(GtsfmMetric.parse_from_dict({metric_name: metric_value}))
        return GtsfmMetricsGroup(metrics_group_name, gtsfm_metrics_list)

    @classmethod
    def parse_from_json(cls, json_filename):
        with open(json_filename) as f:
            metric_group_dict = json.load(f)
        return cls.parse_from_dict(metric_group_dict)
