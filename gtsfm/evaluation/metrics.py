"""Classes to store metrics computed in different GTSfM modules.

The GtsfmMetric class stores a single metric, and the GtsfmMetricsGroup stores a list of metrics.
These classes are used to compute statistics on the metrics (min, max, etc),
save them to JSON, parse metrics from JSON, and plot them.

Authors: Akshay Krishnan
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

import gtsfm.utils.io as io
import gtsfm.utils.logger as logger_utils

# Keys to access data and summary in the dictionary representation of metrics.
FULL_DATA_KEY = "full_data"
SUMMARY_KEY = "summary"

# Type hint for a 1D distribution
Distribution1D = Union[np.ndarray, List[Optional[Union[int, float]]]]

logger = logger_utils.get_logger()


class GtsfmMetric:
    """Class to store a metric computed in a GTSfM module.

    A GtsfmMetric has a name and data which can either be a scalar or a 1D distribution.
    A metric can be represented as a dictionary for serialization.
    A scalar metric is represented as : {"metric_name": metric_value}
    A 1D distribution metric is represented as:
    {
        "metric_name": {
            "full_data": [list of values] (optional)
            "summary": {
                "min": min value
                "max": max value
                "median": median value
                "mean": mean value
                "stddev": standard deviation value
                "histogram": {} (OR) "quartiles": {}
            }
        }
    }
    For a 1D distribution, storing all the values of the metric is optional, as this can be large.
    The summary contains either a histogram or quartiles of the distribution depending on how it is to be plotted.
    When a 1D metric is first created, it is constructed using full_data, but if is parsed from a dict,
    it can also be constructed using just the summary.
    """

    class PlotType(Enum):
        """Used to select how the metric is to be plotted. Also decides the format of the summary.
        Example: Summaries of box plots store quartiles, and histogram plotted metrics store a histogram.
        """

        BAR = 1  # For scalars
        BOX = 2  # For 1D distributions
        HISTOGRAM = 3  # For 1D distributions

    def __init__(
        self,
        name: str,
        data: Optional[Union[float, Distribution1D]] = None,
        summary: Optional[Dict[str, Any]] = None,
        store_full_data: bool = False,
        plot_type: PlotType | None = None,
    ):
        """Creates a GtsfmMetric.

        Args:
            name: Name of the metric.
            data: All values of the metric, optional for 1D distributions, uses summary if not provided.
            summary: A summary dict of the metric, generated previously using the same class.
                Has to be provided if data = None.
            store_full_data: Whether all the values are to be stored or only summary is required.
            plot_type: The plot to use for visualization of the metric.
                Defaults:
                   PlotType.BAR if data is a scalar
                   PlotType.BOX if data is a distribution (other option is PlotType.HISTOGRAM)
                 It is inferred from the summary if plot_type is not provided and summary is.
        """
        if summary is None and data is None:
            raise ValueError("Data and summary cannot both be None.")

        self._name = name
        if data is not None:
            # Cast to a numpy array.
            if isinstance(data, list):
                # Replace None with NaN.
                data = [x if x is not None else np.nan for x in data]
                if all(isinstance(x, int) for x in data):
                    data = np.array(data, dtype=np.int32)
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            if data.ndim > 1:
                raise ValueError("Metrics must be scalars on 1D-distributions.")

            # Save dimension and plot_type for data
            self._dim = data.ndim
            plot_types_for_dim = self._get_plot_types_for_dim(self._dim)
            if plot_type is None:
                if summary is not None:
                    self._plot_type = self.PlotType.HISTOGRAM if "histogram" in summary else self.PlotType.BOX
                else:
                    self._plot_type = plot_types_for_dim[0]
            elif plot_type in plot_types_for_dim:
                self._plot_type = plot_type
            else:
                raise ValueError("Unsupported plot type for the data dimension")

            # Create a summary if the data is a 1D distribution.
            if self._dim == 1:
                self._summary = self._create_summary(data)

            # Store full data only if its a scalar or if asked to.
            if self._dim == 0 or store_full_data:
                self._data = data
            else:
                self._data = None
        else:
            # Metrics created from summary alone are 1D distribution metrics.
            self._dim = 1
            self._summary = summary
            self._plot_type = self.PlotType.HISTOGRAM if "histogram" in summary else self.PlotType.BOX
            self._data = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def plot_type(self) -> PlotType:
        return self._plot_type

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def summary(self) -> Dict[str, Any]:
        return self._summary

    def _get_plot_types_for_dim(self, dim: int) -> List[PlotType]:
        if dim == 0:
            return [self.PlotType.BAR]
        if dim == 1:
            return [self.PlotType.BOX, self.PlotType.HISTOGRAM]
        return []

    def _create_summary(self, data: np.ndarray) -> Dict[str, Any]:
        """Creates a summary of the given data.

        This is useful for analysis as data can be very large. The summary is a dict contains the following fields:
            - Min, max, median of data
            - Mean and std dev of data
            - Either quartiles or histogram of the data depending on plot_type of this metric.

        Args:
            data: 1D array of all values of the metric.

        Returns:
            summary as a dict that can be serialized to JSON for storage.
        """
        if data.ndim != 1:
            raise ValueError("Metric must be a 1D distribution to get summary.")
        if data.size == 0 or np.isnan(data).all():
            return {"min": np.nan, "max": np.nan, "median": np.nan, "mean": np.nan, "stddev": np.nan}
        summary = {
            "min": np.nanmin(data).tolist(),
            "max": np.nanmax(data).tolist(),
            "median": np.nanmedian(data).tolist(),
            "mean": np.nanmean(data).tolist(),
            "stddev": np.nanstd(data).tolist(),
            "len": int(data.size),
            "invalid": int(np.isnan(data).sum()),
        }
        if self._plot_type == self.PlotType.BOX:
            summary.update({"quartiles": get_quartiles_dict(data)})
        elif self._plot_type == self.PlotType.HISTOGRAM:
            summary.update({"histogram": get_histogram_dict(data)})
        return summary

    def get_metric_as_dict(self) -> Dict[str, Any]:
        """Provides a dictionary representation of the metric that can be serialized to JSON.

        The dict contains a single element, for which the key is the name of the metric.
        If metric is a distribution, the dict is in the below format:
        {
            metric_name: {
               FULL_DATA_KEY: [.. raw data if stored ..]
               SUMMARY_KEY: {
                    .. summary (stats) of distribution ..
               }
            }
        }
        If the metric is scalar, it is stored simply as {metric_name: value}.

        Returns:
            The metric as a dict representation explained above.
        """
        if self._dim == 0:
            return {self._name: self._data.tolist()}
        metric_dict = {SUMMARY_KEY: self.summary}
        if self._data is not None:
            metric_dict[FULL_DATA_KEY] = self._data.tolist()
        return {self._name: metric_dict}

    def save_to_json(self, json_filename: str) -> None:
        """Saves this metric's dict representation to a JSON file.

        Args:
            json_filename: Path to the json file.
        """
        io.save_json_file(json_filename, self.get_metric_as_dict())

    @classmethod
    def parse_from_dict(cls, metric_dict: Dict[str, Any]) -> GtsfmMetric:
        """Creates a GtsfmMetric by parsing a dict representation.

        It is assumed that the dict representation is the format created by GtsfmMetric.

        Args:
            metric_dict: Dict representation of the metric.

        Returns:
            Parsed GtsfmMetric instance.
        """
        if len(metric_dict) != 1:
            raise AttributeError("Input metric dict should have a single key-value pair.")

        metric_name = list(metric_dict.keys())[0]
        metric_value = metric_dict[metric_name]
        # 1D distribution metrics
        if isinstance(metric_value, dict):
            data = None
            summary = None
            if FULL_DATA_KEY in metric_value:
                data = metric_value[FULL_DATA_KEY]
            if SUMMARY_KEY in metric_value:
                summary = metric_value[SUMMARY_KEY]
            return cls(
                metric_name,
                data=data,
                summary=summary,
                store_full_data=data is not None,
            )
        # Scalar metrics
        return cls(metric_name, metric_value)


class GtsfmMetricsGroup:
    """Stores a list of `GtsfmMetric`s.

    A GtsfmMetricsGroup comprises a list of metrics that are semantically related, so that they can be
    given a name, saved and plotted together. This is the case when the metrics belong to the same Gtsfm module.

    A GtsfmMetricsGroup can be represented as a dictionary that can be serialized:
    {
        "metrics_group_name": {
            dictionary representation of metric1,
            dictionary representation of metric2,
            ...
        }
    }
    """

    def __init__(self, name: str, metrics: List[GtsfmMetric]):
        self._name = name
        self._metrics = metrics

    @property
    def name(self) -> str:
        return self._name

    @property
    def metrics(self) -> List[GtsfmMetric]:
        return self._metrics

    def add_metric(self, metric: GtsfmMetric) -> None:
        self._metrics.append(metric)

    def add_metrics(self, metrics: List[GtsfmMetric]) -> None:
        self._metrics.extend(metrics)

    def extend(self, metrics_group: GtsfmMetricsGroup) -> None:
        self._metrics.extend(metrics_group.metrics)

    def get_metrics_as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Creates the dictionary representation of the metrics group.

        This is the below format:
        {
            "metrics_group_name": {
                "metric1_name": metric1_dict
                "metric2_name": metric2_dict
                ...
            }
        }

        Returns:
            Metrics group dictionary representation.
        """
        metrics_dict = {}
        for metric in self._metrics:
            metrics_dict.update(metric.get_metric_as_dict())
        return {self._name: metrics_dict}

    def save_to_json(self, path: str) -> None:
        """Saves the dictionary representation of the metrics group to json.

        Args:
            path: Path to json file.
        """
        try:
            io.save_json_file(path, self.get_metrics_as_dict())
        except Exception as e:
            logger.error("Error saving metric %s to json %s", self._name, e)

    @classmethod
    def parse_from_dict(cls, metrics_group_dict: Dict[str, Any]) -> GtsfmMetricsGroup:
        """Creates a metric group from its dictionary representation.

        Args:
            metrics_group_dict: Dictionary representation generated by get_metrics_as_dict().

        Returns:
            A new GtsfmMetricsGroup parsed from the dict.
        """
        if len(metrics_group_dict) != 1:
            raise AttributeError("Metrics group dict must have a single key-value pair.")
        metrics_group_name = list(metrics_group_dict.keys())[0]
        metrics_dict = metrics_group_dict[metrics_group_name]
        gtsfm_metrics_list = []
        for metric_name, metric_value in metrics_dict.items():
            gtsfm_metrics_list.append(GtsfmMetric.parse_from_dict({metric_name: metric_value}))
        return GtsfmMetricsGroup(metrics_group_name, gtsfm_metrics_list)

    @classmethod
    def parse_from_json(cls, json_filename: str) -> GtsfmMetricsGroup:
        """Loads the JSON file that contains the metrics group represented as dict and parses it.

        Args:
            json_filename: Path to the JSON file.

        Returns:
            A new GtsfmMetricsGroup parsed from the JSON.
        """
        metric_group_dict = io.read_json_file(json_filename)
        return cls.parse_from_dict(metric_group_dict)


def get_histogram_dict(data: np.ndarray) -> Dict[str, Union[float, int]]:
    """Returns the histogram of data as a dictionary.

    If the data is float, the keys of the dictionary are interval buckets.
    If the data is int, the keys are also int.

    Args:
        data: 1D array of all values of the metric

    Returns:
        Histogram of data as a dict from bucket to count.
    """
    if data.size == 0:
        logger.info("Requested histogram for empty data metric, returning None.")
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


def get_quartiles_dict(data: np.ndarray) -> Dict[int, float]:
    """Returns quartiles for the provided data as a dict.

    Args:
        data: 1D distribution of metric values

    Returns:
        Quartiles of the data as a dict where keys are q0, q1, q2, q3, and q4
    """
    query = list(range(0, 101, 25))
    quartiles = np.percentile(data, query)
    output = {}
    for i, q in enumerate(query):
        output["q" + str(i)] = quartiles[i].tolist()
    return output
