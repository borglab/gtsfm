"""Tests for GtsfmMetric and GtsfmMetricsGroup. 

Authors: Akshay Krishnan
"""
import copy
import os
import tempfile
import unittest

import numpy as np

from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup


class TestGtsfmMetric(unittest.TestCase):
    """Unit tests for GtsfmMetric class."""

    def setUp(self) -> None:
        super().setUp()
        # A metric with quartiles in its summary.
        self._metric_dict_quartiles = {
            "foo_metric": {
                "summary": {
                    "min": 0,
                    "max": 7,
                    "median": 3.5,
                    "mean": 3.5,
                    "stddev": 2.29,
                    "quartiles": {
                        "q0": 0,
                        "q1": 1.75,
                        "q2": 3.5,
                        "q3": 5.25,
                        "q4": 7,
                    },
                },
                "full_data": [0, 1, 2, 3, 4, 5, 6, 7],
            }
        }
        # A metric with histogram in its summary.
        self._metric_dict_histogram = {
            "bar_metric": {
                "summary": {
                    "min": 0,
                    "max": 3,
                    "median": 1.5,
                    "mean": 1.5,
                    "stddev": 1.12,
                    "histogram": {
                        "0": 2,
                        "1": 2,
                        "2": 2,
                        "3": 2,
                    },
                },
                "full_data": [0, 1, 0, 2, 2, 1, 3, 3],
            }
        }
        self._metric_dict_no_data = copy.deepcopy(self._metric_dict_histogram)
        # A metric with only summary and no full data.
        del self._metric_dict_no_data["bar_metric"]["full_data"]

    def test_create_scalar_metric(self) -> None:
        """Check that a scalar metric created has the right attributes."""
        metric = GtsfmMetric("a_scalar", 2)
        self.assertEqual(metric.name, "a_scalar")
        np.testing.assert_equal(metric.data, np.array([2]))
        self.assertEqual(metric.plot_type, GtsfmMetric.PlotType.BAR)

    def test_create_1d_distribution_metric(self) -> None:
        """Check that a 1D distribution metric created has the right attributes."""
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        metric = GtsfmMetric("dist_metric", data)
        self.assertEqual(metric.name, "dist_metric")
        np.testing.assert_equal(metric.data, data)
        self.assertEqual(metric.plot_type, GtsfmMetric.PlotType.BOX)

    def test_parses_from_dict_scalar(self) -> None:
        """Check that a scalar metric can be parsed from its dict representation."""
        scalar_metric_dict = {"foo_metric": 2}
        parsed_metric = GtsfmMetric.parse_from_dict(scalar_metric_dict)
        self.assertEqual(parsed_metric.name, "foo_metric")
        np.testing.assert_equal(parsed_metric.data, 2)

    def test_parses_from_dict_1D_distribution(self) -> None:
        """Check that a 1D distribution metric can be parsed from its dict representation."""
        parsed_metric = GtsfmMetric.parse_from_dict(self._metric_dict_quartiles)
        self.assertEqual(parsed_metric.name, "foo_metric")
        self.assertEqual(parsed_metric.plot_type, GtsfmMetric.PlotType.BOX)
        self.assertIn("quartiles", parsed_metric.summary)
        self.assertIn("full_data", parsed_metric.get_metric_as_dict()[parsed_metric.name])

    def test_parses_from_dict_1D_distribution_histogram(self) -> None:
        """Check that a 1D distribution metric with histogram summary can be parsed from dict."""
        parsed_metric = GtsfmMetric.parse_from_dict(self._metric_dict_histogram)
        self.assertEqual(parsed_metric.name, "bar_metric")
        self.assertEqual(parsed_metric.plot_type, GtsfmMetric.PlotType.HISTOGRAM)
        self.assertIn("histogram", parsed_metric.summary)
        self.assertIn("full_data", parsed_metric.get_metric_as_dict()[parsed_metric.name])

    def test_parses_from_dict_no_full_data(self) -> None:
        """Check that a 1D distribution metric can be parsed from dict without full data field."""
        parsed_metric = GtsfmMetric.parse_from_dict(self._metric_dict_no_data)
        self.assertEqual(parsed_metric.name, "bar_metric")
        self.assertEqual(parsed_metric.plot_type, GtsfmMetric.PlotType.HISTOGRAM)
        self.assertIn("histogram", parsed_metric.summary)
        self.assertNotIn("full_data", parsed_metric.get_metric_as_dict()[parsed_metric.name])

    def test_saves_to_json(self) -> None:
        """Check that no errors are raised when saving metric to json."""
        metric = GtsfmMetric("to_be_written_metric", np.arange(10.0))
        with tempfile.TemporaryDirectory() as tempdir:
            metric.save_to_json(os.path.join(tempdir, "test_metrics.json"))
            
    def test_list_input_with_none_entries(self) -> None:
        """Ensure that a GtsfmMetric can be successfully instantiated even with None entries."""
        per_rejected_track_avg_errors = [
            0.4943377406921827,
            0.07750727215807857,
            None,
            50.555624633887966,
            0.4105653459580154,
            None,
            None,
        ]
        metric = GtsfmMetric("rejected_track_avg_errors_px", per_rejected_track_avg_errors, store_full_data=False)


class TestGtsfmMetricsGroup(unittest.TestCase):
    """Unit tests for GtsfmMetricsGroup class."""

    def setUp(self) -> None:
        super().setUp()
        self._metrics_list = []
        self._metrics_list.append(GtsfmMetric(name="metric1", data=2))
        self._metrics_list.append(GtsfmMetric(name="metric2", data=np.array([1, 2, 3])))
        self._metrics_group = GtsfmMetricsGroup(name="test_metrics", metrics=self._metrics_list)

    def test_get_metric_as_dict(self) -> None:
        """Check that dictionary representation of the metrics groups is as expected."""
        metrics_group_dict = self._metrics_group.get_metrics_as_dict()
        self.assertEqual(len(metrics_group_dict), 1)
        self.assertEqual(len(metrics_group_dict["test_metrics"]), 2)
        metric1_dict = metrics_group_dict["test_metrics"]["metric1"]
        metric2_dict = metrics_group_dict["test_metrics"]["metric2"]
        np.testing.assert_equal(metric1_dict, np.array(2))
        np.testing.assert_equal(metric2_dict["full_data"], np.array([1, 2, 3]))

    def test_saves_to_json(self) -> None:
        """Check that no errors are raised when saving metrics group to json."""        
        with tempfile.TemporaryDirectory() as tempdir:
            self._metrics_group.save_to_json(os.path.join(tempdir, "test_metrics.json"))

    def test_parse_metrics_from_dict(self) -> None:
        """Check that metrics group can be parsed from json."""
        metrics_group_dict = self._metrics_group.get_metrics_as_dict()
        parsed_metrics = GtsfmMetricsGroup.parse_from_dict(metrics_group_dict)
        self.assertEqual(parsed_metrics.name, "test_metrics")
        self.assertEqual(len(parsed_metrics.metrics), 2)


if __name__ == "__main__":
    unittest.main()
