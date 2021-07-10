"""Tests for metrics container. 

Authors: Akshay Krishnan
"""
import numpy as np
import unittest

from gtsfm.evaluation.metric import GtsfmMetric, GtsfmMetricsGroup


class TestGtsfmMetric(unittest.TestCase):
    """Tests for the Metrics class."""

    def setUp():
        super.setUp()

    def test_create_scalar_metric(self):
        metric = GtsfmMetric('a_scalar', 2)
        self.assertEqual(metric.name, 'a_scalar')
        np.testing.assert_equal(metric.data, np.array([2]))
        self.assertEqual(metrc.plot_type, GtsfmMetric.PlotType.BAR)

    def test_create_1d_distribution_metric(self):
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        metric = GtsfmMetric('dist_metric', data)
        self.assertEqual(metric.name, 'dist_metric')
        np.testing.assert_equal(metric.data, data)
        self.assertEqual(metric.plot_type, GtsfmMetric.PlotType.BOX)

    def test_parses_from_dict(self):
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        metric = GtsfmMetric('dist_metric', data)
        metric_dict = metric.get_metric_as_dict()
        parsed_metric = GtsfmMetric.parse_from_dict(metric_dict)
        self.assertEqual(parsed_metric.name, metric.name)
        np.testing.assert_equal(parsed_metric.data, metric.data)


class TestGtsfmMetricsGroup(unittest.TestCase):

    def setUp():
        super.setUp()

    def test_create_from_metrics(self):
        metrics = []
        metrics.append(GtsfmMetric('metric1', 2))
        metrics.append(GtsfmMetric('metric2', np.array([1, 2, 3])))

        metrics_group = GtsfmMetricsGroup('test_metrics', metrics)
        metrics_group_dict = metrics_group.get_metrics_as_dict()
        self.assertEqual(len(metrics_group_dict), 1)
        self.assertEqual(len(metrics_group_dict['test_metrics']), 2)
        metric1_dict = metrics_group_dict['test_metrics']['metric1']
        metric2_dict = metrics_group_dict['test_metrics']['metric2']
        np.testing.assert_equal(metric1_dict, np.array([2]))
        np.testing.assert_equal(metric2_dict['full_data'], np.array([2]))
