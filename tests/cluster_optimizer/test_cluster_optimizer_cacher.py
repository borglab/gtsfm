"""Unit tests for cluster optimizer cacher.

Authors: GitHub Copilot
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph, ClusterContext
from gtsfm.cluster_optimizer.cluster_optimizer_cacher import ClusterOptimizerCacher
from gtsfm.common.gtsfm_data import GtsfmData

# Dummy GtsfmData for testing
DUMMY_GTSFM_DATA = GtsfmData(number_images=5)

ROOT_PATH = Path(__file__).resolve().parent.parent.parent


class TestClusterOptimizerCacher(unittest.TestCase):
    """Unit tests for ClusterOptimizerCacher."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock optimizer
        self.mock_optimizer = MagicMock()
        self.mock_optimizer.pose_angular_error_thresh = 3.0
        self.mock_optimizer._output_worker = None
        self.mock_optimizer.__class__.__name__ = "MockOptimizer"
        self.mock_optimizer.__repr__ = MagicMock(return_value="MockOptimizer()")

        # Create a mock cluster context
        self.mock_context = MagicMock(spec=ClusterContext)
        self.mock_context.num_images = 10
        self.mock_context.cluster_path = (0, 1)
        self.mock_context.label = "cluster_0_1"
        self.mock_context.visibility_graph = [(0, 1), (1, 2)]
        self.mock_context.one_view_data_dict = {}

    @patch("gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.read_from_bz2_file", return_value=None)
    @patch("gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.write_to_bz2_file")
    def test_cache_miss(self, write_mock: MagicMock, read_mock: MagicMock) -> None:
        """Test the scenario of cache miss."""
        # Set up the mock optimizer to return a computation graph
        mock_delayed = MagicMock()
        mock_computation = ClusterComputationGraph(
            io_tasks=tuple(), metric_tasks=tuple(), sfm_result=mock_delayed
        )
        self.mock_optimizer.create_computation_graph.return_value = mock_computation

        # Create the cacher
        obj_under_test = ClusterOptimizerCacher(optimizer=self.mock_optimizer)

        # Call create_computation_graph
        result = obj_under_test.create_computation_graph(self.mock_context)

        # Assert that the underlying optimizer was called
        self.mock_optimizer.create_computation_graph.assert_called_once_with(self.mock_context)

        # Assert that a computation graph was returned
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ClusterComputationGraph)

        # Assert that read function was called (cache miss)
        self.assertEqual(read_mock.call_count, 1)

    @patch(
        "gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.read_from_bz2_file",
        return_value=DUMMY_GTSFM_DATA,
    )
    @patch("gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.write_to_bz2_file")
    def test_cache_hit(self, write_mock: MagicMock, read_mock: MagicMock) -> None:
        """Test the scenario of cache hit."""
        # Create the cacher
        obj_under_test = ClusterOptimizerCacher(optimizer=self.mock_optimizer)

        # Call create_computation_graph
        result = obj_under_test.create_computation_graph(self.mock_context)

        # Assert that the underlying optimizer was NOT called (cache hit)
        self.mock_optimizer.create_computation_graph.assert_not_called()

        # Assert that a computation graph was returned
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ClusterComputationGraph)

        # Assert that read function was called (cache hit)
        self.assertEqual(read_mock.call_count, 1)

        # Assert that write function was not called (cache hit)
        write_mock.assert_not_called()

    def test_repr_delegation(self) -> None:
        """Test that __repr__ delegates to the wrapped optimizer."""
        obj_under_test = ClusterOptimizerCacher(optimizer=self.mock_optimizer)
        result = repr(obj_under_test)
        self.assertEqual(result, "MockOptimizer()")

    def test_attribute_delegation(self) -> None:
        """Test that public attributes are delegated to the wrapped optimizer."""
        self.mock_optimizer.some_public_attribute = "test_value"
        obj_under_test = ClusterOptimizerCacher(optimizer=self.mock_optimizer)
        
        # Test that public attributes are delegated
        self.assertEqual(obj_under_test.some_public_attribute, "test_value")

    def test_private_attribute_not_delegated(self) -> None:
        """Test that private attributes are not delegated."""
        obj_under_test = ClusterOptimizerCacher(optimizer=self.mock_optimizer)
        
        # Test that accessing private attributes raises AttributeError
        with self.assertRaises(AttributeError):
            _ = obj_under_test._some_private_attr

    def test_cache_key_generation(self) -> None:
        """Test that cache keys are generated consistently."""
        obj_under_test = ClusterOptimizerCacher(optimizer=self.mock_optimizer)
        
        # Generate cache key twice with the same context
        key1 = obj_under_test._generate_cache_key(self.mock_context)
        key2 = obj_under_test._generate_cache_key(self.mock_context)
        
        # Assert that the keys are the same
        self.assertEqual(key1, key2)
        
        # Assert that the key is a string
        self.assertIsInstance(key1, str)


if __name__ == "__main__":
    unittest.main()
