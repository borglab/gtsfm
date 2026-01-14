"""Unit tests for cluster optimizer cacher.

Authors: Ayush Baid
"""
import unittest
from unittest.mock import MagicMock, patch

from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph
from gtsfm.cluster_optimizer.cluster_optimizer_cacher import ClusterOptimizerCacher
from gtsfm.common.gtsfm_data import GtsfmData


class TestClusterOptimizerCacher(unittest.TestCase):
    """Unit tests for ClusterOptimizerCacher."""

    @patch("gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.read_from_bz2_file", return_value=None)
    @patch("gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.write_to_bz2_file")
    def test_cache_miss(self, write_mock: MagicMock, read_mock: MagicMock) -> None:
        """Test the scenario of cache miss."""
        # Mock the underlying optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.pose_angular_error_thresh = 3.0
        mock_optimizer._output_worker = None
        
        mock_delayed = MagicMock()
        mock_computation = ClusterComputationGraph(
            io_tasks=tuple(), metric_tasks=tuple(), sfm_result=mock_delayed
        )
        mock_optimizer.create_computation_graph.return_value = mock_computation

        cacher = ClusterOptimizerCacher(optimizer=mock_optimizer)

        mock_context = MagicMock()
        mock_context.num_images = 10
        mock_context.cluster_path = (0, 1)
        mock_context.label = "cluster_0_1"
        mock_context.visibility_graph = [(0, 1), (1, 2)]
        mock_context.one_view_data_dict = {}

        result = cacher.create_computation_graph(mock_context)

        # Assert underlying optimizer was called
        mock_optimizer.create_computation_graph.assert_called_once_with(mock_context)

        # Assert computation graph was returned
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ClusterComputationGraph)

        # Assert read was called
        read_mock.assert_called_once()

    @patch(
        "gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.read_from_bz2_file",
        return_value=GtsfmData(number_images=5),
    )
    @patch("gtsfm.cluster_optimizer.cluster_optimizer_cacher.io_utils.write_to_bz2_file")
    def test_cache_hit(self, write_mock: MagicMock, read_mock: MagicMock) -> None:
        """Test the scenario of cache hit."""
        # Mock the underlying optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.pose_angular_error_thresh = 3.0
        mock_optimizer._output_worker = None

        cacher = ClusterOptimizerCacher(optimizer=mock_optimizer)

        mock_context = MagicMock()
        mock_context.num_images = 10
        mock_context.cluster_path = (0, 1)
        mock_context.label = "cluster_0_1"
        mock_context.visibility_graph = [(0, 1), (1, 2)]
        mock_context.one_view_data_dict = {}

        result = cacher.create_computation_graph(mock_context)

        # Assert underlying optimizer was NOT called (cache hit)
        mock_optimizer.create_computation_graph.assert_not_called()

        # Assert computation graph was returned
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ClusterComputationGraph)

        # Assert read was called
        read_mock.assert_called_once()

        # Assert write was not called (cache hit)
        write_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
