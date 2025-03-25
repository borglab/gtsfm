"""Unit tests for graph partitioning functionality.

Authors: Zongyue Liu
"""

import unittest
from typing import List, Tuple

import numpy as np

from gtsfm.graph_partitioner.single_partition import SinglePartition


class TestGraphPartitioning(unittest.TestCase):
    """Tests for graph partitioning functionality."""

    def test_single_partition(self):
        """Test that SinglePartition correctly returns all pairs as one partition."""
        # Create some dummy image pairs
        image_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
        
        # Create a SinglePartition instance
        partitioner = SinglePartition()
        
        # Get partitioned result
        partitioned_pairs = partitioner.partition_image_pairs(image_pairs)
        
        # Check that we get exactly one partition
        self.assertEqual(len(partitioned_pairs), 1)
        
        # Check that the partition contains all the original pairs
        self.assertEqual(set(partitioned_pairs[0]), set(image_pairs))

    def test_single_partition_with_threshold(self):
        """Test that SinglePartition respects the threshold parameter."""
        # Create a SinglePartition with a threshold
        partitioner = SinglePartition(threshold=0.5)
        
        # Create some dummy image pairs
        image_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
        
        # Get partitioned result
        partitioned_pairs = partitioner.partition_image_pairs(image_pairs)
        
        # For SinglePartition, the threshold shouldn't affect the result - all pairs should be included
        self.assertEqual(len(partitioned_pairs), 1)
        self.assertEqual(set(partitioned_pairs[0]), set(image_pairs))
        
        # Verify that it follows the interface defined in the base class
        self.assertTrue(hasattr(partitioner, 'partition_image_pairs'))
        self.assertEqual(partitioner.process_name, "SinglePartition")


if __name__ == "__main__":
    unittest.main()
