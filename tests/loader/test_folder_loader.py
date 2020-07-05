"""Unit test for the Folder Loader class.

Authors:Ayush Baid
"""
import unittest

import dask
import numpy as np

import utils.io as io_utils
from loader.folder_loader import FolderLoader


class TestFolderLoader(unittest.TestCase):
    """
    Unit tests for folder loader, which loads image from a folder on disk.
    """

    def setUp(self):
        """
        Set up the loader for the test.
        """
        super().setUp()

        # set up ground truth data for comparison

        self.loader = FolderLoader('tests/data/lund', image_extension='jpg')

    def test_len(self):
        """
        Test the number of entries in the loader.
        """

        self.assertEqual(29, len(self.loader))

    def test_get_image_validindex(self):
        """
        Tests that get_image works for all valid indices.
        """

        for idx in range(len(self.loader)):
            self.assertIsNotNone(self.loader.get_image(idx))

    def test_get_image_invalidindex(self):
        """
        Test that get_image raises an exception on an invalid index.
        """

        # negative index
        with self.assertRaises(IndexError):
            self.loader.get_image(-1)
        # len() as index
        with self.assertRaises(IndexError):
            self.loader.get_image(29)
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_image(35)

    def test_image_contents(self):
        """
        Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """

        index_to_test = 5
        file_path = 'tests/data/lund/images/06.jpg'

        loader_image = self.loader.get_image(index_to_test)

        expected_image = io_utils.load_image(file_path)

        np.testing.assert_allclose(
            expected_image.image_array, loader_image.image_array)

    def test_delayed_get_image(self):
        """
        Checks that the delayed get API functions exactly as the normal get API
        """

        index_to_test = 5

        delayed_result = self.loader.delayed_get_image(index_to_test).compute()

        normal_result = self.loader.get_image(index_to_test)

        np.testing.assert_allclose(
            normal_result.image_array, delayed_result.image_array)

    def test_create_computation_graph(self):
        """
        Tests the graph for loading all the images
        """

        loader_graph = self.loader.create_computation_graph()

        # check the length of the graph
        self.assertEqual(29, len(loader_graph))

        results = dask.compute(loader_graph)[0]

        # randomly check image loads from a few indices
        np.testing.assert_allclose(
            results[5].image_array, self.loader.get_image(5).image_array
        )

        np.testing.assert_allclose(
            results[12].image_array, self.loader.get_image(12).image_array)
