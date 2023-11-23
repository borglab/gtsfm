"""
Unit tests for ProcessGraphGenerator.

Author: Kevin Fu
"""

import os
import tempfile
import unittest

from gtsfm.ui.process_graph_generator import ProcessGraphGenerator

# needed to test registry properly
from tests.ui.test_gtsfm_process import FakeImageLoader, FakeOutputGTSFM  # noqa: F401


class TestProcessGraphGenerator(unittest.TestCase):
    def test_save_graph(self):
        """Ensure ProcessGraphGenerator saves graph as png/svg files without crashing."""

        process_graph_generator = ProcessGraphGenerator(test_mode=True)

        with tempfile.TemporaryDirectory() as tempdir:
            png_path = os.path.join(tempdir, "output.png")
            svg_path = os.path.join(tempdir, "output.svg")
            extra_dir_path = os.path.join(tempdir, "output", "output.png")

            process_graph_generator.save_graph(png_path)
            process_graph_generator.save_graph(svg_path)
            process_graph_generator.save_graph(extra_dir_path)

            # check that files were saved, not their correctness
            self.assertTrue(os.path.exists(png_path))
            self.assertTrue(os.path.exists(svg_path))
            self.assertTrue(os.path.exists(extra_dir_path))

    def test_no_fake_nodes(self):
        """Ensure that no test nodes from test_registry.py are in the final
        graph when test mode is off.
        """

        process_graph_generator = ProcessGraphGenerator()
        process_graph_generator._build_graph()

        output_raw_dot = process_graph_generator._main_graph.to_string()

        self.assertTrue("GTSFMProcess" not in output_raw_dot)
        self.assertTrue("Fake" not in output_raw_dot)

    def test_build_graph(self):
        """Check that the test nodes are in the graph and connected correctly.
        Does not check for style elements.
        """

        process_graph_generator = ProcessGraphGenerator(test_mode=True)
        process_graph_generator._build_graph()

        output_raw_dot = process_graph_generator._main_graph.to_string()

        # can't assert the full graph directly because the REGISTRY will have other classes
        self.assertTrue("FakeImageLoader [" in output_raw_dot)
        self.assertTrue('"Internal Data" [' in output_raw_dot)
        self.assertTrue('FakeImageLoader -> "Internal Data"  [' in output_raw_dot)
        self.assertTrue('"Raw Images" -> FakeImageLoader  [' in output_raw_dot)
        self.assertTrue('FakeOutput -> "GTSFM Output"  [' in output_raw_dot)
        self.assertTrue('"Internal Data" -> FakeOutput  [' in output_raw_dot)
        self.assertTrue("subgraph cluster_ParentPlate" in output_raw_dot)
        self.assertTrue("label=ParentPlate" in output_raw_dot)
        self.assertTrue("FakeOutput [" in output_raw_dot)
        self.assertTrue('"GTSFM Output" [' in output_raw_dot)


if __name__ == "__main__":
    unittest.main()
