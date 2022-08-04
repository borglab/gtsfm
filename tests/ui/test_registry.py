"""
Unit tests for RegistryHolder and BlueNode.

Author: Kevin Fu
"""

import unittest

from gtsfm.ui.registry import RegistryHolder, BlueNode
import abc


class FakeImageLoader(BlueNode):
    """Test class."""

    def __init__(self, fake_image_dir):
        super().__init__("FakeImageLoader", ["Raw Images"], ["Internal Data"], "")

        self._fake_image_dir = fake_image_dir

    @property
    def fake_image_dir(self):
        return self._fake_image_dir


class FakeOutputBase(BlueNode):
    """Test class."""

    def __init__(self):
        self._dummy_var = 0
        super().__init__("FakeOutputBase", ["Internal Data"], ["GTSFM Output"], "Processor")

    @abc.abstractmethod
    def base_method(self):
        ...


class FakeOutputGTSFM(FakeOutputBase):
    """Test class."""

    def __init__(self):
        super().__init__()

    def base_method(self):
        return True


class TestRegistryUtils(unittest.TestCase):
    def test_registry_holder(self):
        """
        Ensure registry holder has at least the test case classes above. (Since
        classes are added to the registry when defined, more classes will exist
        than just these test classes.)
        """

        registry = RegistryHolder.get_registry()

        expected_result = {
            "FakeImageLoader": FakeImageLoader,
            "FakeOutputGTSFM": FakeOutputGTSFM,
            "BlueNode": BlueNode,
        }

        for cls_name, cls_type in expected_result.items():
            self.assertTrue(cls_name in registry)
            self.assertEqual(registry[cls_name], cls_type)

    def test_blue_node(self):
        """Verify BlueNode fills in UI metadata properly."""

        fake_image_dir = "/dev/null"
        fake_image_loader = FakeImageLoader(fake_image_dir)

        self.assertEqual(fake_image_loader.fake_image_dir, fake_image_dir)
        self.assertEqual(fake_image_loader.display_name, "FakeImageLoader")
        self.assertEqual(fake_image_loader.input_gray_nodes, ["Raw Images"])
        self.assertEqual(fake_image_loader.output_gray_nodes, ["Internal Data"])
        self.assertEqual(fake_image_loader.parent_plate, "")

    def test_abs_base(self):
        """Test that putting BlueNode inside an abstract base class allows both BlueNode and the base class to work properly.
        """

        fake_output_gtsfm = FakeOutputGTSFM()

        self.assertTrue(fake_output_gtsfm.base_method)
        self.assertEqual(fake_output_gtsfm._dummy_var, 0)

        self.assertEqual(fake_output_gtsfm.display_name, "FakeOutputBase")
        self.assertEqual(fake_output_gtsfm.input_gray_nodes, ["Internal Data"])
        self.assertEqual(fake_output_gtsfm.output_gray_nodes, ["GTSFM Output"])
        self.assertEqual(fake_output_gtsfm.parent_plate, "Processor")


if __name__ == "__main__":
    unittest.main()
