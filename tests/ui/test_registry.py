"""
Unit tests for RegistryHolder and GTSFMProcess.

Author: Kevin Fu
"""

import unittest

from gtsfm.ui.registry import RegistryHolder, GTSFMProcess, UiMetadata
import abc


class FakeImageLoader(GTSFMProcess):
    """Test class."""

    def __init__(self, fake_image_dir):
        self._fake_image_dir = fake_image_dir

    def get_ui_metadata():
        return UiMetadata("FakeImageLoader", ["Raw Images"], ["Internal Data"], "")

    @property
    def fake_image_dir(self):
        return self._fake_image_dir


class FakeOutputBase(GTSFMProcess):
    """Test class."""

    def __init__(self):
        self._dummy_var = 0

    @abc.abstractmethod
    def base_method(self):
        ...

    def get_ui_metadata():
        return UiMetadata("FakeOutput", ["Internal Data"], ["GTSFM Output"], "Processor")


class FakeOutputGTSFM(FakeOutputBase):
    """Test class."""

    def __init__(self):
        # should have UI metadata from superclass FakeOutputBase
        super().__init__()

    def base_method(self):
        return True


class TestRegistryUtils(unittest.TestCase):
    """Test cases for registry and concrete registerable classes."""

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
            "GTSFMProcess": GTSFMProcess,
        }

        for cls_name, cls_type in expected_result.items():
            self.assertTrue(cls_name in registry)
            self.assertEqual(registry[cls_name], cls_type)

    def test_basic(self):
        """Test basic storing of UI metadata."""

        metadata = FakeImageLoader.get_ui_metadata()

        self.assertEqual(metadata.display_name, "FakeImageLoader")
        self.assertEqual(metadata.input_products, ["Raw Images"])
        self.assertEqual(metadata.output_products, ["Internal Data"])
        self.assertEqual(metadata.parent_plate, "")

    def test_abs_base(self):
        """Test that abstract base classes can have UI metadata."""

        # test that metadata can be accessed without init
        metadata = FakeOutputBase.get_ui_metadata()

        self.assertEqual(metadata.display_name, "FakeOutput")
        self.assertEqual(metadata.input_products, ["Internal Data"])
        self.assertEqual(metadata.output_products, ["GTSFM Output"])
        self.assertEqual(metadata.parent_plate, "Processor")

        # test that concrete objects work as intended
        concrete_obj = FakeOutputGTSFM()
        self.assertTrue(concrete_obj.base_method)
        self.assertEqual(concrete_obj._dummy_var, 0)


if __name__ == "__main__":
    unittest.main()
