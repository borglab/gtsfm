"""
Unit tests for RegistryHolder and GTSFMProcess.

Author: Kevin Fu
"""

import unittest

from gtsfm.ui.registry import RegistryHolder, GTSFMProcess
import abc


class FakeImageLoader(GTSFMProcess):
    """Test class."""

    def __init__(self, fake_image_dir=None):
        super().__init__("FakeImageLoader", ["Raw Images"], ["Internal Data"], "")

        if fake_image_dir is not None:
            self._fake_image_dir = fake_image_dir

    @property
    def fake_image_dir(self):
        return self._fake_image_dir


class FakeOutputBase(GTSFMProcess):
    """Test class."""

    def __init__(self):
        super().__init__("FakeOutput", ["Internal Data"], ["GTSFM Output"], "Processor")

        self._dummy_var = 0

    @abc.abstractmethod
    def base_method(self):
        ...


class FakeOutputGTSFM(FakeOutputBase):
    """Test class."""

    def __init__(self):
        # should have UI metadata from superclass FakeOutputBase
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
            "GTSFMProcess": GTSFMProcess,
        }

        for cls_name, cls_type in expected_result.items():
            self.assertTrue(cls_name in registry)
            self.assertEqual(registry[cls_name], cls_type)

    def test_basic(self):
        """Verify GTSFMProcess fills in UI metadata properly."""

        fake_image_dir = "/dev/null"
        fake_image_loader = FakeImageLoader(fake_image_dir)

        self.assertEqual(fake_image_loader.fake_image_dir, fake_image_dir)
        self.assertEqual(fake_image_loader.display_name, "FakeImageLoader")
        self.assertEqual(fake_image_loader.input_products, ["Raw Images"])
        self.assertEqual(fake_image_loader.output_products, ["Internal Data"])
        self.assertEqual(fake_image_loader.parent_plate, "")

    def test_abs_base(self):
        """Test that putting GTSFMProcess inside an abstract base class allows both GTSFMProcess and the base class to work properly."""

        fake_output_gtsfm = FakeOutputGTSFM()

        self.assertTrue(fake_output_gtsfm.base_method)
        self.assertEqual(fake_output_gtsfm._dummy_var, 0)

        self.assertEqual(fake_output_gtsfm.display_name, "FakeOutput")
        self.assertEqual(fake_output_gtsfm.input_products, ["Internal Data"])
        self.assertEqual(fake_output_gtsfm.output_products, ["GTSFM Output"])
        self.assertEqual(fake_output_gtsfm.parent_plate, "Processor")


if __name__ == "__main__":
    unittest.main()
