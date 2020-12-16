
import unittest
from pathlib import Path

from utils.cfgnode import GtsfmCfgNode, GtsfmArgsCfgNode, YACS
from config.defaults import get_cfg_defaults

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config"


class TestCfgNode(unittest.TestCase):
    """ Main test for Configuration management applet """

    def setup(self):
        super(TestCfgNode, self).setUp()

    def test_configuration_type(self):
        """ Test that correct types of input hyperparameters are read from 1 yaml file """

        self.config = GtsfmCfgNode(get_cfg_defaults())
        self.config.load_file(str(CONFIG_PATH) + "/config1.yaml")

        self.assertTrue(isinstance(self.config.param.FeatureExtractor, YACS))
        self.assertTrue(isinstance(self.config.param.FeatureExtractor.matching, YACS))
        self.assertTrue(isinstance(self.config.param.SceneOptimizer.path, str))
        self.assertTrue(isinstance(self.config.param.SceneOptimizer.enable_gpu, bool))
        self.assertTrue(isinstance(self.config.param.FeatureExtractor.matching.num_features, int))
        self.assertTrue(isinstance(self.config.param.FeatureExtractor.matching.tol, float))
        self.assertTrue(isinstance(self.config.param.SceneOptimizer.image_path, list))
        self.assertTrue(isinstance(self.config.param.FeatureExtractor.matching.pyramid_scale, list))

    def test_configuration_value(self):
        """ Test whether values of input hyperparameters are correctly read from 1 yaml file """
        self.config = GtsfmCfgNode(get_cfg_defaults())
        self.config.load_file(str(CONFIG_PATH) + "/config1.yaml")

        self.assertTrue(self.config.param.SceneOptimizer.path == "~/gtsfm")
        self.assertTrue(self.config.param.SceneOptimizer.enable_gpu == False)
        self.assertTrue(self.config.param.FeatureExtractor.matching.num_features == 2000)
        self.assertTrue(self.config.param.FeatureExtractor.matching.tol == 1e-3)
        self.assertTrue(self.config.param.SceneOptimizer.image_path == ["images1", "images2", "images3"])
        self.assertTrue(self.config.param.FeatureExtractor.matching.pyramid_scale == [1, 2, 4])

    def test_configuration_combination(self):
        """ Test whether 2 configuration files can be correctly merged and loaded together """
        self.config = GtsfmCfgNode(get_cfg_defaults())
        self.config.load_file(str(CONFIG_PATH) + "/config1.yaml")
        self.config.load_file(str(CONFIG_PATH) + "/config2.yaml")

        self.assertTrue(self.config.param.FeatureExtractor.matching.num_features == 1000)
        self.assertTrue(self.config.param.FeatureExtractor.matching.deep_feature == "superpoint")
        self.assertTrue(self.config.param.TwoViewEstimator.mode_triangulation == "ransac")
        self.assertTrue(self.config.param.MultiViewOptimizer.enable_vis == True)


if __name__ == "__main__":
    unittest.main()
