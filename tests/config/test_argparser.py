import unittest
from pathlib import Path

from config.defaults import get_cfg_defaults
from gtsfm.utils.gtsfm_cfgnode import GtsfmCfgNode, GtsfmArgsCfgNode

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config"


class TestArgsCfgNode(unittest.TestCase):
    """ Main test for Argparser and Configuration Interaction """

    def setup(self):
        super(TestArgsCfgNode, self).setUp()

    def test_argparser_file(self) -> None:
        """ Test the configuration loading from argparser using yaml file"""

        config = GtsfmCfgNode(get_cfg_defaults())

        parser = GtsfmArgsCfgNode("Test for config loading")

        config = parser.init_config(
            config,
            config_fpaths=[
                str(CONFIG_PATH) + "/config1.yaml",
                str(CONFIG_PATH) + "/config2.yaml",
            ],
        )

        print(config.param)
        self.assertTrue(config.param.FeatureExtractor.matching.num_features == 1000)
        self.assertTrue(config.param.FeatureExtractor.matching.deep_feature == "superpoint")
        self.assertTrue(config.param.TwoViewEstimator.mode_triangulation == "ransac")
        self.assertTrue(config.param.MultiViewOptimizer.enable_vis == True)

    def test_argparser_param(self) -> None:
        """Test functionality when we wish to merge command line input and config parameters.

        CLI input from argparse should override the parameters loaded from a config file config parameters
        """
        config = GtsfmCfgNode(get_cfg_defaults())

        parser = GtsfmArgsCfgNode("Test for config loading")

        config = parser.init_config(
            config,
            config_fpaths=[
                str(CONFIG_PATH) + "/config1.yaml",
                str(CONFIG_PATH) + "/config2.yaml",
            ],
        )
        config = parser.init_config(
            config,
            config_param=[
                "TwoViewEstimator.mode_triangulation",
                "baseline",
                "FeatureExtractor.matching.num_features",
                "3000",
            ],
        )
        config = parser.init_config(config)

        self.assertTrue(config.param.FeatureExtractor.matching.num_features == 3000)
        self.assertTrue(config.param.FeatureExtractor.matching.deep_feature == "superpoint")
        self.assertTrue(config.param.TwoViewEstimator.mode_triangulation == "baseline")
        self.assertTrue(config.param.MultiViewOptimizer.enable_vis == True)


if __name__ == "__main__":
    unittest.main()
