from utils.cfgnode import CfgNode, ArgsCfgNode, YACS
from config.default import get_cfg_defaults
import unittest

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config'

class TestArgsCfgNode(unittest.TestCase):
    """ Main test for Argparser and Configuration Interaction """
    def setup(self):
        super(TestArgsCfgNode, self).setUp()

    def test_argparser_file(self):
        """ Test the configuration loading from argparser using yaml file"""
        
        config = CfgNode(
            get_cfg_defaults()
        )

        parser = ArgsCfgNode('Test for config loading')

        config = parser.init_config(
            config, 
            config_fpaths=[str(CONFIG_PATH)+'/config1.yaml', str(CONFIG_PATH)+'/config2.yaml']
        )
        
        print(config.param)
        self.assertTrue(
            (config.param.FeatureExtractor.matching.num_features == 1000)          and
            (config.param.FeatureExtractor.matching.deep_feature == 'superpoint')  and
            (config.param.TwoViewEstimator.mode_triangulation == 'ransac')          and
            (config.param.MultiViewOptimizer.enable_vis == True)
        )

    def test_argparser_param(self):
        """ Test functionality when we wish to merge command line input and config parameters.
        
        CLI input from argparse should override the parameters loaded from a config file config parameters
        """
        config = CfgNode(get_cfg_defaults())

        parser = ArgsCfgNode('Test for config loading')

        config = parser.init_config(
            config, 
            config_fpaths=[str(CONFIG_PATH)+'/config1.yaml', str(CONFIG_PATH)+'/config2.yaml']
        )
        config = parser.init_config(
            config,
            config_param=['TwoViewEstimator.mode_triangulation', 'baseline', 'FeatureExtractor.matching.num_features', '3000']
        )
        config = parser.init_config(config)
        
        self.assertTrue(
            (config.param.FeatureExtractor.matching.num_features == 3000)          and
            (config.param.FeatureExtractor.matching.deep_feature == 'superpoint')  and
            (config.param.TwoViewEstimator.mode_triangulation == 'baseline')          and
            (config.param.MultiViewOptimizer.enable_vis == True)
        )
    
if __name__ == "__main__":
    unittest.main()

