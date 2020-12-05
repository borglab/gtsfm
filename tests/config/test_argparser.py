from utils.cfgnode import CfgNode, ArgsCfgNode, YACS
import unittest

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config'

class TestArgsCfgNode(unittest.TestCase):
    """ Main test for Argparser and Configuration Interaction """
    def setup(self):
        super(TestArgsCfgNode, self).setUp()

    def test_argparser_file(self):
        """ Test the configuration loading from argparser using yaml file"""
        
        config = CfgNode()

        parser = ArgsCfgNode('Test for config loading')

        config = parser.init_config(
            config, 
            config_file=[str(CONFIG_PATH)+'/config1.yaml', str(CONFIG_PATH)+'/config2.yaml']
        )
        
        print(config.param)
        self.assertTrue(
            (config.param.frontend.matching.num_features == 1000)          and
            (config.param.frontend.matching.deep_feature == 'superpoint')  and
            (config.param.backend.mode_triangulation == 'ransac')          and
            (config.param.visualization.enable_vis == True)
        )

    def test_argparser_param(self):
        """ Test the configuration loading from argparser with list of param"""
        config = CfgNode()

        parser = ArgsCfgNode('Test for config loading')

        config = parser.init_config(
            config, 
            config_file=[str(CONFIG_PATH)+'/config1.yaml', str(CONFIG_PATH)+'/config2.yaml']
        )
        config = parser.init_config(
            config,
            config_param=['backend.mode_triangulation', 'baseline', 'frontend.matching.num_features', '3000']
        )
        config = parser.init_config(config)
        
        self.assertTrue(
            (config.param.frontend.matching.num_features == 3000)          and
            (config.param.frontend.matching.deep_feature == 'superpoint')  and
            (config.param.backend.mode_triangulation == 'baseline')          and
            (config.param.visualization.enable_vis == True)
        )
    
if __name__ == "__main__":
    unittest.main()

