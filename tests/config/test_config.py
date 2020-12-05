from utils.cfgnode import CfgNode, ArgsCfgNode, YACS
import unittest

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config'

class TestCfgNode(unittest.TestCase):
    """ Main test for Configuration management applet """

    def setup(self):
        super(TestCfgNode, self).setUp()

    def test_configuration_type(self):
        """ Test the types of input hyperparameters from yaml file """
        
        self.config = CfgNode()
        self.config.load_file(str(CONFIG_PATH) + '/config1.yaml')
        
        self.assertTrue(
            (isinstance(self.config.param.frontend,                         YACS))   and
            (isinstance(self.config.param.frontend.matching,                YACS))   and
            (isinstance(self.config.param.gtsfm.path,                       str))    and
            (isinstance(self.config.param.gtsfm.enable_gpu,                 bool))   and
            (isinstance(self.config.param.frontend.matching.num_features,   int))    and
            (isinstance(self.config.param.frontend.matching.tol,            float))  and
            (isinstance(self.config.param.gtsfm.image_path,                 list))   and
            (isinstance(self.config.param.frontend.matching.pyramid_scale,  list))
        )

    def test_configuration_value(self):
        """ Test the values of input hyperparameters from yaml file """
        
        self.config = CfgNode()
        self.config.load_file(str(CONFIG_PATH) + '/config1.yaml')
        
        self.assertTrue(
            (self.config.param.gtsfm.path == '~/gtsfm')                               and
            (self.config.param.gtsfm.enable_gpu == False)                             and
            (self.config.param.frontend.matching.num_features == 2000)                and
            (self.config.param.frontend.matching.tol == 1e-3)                         and
            (self.config.param.gtsfm.image_path == ['images1', 'images2', 'images3']) and
            (self.config.param.frontend.matching.pyramid_scale == [1, 2, 4])
        )

    def test_configuration_combination(self):
        """ Test the configuration files combination """
        
        self.config = CfgNode()

        self.config.load_file(str(CONFIG_PATH) + '/config1.yaml')
        self.config.load_file(str(CONFIG_PATH) + '/config2.yaml')

        self.assertTrue(
            (self.config.param.frontend.matching.num_features == 1000)          and
            (self.config.param.frontend.matching.deep_feature == 'superpoint')  and
            (self.config.param.backend.mode_triangulation == 'ransac')          and
            (self.config.param.visualization.enable_vis == True)
        )
    
if __name__ == "__main__":
    unittest.main()

