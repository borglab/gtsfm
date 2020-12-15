from utils.cfgnode import CfgNode, ArgsCfgNode, YACS
from config.default import get_cfg_defaults
import unittest

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config'

class TestCfgNode(unittest.TestCase):
    """ Main test for Configuration management applet """

    def setup(self):
        super(TestCfgNode, self).setUp()

    def test_configuration_type(self):
        """ Test that correct types of input hyperparameters are read from 1 yaml file """
        
        self.config = CfgNode(
            get_cfg_defaults()
        )
        self.config.load_file(str(CONFIG_PATH) + '/config1.yaml')
        
        self.assertTrue(
            (isinstance(self.config.param.FeatureExtractor,                         YACS))   and
            (isinstance(self.config.param.FeatureExtractor.matching,                YACS))   and
            (isinstance(self.config.param.SceneOptimizer.path,                       str))    and
            (isinstance(self.config.param.SceneOptimizer.enable_gpu,                 bool))   and
            (isinstance(self.config.param.FeatureExtractor.matching.num_features,   int))    and
            (isinstance(self.config.param.FeatureExtractor.matching.tol,            float))  and
            (isinstance(self.config.param.SceneOptimizer.image_path,                 list))   and
            (isinstance(self.config.param.FeatureExtractor.matching.pyramid_scale,  list))
        )

    def test_configuration_value(self):
        """ Test whether values of input hyperparameters are correctly read from 1 yaml file """
        self.config = CfgNode(get_cfg_defaults())
        self.config.load_file(str(CONFIG_PATH) + '/config1.yaml')
        
        self.assertTrue(
            (self.config.param.SceneOptimizer.path == '~/gtsfm')                               and
            (self.config.param.SceneOptimizer.enable_gpu == False)                             and
            (self.config.param.FeatureExtractor.matching.num_features == 2000)                and
            (self.config.param.FeatureExtractor.matching.tol == 1e-3)                         and
            (self.config.param.SceneOptimizer.image_path == ['images1', 'images2', 'images3']) and
            (self.config.param.FeatureExtractor.matching.pyramid_scale == [1, 2, 4])
        )

    def test_configuration_combination(self):
        """ Test whether 2 configuration files can be correctly merged and loaded together """
        self.config = CfgNode(get_cfg_defaults())
        self.config.load_file(str(CONFIG_PATH) + '/config1.yaml')
        self.config.load_file(str(CONFIG_PATH) + '/config2.yaml')

        self.assertTrue(
            (self.config.param.FeatureExtractor.matching.num_features == 1000)          and
            (self.config.param.FeatureExtractor.matching.deep_feature == 'superpoint')  and
            (self.config.param.TwoViewEstimator.mode_triangulation == 'ransac')          and
            (self.config.param.MultiViewOptimizer.enable_vis == True)
        )
    
if __name__ == "__main__":
    unittest.main()

