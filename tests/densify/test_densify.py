""" Unit tests for the densify module (mvsnet)

Authors: Ren Liu
"""

import unittest
import os 
import glob
import cv2
import shutil
from pathlib import Path

import torch
import numpy as np
from gtsam import readBal
from hydra.experimental import compose, initialize_config_module

from gtsfm.loader.folder_loader import FolderLoader
from gtsfm.densify.mvsnets.mvsParser import Parser
from gtsfm.densify.mvsnets.mvsWriter import Writer
from gtsfm.densify.mvsnets.mvsLoader import Loader
from gtsfm.densify.mvsnets.mvsUtills import MVSNetsModelManager
from gtsfm.densify.dense_tools.depthmap_reader import DepthmapReaderManager
from gtsfm.densify.dense_tools.depthmap_metrics import Accuracy, Completeness

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
TEMP_ROOT_PATH = Path(__file__).resolve().parent / "temp"
TEMP_INPUT_PATH = Path(__file__).resolve().parent / "temp/inputs"
TEMP_OUTPUT_PATH = Path(__file__).resolve().parent / "temp/outputs"
DEFAULT_EXAMPLE_PATH = DATA_ROOT_PATH / "set1_1_lund_door"
EXAMPLE_BAL_PATH = DEFAULT_EXAMPLE_PATH / "sample_bal/ba_output.bal" 
CKPT_PATH = Path(__file__).resolve().parent.parent.parent / "gtsfm/densify/mvsnets/checkpoints/patchmatchnet.ckpt"

class TestDensifyMVSNets(unittest.TestCase):
    """Unit test for densifying the SfM results calculated by SceneOptimizer, using MVSNets::PatchmatchNet"""

    def setUp(self) -> None:
        self.loader = FolderLoader(str(DEFAULT_EXAMPLE_PATH), image_extension="JPG")
        assert len(self.loader)

    def test_densify(self):
        """Test will begin"""

        with initialize_config_module(config_module="gtsfm.configs"):

            sfmData = readBal(str(EXAMPLE_BAL_PATH))

            images = Loader.load_raw_images(DEFAULT_EXAMPLE_PATH, "JPG")
            mvsnetsData_fetch = Parser.to_mvsnets_data(images, sfmData)

            """Test Data Completeness"""
            self.assertTrue(len(mvsnetsData_fetch['cameras']) == len(mvsnetsData_fetch['images']))
            self.assertTrue(len(mvsnetsData_fetch['cameras']) == len(mvsnetsData_fetch['pairs']))
            
            """Test Depth Range"""
            self.assertTrue(len(mvsnetsData_fetch['depthRange']) == 3)
            self.assertTrue(np.abs(mvsnetsData_fetch['depthRange'][0] - 13) < 5)
            self.assertTrue(np.abs(mvsnetsData_fetch['depthRange'][1] - 21) < 5)

            """Test Camera Calibration"""
            labeled_cameras = Loader.load_labeled_cameras(DEFAULT_EXAMPLE_PATH)
            dist = np.subtract(mvsnetsData_fetch['cameras'][5][0][:,:-1], labeled_cameras[5][0][:,:-1]).flatten()
            dist = np.sqrt(np.mean(np.square(dist)))
            z = np.abs(labeled_cameras[5][0][:,:-1].flatten()).mean() 
            self.assertTrue(dist < 0.01 * z)

            """Test SfMData - MVSNet data conversion"""
            prepared_input_path = Writer.write_mvsnets_data(mvsnetsData_fetch, path=str(TEMP_INPUT_PATH))
            self.assertTrue(prepared_input_path == str(TEMP_INPUT_PATH))
            
            """Test MVSNet pipeline"""
            status = -1

            try:
                args = {
                    'dataset': "gtsfm_eval",
                    'testpath': prepared_input_path,
                    'img_wh':   images[0].size,
                    'outdir':   str(TEMP_OUTPUT_PATH),
                    'n_views':  5,
                    'thres':    [1.0, 0.01, 0.8],
                    'gpu':      torch.cuda.is_available(),
                    'loadckpt': str(CKPT_PATH)
                }

                MVSNetsModelManager.test('PatchmatchNet', args)

                status = 0
            except:
                pass 

            self.assertTrue(status == 0)
            
            """Test output depthmaps and mesh"""
            colmap_depthmap_pattern = str(DEFAULT_EXAMPLE_PATH / 'depth_img/depth_*.png')
            colmap_mask_pattern = str(DEFAULT_EXAMPLE_PATH / 'mask/*_final.png')
            colmap_reader = DepthmapReaderManager.build_depthmap_reader(colmap_depthmap_pattern, 'PNG')
            colmap_depthmap = colmap_reader.load()[0]
            colmap_mask = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) > 0 for mask in sorted(glob.glob(colmap_mask_pattern))][0]
            
            mvsnet_depthmap_pattern = str(TEMP_OUTPUT_PATH / 'scan1/depth_img/depth_*.png')
            mvsnet_mask_pattern = str(TEMP_OUTPUT_PATH / 'scan1/mask/*_final.png')
            mvsnet_reader = DepthmapReaderManager.build_depthmap_reader(mvsnet_depthmap_pattern, 'PNG')
            mvsnet_depthmap = mvsnet_reader.load()[5]
            mvsnet_mask = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) > 0 for mask in sorted(glob.glob(mvsnet_mask_pattern))][5]
            
            depthmap_sample = [mvsnet_depthmap, colmap_depthmap]
            mask_sample = [mvsnet_mask, colmap_mask]
            self.assertTrue(Accuracy()(depthmap_sample, mask_sample) < 2.0)
            self.assertTrue(Completeness()(depthmap_sample, mask_sample) > 1e-3)

            """Clean temp files"""
            status = -1
            try:
                shutil.rmtree(str(TEMP_ROOT_PATH))
                if not os.path.exists(str(TEMP_ROOT_PATH)):
                    status = 0
            except:
                pass 
            
            self.assertTrue(status == 0)

if __name__ == "__main__":
    unittest.main()
