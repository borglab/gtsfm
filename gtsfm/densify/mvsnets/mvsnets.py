"""MVSNets class inherited from MVSBase
    with PatchMatchNet integrated

Authors: Ren Liu
"""
from typing import Dict

import torch
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.densify.mvsnets.mvs_parser import Parser
from gtsfm.densify.mvsnets.mvs_utils import MVSNetsModelManager

logger = logger_utils.get_logger()


class MVSNets(MVSBase):
    """Base class for all MVSNets. It inherits MVSBase class. """

    def __init__(self) -> None:
        """Initialize the MVSNets module """
        super(MVSNets, self).__init__()

    def densify(
        self,
        images: Dict[int, Image],
        sfm_result: GtsfmData,
        view_number: int = 5,
        thres: list = [1.0, 0.01, 0.8],
        method: str = "PatchmatchNet",
        save_output: bool = False,
        output_path: str = "results_densify",
    ) -> np.ndarray:
        """Densify the GtsfmData using MVSNets method
        Args:
            images: image dictionary for each view,
            sfm_result: pre-computed GtsfmData,
            view_number: the number of views when setting up MVSNets, including 1 reference view and (view_number - 1) other views,
            thres: thresholds used in densify filters, the order is [geometric_pixel_threshold, geometric_depth_threshold, photography_threshold]
            method: used to specify one of the supported MVSNets, the default one is PatchmatchNet
            save_output: decide whether to save densify results, including depth maps, confidence maps, mesh file(.ply).
            output_path: if save_output is True, this string will be the relative output path

        Returns:
            Dense point cloud, as an array of shape (N,3)
        """

        num_images = sfm_result.number_images()

        image_values = []

        for img_id in range(num_images):
            image_values.append(images[img_id].value_array)

        logger.info("[Densify] begin to densify use : %s", method)

        logger.info("[Densify] step 1: parsing sfm_result to mvsnetsData")

        mvsnetsData = Parser.to_mvsnets_data(image_values, sfm_result)

        args = {
            "mvsnetsData": mvsnetsData,
            "img_wh": (image_values[0].shape[1], image_values[0].shape[0]),
            "n_views": view_number,
            "thres": thres,
            "gpu": torch.cuda.is_available(),
            "loadckpt": "gtsfm/densify/mvsnets/checkpoints/{}.ckpt".format(method.lower()),
            "save_output": save_output,
            "outdir": output_path,
        }

        logger.info("[Densify] step 2: going through %s ...", method)

        return MVSNetsModelManager.test(method, args)