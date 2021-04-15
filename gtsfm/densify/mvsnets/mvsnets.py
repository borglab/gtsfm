"""MVSNets class inherited from MVSBase
    with PatchMatchNet integrated

Authors: Ren Liu
"""

import torch
import gtsfm.utils.logger as logger_utils
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.densify.mvsnets.mvsParser import Parser
from gtsfm.densify.mvsnets.mvsUtills import MVSNetsModelManager
logger = logger_utils.get_logger()

class MVSNets(MVSBase):
    
    def __init__(self):
        super(MVSNets, self).__init__()
    
    #  ==== for gtsfm ====  

    def densify(self, images, sfm_result,
                view_number=5, 
                thres=[1.0, 0.01, 0.8], 
                method="mvsnets/PatchmatchNet", 
                use_gt_cam=False,
                save_output=False,
                output_path='results_densify'):

        num_images = sfm_result.number_images()
        
        image_values = []
        
        for img_id in range(num_images):
            image_values.append(images[img_id].value_array)

        logger.info("[Densify] begin to densify use : %s", method)

        method = method.strip().split("/")

        assert method[0].lower() == "mvsnets"

        logger.info("[Densify] step 1: parsing sfm_result to mvsnetsData")

        mvsnetsData = Parser.to_mvsnets_data(image_values, sfm_result)
       
        args = {
            'mvsnetsData': mvsnetsData,
            'img_wh':   (image_values[0].shape[1], image_values[0].shape[0]),
            'n_views':  view_number,
            'thres':    thres,
            'gpu':      torch.cuda.is_available(),
            'loadckpt': 'gtsfm/densify/mvsnets/checkpoints/{}.ckpt'.format(method[1].lower()),
            'save_output': save_output,
            'outdir':   output_path
        }

        logger.info("[Densify] step 2: going through %s ...", method[1])

        return MVSNetsModelManager.test(method[1], args)