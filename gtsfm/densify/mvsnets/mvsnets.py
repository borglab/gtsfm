import torch
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.densify.mvsnets.mvsParser import Parser
from gtsfm.densify.mvsnets.mvsWriter import Writer
from gtsfm.densify.mvsnets.mvsLoader import Loader
from gtsfm.densify.mvsnets.mvsUtills import MVSNetsModelManager


class MVSNets(MVSBase):
    
    def __init__(self):
        super(MVSNets, self).__init__()
        pass
    
    #  ==== for gtsfm ====  
    def densify(self, images, sfm_result,
                view_number=5, 
                thres=[1.0, 0.01, 0.8], 
                method="mvsnets/PatchmatchNet", 
                use_gt_cam=False,
                save_output=False):


        num_images = sfm_result.number_images()
        
        image_values = []
        
        for img_id in range(num_images):
            image_values.append(images[img_id].value_array)

        print("Begin to densify use {}".format(method))

        method = method.strip().split("/")

        assert method[0].lower() == "mvsnets"

        print("[1/2]Parsing sfm_result to mvsnetsData...")

        mvsnetsData = Parser.to_mvsnets_data(image_values, sfm_result)
       
        args = {
            'mvsnetsData': mvsnetsData,
            'img_wh':   (image_values[0].shape[1], image_values[0].shape[0]),
            'outdir':   Writer.DENSIFY_RESULTS_PATH,
            'n_views':  view_number,
            'thres':    thres,
            'gpu':      torch.cuda.is_available(),
            'loadckpt': 'gtsfm/densify/mvsnets/checkpoints/{}.ckpt'.format(method[1].lower()),
            'save_output': save_output
        }

        print("[2/2] Going through {}...".format(method[1]))

        return MVSNetsModelManager.test(method[1], args)


    # #  ==== for test ====
    # @classmethod    
    # def densify_from_img_path(cls, 
    #             sfmData, 
    #             image_path, 
    #             image_extension, 
    #             view_number=5, 
    #             thres=[1.0, 0.01, 0.8], 
    #             method="mvsnets/PatchmatchNet", 
    #             use_gt_cam=False,
    #             save_output=False):

    #     print("Begin to densify use {}".format(method))

    #     method = method.strip().split("/")

    #     assert method[0].lower() == "mvsnets" 

    #     images = Loader.load_raw_images(image_path, image_extension)

    #     print("[1/2]Parsing sfmData to mvsnetsData...")

    #     if not use_gt_cam:
    #         # == if use measured camera matrices ==
    #         mvsnetsData = Parser.to_mvsnets_data(images, sfmData)
    #     else:
    #         # == if use accurate camera matrices ==
    #         labeled_cameras = Loader.load_labeled_cameras(image_path)
    #         mvsnetsData = Parser.to_mvsnets_data(images, sfmData, labeled_cameras)

    #     # prepared_input_path = Writer.write_mvsnets_data(mvsnetsData)

    #     args = {
    #         'mvsnetsData': mvsnetsData,
    #         'img_wh':   (images[0].shape[1], images[0].shape[0]),
    #         'outdir':   Writer.DENSIFY_RESULTS_PATH,
    #         'n_views':  view_number,
    #         'thres':    thres,
    #         'gpu':      torch.cuda.is_available(),
    #         'loadckpt': 'gtsfm/densify/mvsnets/checkpoints/{}.ckpt'.format(method[1].lower()),
    #         'save_output': save_output
    #     }

    #     print("[2/2] Going through {}...".format(method[1]))

    #     return MVSNetsModelManager.test(method[1], args)
    