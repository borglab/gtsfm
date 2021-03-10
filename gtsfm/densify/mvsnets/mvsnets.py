import torch
from .mvsParser import Parser
from .mvsWriter import Writer
from .mvsLoader import Loader
from .mvsUtills import MVSNetsModelManager

class MVSNets:
    def __init__(self):
        pass 

    @classmethod    
    def densify(cls, sfmData, image_path, image_extension, view_number=5, thres=[1.0, 0.01, 0.8], method="mvsnets/PatchmatchNet"):

        method = method.strip().split("/")

        assert method[0].lower() == "mvsnets" 

        images = Loader.load_raw_images(image_path, image_extension)

        mvsnetsData = Parser.to_mvsnets_data(images, sfmData)

        prepared_input_path = Writer.write_mvsnets_data(mvsnetsData)

        del mvsnetsData

        args = {
            'testpath': prepared_input_path,
            'img_wh':   images[0].size,
            'outdir':   Writer.DENSIFY_RESULTS_PATH,
            'n_views':  view_number,
            'thres':    thres
        }
        
        print(args)

        MVSNetsModelManager.test(method[1], args)

        return NotImplemented
