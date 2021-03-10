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

        Writer.writeOKLog("\nBegin to densify use {}".format(method))

        method = method.strip().split("/")

        assert method[0].lower() == "mvsnets" 

        images = Loader.load_raw_images(image_path, image_extension)

        Writer.writeOKLog("\n[1/4]Parsing sfmData to mvsnetsData...")

        # == if use measured camera matrices ==
        mvsnetsData = Parser.to_mvsnets_data(images, sfmData)

        # == if use accurate camera matrices ==
        # labeled_cameras = Loader.load_labeled_cameras(image_path)
        # mvsnetsData = Parser.to_mvsnets_data(images, sfmData, labeled_cameras)

        prepared_input_path = Writer.write_mvsnets_data(mvsnetsData)

        Writer.writeOKLog("\n[2/4]Writing parsed mvsnets data into {}...".format(prepared_input_path))

        del mvsnetsData

        args = {
            'dataset': "gtsfm_eval",
            'testpath': prepared_input_path,
            'img_wh':   images[0].size,
            'outdir':   Writer.DENSIFY_RESULTS_PATH,
            'n_views':  view_number,
            'thres':    thres,
            'gpu':      torch.cuda.is_available(),
            'loadckpt': 'gtsfm/densify/mvsnets/checkpoints/{}.ckpt'.format(method[1].lower())
        }

        Writer.writeOKLog("\n[3/4]Going through {}...".format(method[1]))
        print(args)

        MVSNetsModelManager.test(method[1], args)

        Writer.writeOKLog("\n[4/4]Densified results are written to {}...".format(args['outdir']))


        return True
