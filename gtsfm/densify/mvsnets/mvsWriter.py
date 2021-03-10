from PIL import Image
import os 
import numpy as np 

class Writer(object):
    
    RESULTS_PATH = './results'
    RESULTS_METRICS_PATH = './results_metrics'
    PLOTS_PATH = './plots'
    LOGS_PATH = './logs'
    DENSIFY_MVSNETS_DATA_PATH = './results_densify/inputs'
    DENSIFY_RESULTS_PATH = './results_densify/outputs'

    OKGREEN = '\033[92m' 
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def write_mvsnets_data(cls, mvsnetsData, path=None):
        if not path:
            path = cls.DENSIFY_MVSNETS_DATA_PATH

        num_samples = len(mvsnetsData['images'])
        
        images = mvsnetsData['images']
        cameras = mvsnetsData['cameras']
        pairs = mvsnetsData['pairs']
        depth_range = mvsnetsData['depthRange']
        
        imgs_root = os.path.join(path, 'scan1', 'images')
        cams_root = os.path.join(path, 'scan1', 'cams_1')
        
        if not os.path.exists(imgs_root):
            os.makedirs(imgs_root)

        if not os.path.exists(cams_root):
            os.makedirs(cams_root)
        
        pair_lines = ["{}\n".format(num_samples)]

        for i in range(num_samples):
        
            # write images

            img = images[i]
            img.save(os.path.join(imgs_root,'{:08d}.jpg'.format(i)))

            # write cameras

            camfile = open(os.path.join(cams_root,'{:08d}_cam.txt'.format(i)), 'w+')

            camlines = ['extrinsics\n']
            for ei in range(4):
                camlines.append('')
                for ej in range(4):
                    camlines[-1] += '{:.3f} '.format(cameras[i][1][ei, ej]) 
                camlines[-1] += '\n'

            camlines.append('\nintrinsics\n')
            for ii in range(3):
                camlines.append('')
                for ij in range(3):
                    camlines[-1] += '{:.3f} '.format(cameras[i][0][ii, ij]) 
                camlines[-1] += '\n'

            camlines.append('\n')
            camlines.append('{} {}\n'.format(depth_range[0],depth_range[1]))

            camfile.writelines(camlines)
            camfile.close()
        
            # write pairs
            pair_lines.append("{}\n{}".format(i, num_samples-1))
            pair_idx = np.argsort(pairs[i, :])[::-1]
            for pi in range(num_samples):
                if pair_idx[pi] == i:
                    continue
                else:
                    pair_lines[-1]+= ' {} {:.3f}'.format(pair_idx[pi], pairs[i, pair_idx[pi]])
            pair_lines[-1] += '\n'
        
        pair_file = open(os.path.join(path, 'scan1', 'pair.txt'), 'w+')
        pair_file.writelines(pair_lines)
        pair_file.close()
    
        return path

    @classmethod
    def writeOKLog(cls, message):
        print("{}{}{}".format(cls.OKGREEN, message, cls.ENDC))



    