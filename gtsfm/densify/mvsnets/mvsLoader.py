from PIL import Image
import glob 
import os 
import cv2
import re
import numpy as np

class Loader(object):
    @classmethod
    def load_raw_images(cls, image_path, image_extension):
        img_files = glob.glob(os.path.join(image_path, "images", "*.{}".format(image_extension)))
        img_files = sorted(img_files)
        images = []
        for img_file in img_files:
            im = Image.open(img_file)
            images.append(im)

        return images

    @classmethod
    def load_labeled_cameras(cls, camera_path):
        intrinsic_files = glob.glob(os.path.join(camera_path, "intrinsics", "*.npy"))
        extrinsic_files = glob.glob(os.path.join(camera_path, "extrinsics", "*.npy"))
        intrinsic_files = sorted(intrinsic_files)
        extrinsic_files = sorted(extrinsic_files)

        cameras = []

        for i in range(len(intrinsic_files)):
            intrinsic = np.load(intrinsic_files[i])
            extrinsic = np.load(extrinsic_files[i])
            cameras.append([intrinsic, extrinsic])

        return cameras

    @classmethod
    def load_pfm(cls, file):
        
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0: 
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)

        return data
    