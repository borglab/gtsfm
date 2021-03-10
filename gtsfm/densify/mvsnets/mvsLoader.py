from PIL import Image
import glob 
import os 

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