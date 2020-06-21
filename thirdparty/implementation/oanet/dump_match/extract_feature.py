import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import cv2
import h5py


def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='../raw_data/yfcc100m/',
  help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--img_glob', type=str, default='*/*/images/*.jpg',
  help='Glob match if directory of images is specified (default: \'*/images/*.jpg\').')
parser.add_argument('--num_kp', type=int, default='2000',
  help='keypoint number, default:2000')
parser.add_argument('--suffix', type=str, default='sift-2000',
  help='suffix of filename, default:sift-2000')



class ExtractSIFT(object):
  def __init__(self, num_kp, contrastThreshold=1e-5):
    self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.sift.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc

def write_feature(pts, desc, filename):
  with h5py.File(filename, "w") as ifp:
      ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
      ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
      ifp["keypoints"][:] = pts
      ifp["descriptors"][:] = desc

if __name__ == "__main__":
  opt = parser.parse_args()
  detector = ExtractSIFT(opt.num_kp)   
  # get image lists
  search = os.path.join(opt.input_path, opt.img_glob)
  listing = glob.glob(search)

  for img_path in tqdm(listing):
    kp, desc = detector.run(img_path)
    save_path = img_path+'.'+opt.suffix+'.hdf5'
    write_feature(kp, desc, save_path)
    




