import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import cv2

from io_util import write_keypoints, write_descriptors


def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='/home/liao/zjh/datasets/',
  help='datasets path.')
parser.add_argument('--seqs', type=str, default='Fountain',
  help='split by .')
parser.add_argument('--img_glob', type=str, default='*',
  help='Glob match if directory of images is specified (default: \'*.png\').')
parser.add_argument('--num_kp', type=int, default='8000',
  help='keypoint number')
parser.add_argument('--suffix', type=str, default='sift-8000',
  help='prefix of filename.')



class ExtractSIFT(object):
  def __init__(self, num_kp, contrastThreshold=1e-5):
    self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.sift.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc


if __name__ == "__main__":
  opt = parser.parse_args()
  seqs = opt.seqs.split('.')
  detector = ExtractSIFT(opt.num_kp)
  for seq in seqs:
    if not os.path.exists(opt.input_path+seq+'/keypoints'):
        os.system('mkdir '+opt.input_path+seq+'/keypoints')
    if not os.path.exists(opt.input_path+seq+'/descriptors'):
        os.system('mkdir '+opt.input_path+seq+'/descriptors')
    # get image lists
    search = os.path.join(opt.input_path, seq, 'images', opt.img_glob)
    listing = glob.glob(search)

    for img_path in tqdm(listing):
      kp, desc = detector.run(img_path)
      paths = img_path.split('/')
      img_name = paths[-1]
      save_path = '/'.join(paths[:-2])
      kp_path = os.path.join(save_path, 'keypoints', img_name+'.'+opt.suffix+'.bin')
      desc_path = os.path.join(save_path, 'descriptors', img_name+'.'+opt.suffix+'.bin')
      write_keypoints(kp_path, kp)
      write_descriptors(desc_path, desc)
    












