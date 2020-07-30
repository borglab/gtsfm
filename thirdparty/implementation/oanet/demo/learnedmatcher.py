import torch
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
from collections import namedtuple
import sys
sys.path.append('../core')
from oan import OANet
from io_util import read_keypoints, read_descriptors, write_matches

class NNMatcher(object):
    """docstring for NNMatcher"""
    def __init__(self, ):
        super(NNMatcher, self).__init__()

    def run(self, nkpts, descs):
        # pts1, pts2: N*2 GPU torch tensor
        # desc1, desc2: N*C GPU torch tensor
        # corr: N*4
        # sides: N*2
        # corr_idx: N*2

        pts1, pts2, desc1, desc2 = nkpts[0], nkpts[1], descs[0], descs[1]
        d1, d2 = (desc1**2).sum(1), (desc2**2).sum(1)
        distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc1, desc2.transpose(0,1))).sqrt()
        dist_vals, nn_idx1 = torch.topk(distmat, k=2, dim=1, largest=False)
        nn_idx1 = nn_idx1[:,0]
        _, nn_idx2 = torch.topk(distmat, k=1, dim=0, largest=False)
        nn_idx2= nn_idx2.squeeze()
        mutual_nearest = (nn_idx2[nn_idx1] == torch.arange(nn_idx1.shape[0]).cuda())
        ratio_test = dist_vals[:,0] / dist_vals[:,1].clamp(min=1e-15)
        pts2_match = pts2[nn_idx1, :]
        corr = torch.cat([pts1, pts2_match], dim=-1)
        corr_idx = torch.cat([torch.arange(nn_idx1.shape[0]).unsqueeze(-1), nn_idx1.unsqueeze(-1).cpu()], dim=-1)
        sides = torch.cat([ratio_test.unsqueeze(1), mutual_nearest.float().unsqueeze(1)], dim=1)
        return corr, sides, corr_idx

    def infer(self, kpt_list, desc_list):
        nkpts = [torch.from_numpy(i[:,:2].astype(np.float32)).cuda() for i in kpt_list]
        descs = [torch.from_numpy(desc.astype(np.float32)).cuda() for desc in desc_list]
        corr, sides, corr_idx = self.run(nkpts, descs)
        inlier_idx = np.where(sides[:,1].cpu().numpy())
        matches = corr_idx[inlier_idx[0], :].numpy().astype('int32')
        corr0 = kpt_list[0][matches[:, 0]]
        corr1 = kpt_list[1][matches[:, 1]]
        return matches, corr0, corr1

        
class LearnedMatcher(object):
    def __init__(self, model_path, inlier_threshold=0, use_ratio=2, use_mutual=2):
        self.default_config = {}
        self.default_config['net_channels'] = 128
        self.default_config['net_depth'] = 12
        self.default_config['clusters'] = 500
        self.default_config['use_ratio'] = use_ratio
        self.default_config['use_mutual'] = use_mutual
        self.default_config['iter_num'] = 1
        self.default_config['inlier_threshold'] = inlier_threshold
        self.default_config = namedtuple("Config", self.default_config.keys())(*self.default_config.values())

        self.model = OANet(self.default_config)

        print('load model from ' +model_path)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        self.model.eval()
        self.nn_matcher = NNMatcher()

    def normalize_kpts(self, kpts):
        x_mean = np.mean(kpts, axis=0)
        dist = kpts - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3,3])
        T[0,0], T[1,1], T[2,2] = scale, scale, 1
        T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
        nkpts = kpts * np.asarray([T[0, 0], T[1, 1]]) + np.array([T[0, 2], T[1, 2]])
        return nkpts



    def infer(self, kpt_list, desc_list):
        with torch.no_grad():
            nkpts = [torch.from_numpy(self.normalize_kpts(i[:,:2]).astype(np.float32)).cuda() for i in kpt_list]
            descs = [torch.from_numpy(desc.astype(np.float32)).cuda() for desc in desc_list]
            corr, sides, corr_idx = self.nn_matcher.run(nkpts, descs)
            corr, sides = corr.unsqueeze(0).unsqueeze(0), sides.unsqueeze(0)
            data = {}
            data['xs'] = corr
            # currently supported mode:
            if self.default_config.use_ratio==2 and self.default_config.use_mutual==2:
                data['sides'] = sides
            elif self.default_config.use_ratio==0 and self.default_config.use_mutual==1:
                mutual = sides[0,:,1]>0
                data['xs'] = corr[:,:,mutual,:]
                data['sides'] = []
                corr_idx = corr_idx[mutual,:]
            elif self.default_config.use_ratio==1 and self.default_config.use_mutual==0:
                ratio = sides[0,:,0] < 0.8
                data['xs'] = corr[:,:,ratio,:]
                data['sides'] = []
                corr_idx = corr_idx[ratio,:]
            elif self.default_config.use_ratio==1 and self.default_config.use_mutual==1:
                mask = (sides[0,:,0] < 0.8) & (sides[0,:,1]>0)
                data['xs'] = corr[:,:,mask,:]
                data['sides'] = []
                corr_idx = corr_idx[mask,:]
            elif self.default_config.use_ratio==0 and self.default_config.use_mutual==0:
                data['sides'] = []
            else:
                raise NotImplementedError
            
            y_hat, e_hat = self.model(data)
            y = y_hat[-1][0, :].cpu().numpy()
            inlier_idx = np.where(y > self.default_config.inlier_threshold)
            matches = corr_idx[inlier_idx[0], :].numpy().astype('int32')
        corr0 = kpt_list[0][matches[:, 0]]
        corr1 = kpt_list[1][matches[:, 1]]
        return matches, corr0, corr1


def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='/home/liao/zjh/datasets/',
  help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--seqs', type=str, default='Fountain',
  help='split by .')
parser.add_argument('--img_glob', type=str, default='*',
  help='Glob match if directory of images is specified (default: \'*.png\').')
parser.add_argument('--input_suffix', type=str, default='sift-8000',
  help='prefix of filename.')
parser.add_argument('--output_suffix', type=str, default='sift-8000-our',
  help='prefix of filename.')
parser.add_argument('--use_prev_pairs', type=str2bool, default=False,
  help='use previous image pairs')
parser.add_argument('--prev_output_suffix', type=str, default='sift-8000',
  help='previous image pairs suffix')
parser.add_argument('--inlier_threshold', type=float, default=0,
  help='inlier threshold. default: 0')
parser.add_argument('--use_learned_matcher', type=str2bool, default=True,
  help='False: learned matcher, True: NN matcher')
parser.add_argument('--use_mutual', type=int, default=2,
  help='0: not use mutual. 1: use mutual before learned matcher. 2: use mutual as side information')
parser.add_argument('--use_ratio', type=int, default=2,
  help='0: not use ratio test. 1: use ratio test before learned matcher. 2: use ratio test as side information')
def dump_match(matcher, img1_name, img2_name, base_dir, input_suffix, output_suffix):
    kpt1_name = os.path.join(base_dir, 'keypoints', img1_name+'.'+input_suffix+'.bin')
    kpt2_name = os.path.join(base_dir, 'keypoints', img2_name+'.'+input_suffix+'.bin')
    desc1_name = os.path.join(base_dir, 'descriptors', img1_name+'.'+input_suffix+'.bin')
    desc2_name = os.path.join(base_dir, 'descriptors', img2_name+'.'+input_suffix+'.bin')
    kpt1, kpt2 = read_keypoints(kpt1_name), read_keypoints(kpt2_name)
    desc1, desc2 = read_descriptors(desc1_name), read_descriptors(desc2_name)
    match_name = img1_name+'---'+img2_name+'.'+output_suffix+'.bin'
    match_name = os.path.join(base_dir, 'matches', match_name)
    matches, _, _ = matcher.infer([kpt1, kpt2], [desc1, desc2])
    write_matches(match_name, matches)

if __name__ == "__main__":
  opt = parser.parse_args()
  seqs = opt.seqs.split('.')

  if not opt.use_learned_matcher:
    matcher = NNMatcher()
  else:
    if opt.use_ratio < 2 and opt.use_mutual < 2:
        model_path = os.path.join('../model', 'sift-8k/model_best.pth')
        matcher = LearnedMatcher(model_path, opt.inlier_threshold, use_ratio=opt.use_ratio, use_mutual=opt.use_mutual)
    elif opt.use_ratio == 2 and opt.use_mutual == 2:
        model_path = os.path.join('../model', 'sift-side-8k/model_best.pth')
        matcher = LearnedMatcher(model_path, opt.inlier_threshold, use_ratio=2, use_mutual=2)
    else:
        raise NotImplementedError
  for seq in seqs:
    if not os.path.exists(opt.input_path+seq+'/matches'):
        os.system('mkdir '+opt.input_path+seq+'/matches')
    if not opt.use_prev_pairs:
        # get image lists
        search = os.path.join(opt.input_path, seq, 'images', opt.img_glob)
        listing = glob.glob(search)
        listing.sort()
        pairs = []
        for img1 in range(len(listing)):
            for img2 in range(len(listing))[img1+1:]:
                img1_name, img2_name = listing[img1].split('/')[-1], listing[img2].split('/')[-1]
                pairs += [[img1_name, img2_name]]
    else:
        search = os.path.join(opt.input_path, seq, 'matches', "*---*."+opt.prev_output_suffix+'.bin')
        listing = glob.glob(search)
        pairs = [os.path.basename(path[:-5-len(opt.prev_output_suffix)]).split("---") for path in listing]
        
    for pair in tqdm(pairs):
        img1_name, img2_name = pair[0], pair[1]
        dump_match(matcher, img1_name, img2_name, os.path.join(opt.input_path, seq), opt.input_suffix, opt.output_suffix)


        


