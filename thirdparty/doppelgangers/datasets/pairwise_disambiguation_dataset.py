import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from ..utils.dataset import read_loftr_matches

class DoppelgangersDataset(Dataset):
    def __init__(self,
                 image_dir,
                 loftr_match_dir,
                 pair_path,
                 img_size,
                 phase,
                 **kwargs):
        """
        Doppelgangers dataset: loading images and loftr matches for Doppelgangers model.
        
        Args:
            image_dir (str): root directory for images.
            loftr_match_dir (str): root directory for loftr matches.
            pair_path (str): pair_list.npy path. This contains image pair information.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
        """
        super().__init__()

        self.phase = phase
        self.image_dir = image_dir    
        self.loftr_match_dir = loftr_match_dir
        self.pairs_info = np.load(pair_path[0], allow_pickle=True)
        self.pairs_info0_length = len(self.pairs_info)
        if len(pair_path)==2:
            self.pairs_info = np.concatenate([self.pairs_info, np.load(pair_path[1], allow_pickle=True)], axis=0)
        print('loading images, #pairs: ', len(self.pairs_info))
        self.img_size = img_size

        
    def __len__(self):
        return len(self.pairs_info)
    

    def __getitem__(self, idx):
        name0, name1, label, num_matches = self.pairs_info[idx]

        if idx >= self.pairs_info0_length:
            img_name0 = osp.join(self.image_dir[1], name0)
            img_name1 = osp.join(self.image_dir[1], name1)
            # index for loftr matches of Megadepth dataset is counting from 0
            pair_matches = np.load(osp.join(self.loftr_match_dir[1], '%d.npy'%(idx-self.pairs_info0_length)), allow_pickle=True).item()        
        else:
            img_name0 = osp.join(self.image_dir[0], name0)
            img_name1 = osp.join(self.image_dir[0], name1)
            if osp.exists(osp.join(self.loftr_match_dir[0], '%d.npy'%idx)):
                pair_matches = np.load(osp.join(self.loftr_match_dir[0], '%d.npy'%idx), allow_pickle=True).item()
            else:
                # portion of loftr matches of train_set_flip dataset is placed under train_set_noflip
                pair_matches = np.load(osp.join(self.loftr_match_dir[0].replace('flip','noflip'), '%d.npy'%idx), allow_pickle=True).item()

        keypoints0 = np.array(pair_matches['kpt0'])
        keypoints1 = np.array(pair_matches['kpt1'])
        conf = pair_matches['conf']

        if np.sum(conf>0.8) == 0:
            matches = None
        else:
            F, mask = cv2.findFundamentalMat(keypoints0[conf>0.8],keypoints1[conf>0.8],cv2.FM_RANSAC, 3, 0.99)
            if mask is None or F is None:
                matches = None
            else:
                matches = np.array(np.ones((keypoints0.shape[0], 2)) * np.arange(keypoints0.shape[0]).reshape(-1,1)).astype(int)[conf>0.8][mask.ravel()==1]

        image = read_loftr_matches(img_name0, img_name1, self.img_size, 8, True, keypoints0, keypoints1, matches, warp=True, conf=conf)
        
        data = {
            'image': image,  # (4, h, w)
            'gt': int(label)
        }

        return data

def get_datasets(cfg):
    tr_dataset = DoppelgangersDataset(
                cfg.train.image_dir,
                cfg.train.loftr_match_dir,
                cfg.train.pair_path,
                img_size=getattr(cfg.train, "img_size", 640),
                phase='Train')
    te_dataset = DoppelgangersDataset(
                cfg.test.image_dir,
                cfg.test.loftr_match_dir,
                cfg.test.pair_path,
                img_size=getattr(cfg.test, "img_size", 640),
                phase='Test')

    return tr_dataset, te_dataset


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_data_loaders(cfg):
    tr_dataset, te_dataset = get_datasets(cfg)
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=cfg.test.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)
    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader
    }
    return loaders


if __name__ == "__main__":
    pass