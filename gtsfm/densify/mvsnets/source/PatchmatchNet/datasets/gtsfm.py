from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2

class MVSDataset(Dataset):
    def __init__(self, mvsnetsData, nviews, img_wh):
        super(MVSDataset, self).__init__()
        
        self.stage = 4
        self.mode = "test"
        self.nviews = nviews
        self.img_wh = img_wh
        self.data = mvsnetsData

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):

        pairs = self.data['pairs']

        num_viewpoint = pairs.shape[0]

        for i in range(num_viewpoint):
            pairs[i][i] = -np.inf 
        
        pair_idx = np.argsort(pairs, axis=0)[:, 1:][:, ::-1]

        metas = []

        for i in range(num_viewpoint):
            metas.append(("scan1", i, pair_idx[i].tolist()))

        print("dataset", self.mode, "metas:", len(metas))
        return metas
    
    def __len__(self):
        return len(self.metas)
    
    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []
        depth_min = None
        depth_max = None
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            img = self.data['images'][vid]
            np_img = np.array(img, dtype=np.float32) / 255.
            np_img = cv2.resize(np_img, self.img_wh, interpolation=cv2.INTER_LINEAR)
            
            h, w, _ = np_img.shape
                    
            imgs_0.append(np_img)
            imgs_1.append(cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR))
            imgs_2.append(cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_LINEAR))
            imgs_3.append(cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_LINEAR))

            intrinsics, extrinsics = self.data['cameras'][vid]
            depth_min_, depth_max_ = self.data['depthRange'][0][i], self.data['depthRange'][1][i]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)


            if i == 0:  # reference view
                depth_min = depth_min_
                depth_max = depth_max_

        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        imgs = {}
        imgs['stage_0'] = imgs_0
        imgs['stage_1'] = imgs_1
        imgs['stage_2'] = imgs_2
        imgs['stage_3'] = imgs_3
        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        proj={}
        proj['stage_3']=proj_matrices_3
        proj['stage_2']=proj_matrices_2
        proj['stage_1']=proj_matrices_1
        proj['stage_0']=proj_matrices_0

        return {"idx": idx,
                "imgs": imgs,                   # N*3*H0*W0
                "proj_matrices": proj,          # N*4*4
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "filename": '{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
