from __future__ import print_function
import numpy as np
import sys
from tqdm import tqdm
import os
import pickle
import cv2
import itertools
from six.moves import xrange
from feature_match import computeNN
from utils import saveh5, loadh5
from geom import load_geom, parse_geom, get_episym
from transformations import quaternion_from_matrix

class Sequence(object):
    def __init__(self, dataset_path, dump_dir, desc_name, vis_th, pair_num, pair_name=None):
        self.data_path = dataset_path.rstrip("/") + "/"
        self.dump_dir = dump_dir
        self.desc_name = desc_name
        print('dump dir ' + self.dump_dir)
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)
        self.intermediate_dir = os.path.join(self.dump_dir, 'dump')
        if not os.path.exists(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)
        img_list_file = self.data_path + "images.txt"
        geom_list_file = self.data_path + "calibration.txt"
        vis_list_file = self.data_path + "visibility.txt"
        self.image_fullpath_list = self.parse_list_file(self.data_path, img_list_file)
        self.geom_fullpath_list = self.parse_list_file(self.data_path, geom_list_file)
        self.vis_fullpath_list = self.parse_list_file(self.data_path, vis_list_file)
        # load geom and vis
        self.geom, self.vis = [], []
        for geom_file, vis_file in zip(self.geom_fullpath_list, self.vis_fullpath_list):
            self.geom += [load_geom(geom_file)]
            self.vis += [np.loadtxt(vis_file).flatten().astype("float32")]
        self.vis = np.asarray(self.vis)
        img_num = len(self.image_fullpath_list)
        if pair_name is None:
            self.pairs = []
            for ii, jj in itertools.product(xrange(img_num), xrange(img_num)):
                if ii != jj and self.vis[ii][jj] > vis_th:
                    self.pairs.append((ii, jj))
            np.random.seed(1234)
            self.pairs = [self.pairs[i] for i in np.random.permutation(len(self.pairs))[:pair_num]]
        else:
            with open(pair_name, 'rb') as f:
                self.pairs = pickle.load(f)
        print('pair lens' + str(len(self.pairs)))

    def dump_nn(self, ii, jj):
        dump_file = os.path.join(self.intermediate_dir, "nn-{}-{}.h5".format(ii, jj))
        if not os.path.exists(dump_file):
            image_i, image_j = self.image_fullpath_list[ii], self.image_fullpath_list[jj]
            desc_ii = loadh5(image_i+'.'+self.desc_name+'.hdf5')["descriptors"]
            desc_jj = loadh5(image_j+'.'+self.desc_name+'.hdf5')["descriptors"]
            idx_sort, ratio_test, mutual_nearest = computeNN(desc_ii, desc_jj)
            # Dump to disk
            dump_dict = {}
            dump_dict["idx_sort"] = idx_sort
            dump_dict["ratio_test"] = ratio_test
            dump_dict["mutual_nearest"] = mutual_nearest
            saveh5(dump_dict, dump_file)

    def dump_intermediate(self):
        for ii, jj in tqdm(self.pairs):
            self.dump_nn(ii,jj)
        print('Done')

    def unpack_K(self, geom):
        img_size, K = geom['img_size'], geom['K']
        w, h = img_size[0], img_size[1]
        cx = (w - 1.0) * 0.5
        cy = (h - 1.0) * 0.5
        cx += K[0, 2]
        cy += K[1, 2]
        # Get focals
        fx = K[0, 0]
        fy = K[1, 1]
        return cx,cy,[fx,fy]

    def norm_kp(self, cx, cy, fx, fy, kp):
        # New kp
        kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
        return kp

    def make_xy(self, ii, jj):
        geom_i, geom_j = parse_geom(self.geom[ii]), parse_geom(self.geom[jj])
        # should check the image size here
        #load img and check img_size
        image_i, image_j = self.image_fullpath_list[ii], self.image_fullpath_list[jj]
        kp_i = loadh5(image_i+'.'+self.desc_name+'.hdf5')["keypoints"][:, :2]
        kp_j = loadh5(image_j+'.'+self.desc_name+'.hdf5')["keypoints"][:, :2]
        cx1, cy1, f1 = self.unpack_K(geom_i)
        cx2, cy2, f2 = self.unpack_K(geom_j) 
        x1 = self.norm_kp(cx1, cy1, f1[0], f1[1], kp_i)
        x2 = self.norm_kp(cx2, cy2, f2[0], f2[1], kp_j)
        R_i, R_j = geom_i["R"], geom_j["R"]
        dR = np.dot(R_j, R_i.T)
        t_i, t_j = geom_i["t"].reshape([3, 1]), geom_j["t"].reshape([3, 1])
        dt = t_j - np.dot(dR, t_i)
        if np.sqrt(np.sum(dt**2)) <= 1e-5:
            return []
        dtnorm = np.sqrt(np.sum(dt**2))
        dt /= dtnorm
        nn_info = loadh5(os.path.join(self.intermediate_dir, "nn-{}-{}.h5".format(ii, jj)))
        idx_sort, ratio_test, mutual_nearest = nn_info["idx_sort"], nn_info["ratio_test"], nn_info["mutual_nearest"]
        x2 = x2[idx_sort[1],:]
        xs = np.concatenate([x1, x2], axis=1).reshape(1,-1,4)
        geod_d = get_episym(x1, x2, dR, dt)
        ys = geod_d.reshape(-1,1)
        return xs, ys, dR, dt, ratio_test, mutual_nearest, cx1, cy1, f1, cx2, cy2, f2

    def dump_datasets(self):
        ready_file = os.path.join(self.dump_dir, "ready")
        var_name = ['xs', 'ys', 'Rs', 'ts', 'ratios', 'mutuals', 'cx1s', 'cy1s', 'f1s', 'cx2s', 'cy2s', 'f2s']
        res_dict = {}
        for name in var_name:
            res_dict[name] = []
        if not os.path.exists(ready_file):
            print("\n -- No ready file {}".format(ready_file))
            for pair_idx, pair in enumerate(self.pairs):
                print("\rWorking on {} / {}".format(pair_idx, len(self.pairs)), end="")
                sys.stdout.flush()
                res = self.make_xy(pair[0], pair[1])
                if len(res)!=0:
                    for var_idx, name in enumerate(var_name):
                        res_dict[name] += [res[var_idx]]
            for name in var_name:
                out_file_name = os.path.join(self.dump_dir, name) + ".pkl"
                with open(out_file_name, "wb") as ofp:
                    pickle.dump(res_dict[name], ofp)
            # Mark ready
            with open(ready_file, "w") as ofp:
                ofp.write("This folder is ready\n")
        else:
             print('Done!')   


    def parse_list_file(self, data_path, list_file):
        fullpath_list = []
        with open(list_file, "r") as img_list:
            while True:
                # read a single line
                tmp = img_list.readline()
                if type(tmp) != str:
                    line2parse = tmp.decode("utf-8")
                else:
                    line2parse = tmp
                if not line2parse:
                    break
                # strip the newline at the end and add to list with full path
                fullpath_list += [data_path + line2parse.rstrip("\n")]
        return fullpath_list

    
