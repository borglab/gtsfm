import torch
import cv2
import numpy as np
import os.path as osp
import tqdm
from PIL import Image, ImageOps

from ..third_party.loftr import LoFTR, default_cfg

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new


def read_image(img_pth, img_size, df, padding):
    if str(img_pth).endswith('gif'):
        
        pil_image = ImageOps.grayscale(Image.open(str(img_pth)))
        img_raw = np.array(pil_image)
    else:
        img_raw = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)

    w, h = img_raw.shape[1], img_raw.shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    if padding:  # padding
        pad_to = max(h_new, w_new)    
        mask = np.zeros((1,pad_to, pad_to), dtype=bool)
        mask[:,:h_new,:w_new] = True
        mask = mask[:,::8,::8]
    
    image = cv2.resize(img_raw, (w_new, h_new))
    pad_image = np.zeros((1,1, pad_to, pad_to), dtype=np.float32)
    pad_image[0,0,:h_new,:w_new]=image/255.

    return pad_image, mask


def save_loftr_matches(data_path, pair_path, output_path, model_weight_path="weights/outdoor_ds.ckpt"):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_weight_path)['state_dict'])
    matcher = matcher.eval().cuda()

    pairs_info = np.load(pair_path, allow_pickle=True)
    img_size = 1024
    df = 8
    padding = True

    for idx in tqdm.tqdm(range(pairs_info.shape[0])):
        if osp.exists(output_path+'loftr_match/%d.npy'%idx):
            continue
        name0, name1, _, _, _ = pairs_info[idx]

        img0_pth = osp.join(data_path, name0)
        img1_pth = osp.join(data_path, name1)
        img0_raw, mask0 = read_image(img0_pth, img_size, df, padding)
        img1_raw, mask1 = read_image(img1_pth, img_size, df, padding)        
        img0 = torch.from_numpy(img0_raw).cuda()
        img1 = torch.from_numpy(img1_raw).cuda()
        mask0 = torch.from_numpy(mask0).cuda()
        mask1 = torch.from_numpy(mask1).cuda()
        batch = {'image0': img0, 'image1': img1, 'mask0': mask0, 'mask1':mask1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            np.save(output_path+'loftr_match/%d.npy'%idx, {"kpt0": mkpts0, "kpt1": mkpts1, "conf": mconf})


