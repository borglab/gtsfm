import io
import cv2
import numpy as np
import h5py
import torch
import math
from PIL import Image, ImageOps

def imread_rgb(path):
    cv_type = cv2.IMREAD_COLOR
    if str(path).endswith('gif'):        
        pil_image = Image.open(str(path))
        image = np.array(pil_image)
    else:
        image = cv2.imread(str(path), cv_type)

    if len(image.shape)<3:
        image = image[:, :, np.newaxis]
    if image.shape[2]<3:
        image = np.concatenate((image, image, image), axis=2)
    return image  # (h, w, 3)


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


def read_loftr_matches(path0, path1, 
                       resize=None, 
                       df=None, 
                       padding=False, 
                       keypoints0=None, keypoints1=None, 
                       matches=None, 
                       warp=False, 
                       conf=None):
    """
    Args:
        path0, path1 (str): image path.        
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        keypoints0, keypoints1 (numpy.array (n, 2)): keypoint pixel coordiantes of images.
        matches (numpy.array (n, 2)): list of matched keypoint index.
        warp (bool): If the first image is aligned with the second image.
        conf (numpy.array (n,)): loftr matches confidence score
    Returns:
        image (torch.tensor): (2+2+3+3, h, w)     
    """

    image0 = imread_rgb(path0).astype(np.float32)
    w0, h0 = image0.shape[1], image0.shape[0]
    w_new0, h_new0 = get_resized_wh(w0, h0, resize)
    w_new0, h_new0 = get_divisible_wh(w_new0, h_new0, df)

    image1 = imread_rgb(path1).astype(np.float32)
    w1, h1 = image1.shape[1], image1.shape[0]
    w_new1, h_new1 = get_resized_wh(w1, h1, resize)
    w_new1, h_new1 = get_divisible_wh(w_new1, h_new1, df)

    valid_warp_pts = [False]
    warp_rgb = True

    # align image pair by estimating affine transformation from matches
    if (warp == True) and (matches is not None):
        src_pts = np.float32(keypoints0[matches[:, 0]]).reshape(-1, 1, 2)
        dst_pts = np.float32(keypoints1[matches[:, 1]]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold = 20.0)

        warp_keypoints0 = np.float32(keypoints0).reshape(-1, 1, 2)
        warp_keypoints0 = cv2.transform(warp_keypoints0, M)
        warp_keypoints0 = warp_keypoints0.reshape(-1, 2)  

        w_kpt = w_new1
        h_kpt = h_new1
        valid_warp_pts = (warp_keypoints0[:, 0] >= 0) & (warp_keypoints0[:, 0] < w_kpt) & (warp_keypoints0[:, 1] >= 0) & (warp_keypoints0[:, 1] < h_kpt)
        if valid_warp_pts.size == 0:
            valid_warp_pts = [False]
            warp_rgb = False

    # do not align image pair when warp == Flase, or when matches is not available, or when no valid keypoint after warping 
    if (warp == False) or (matches is None) or (np.sum(valid_warp_pts)==0):
        w_kpt = w_new0
        h_kpt = h_new0
        warp_keypoints0 = keypoints0
        valid_warp_pts = (warp_keypoints0[:,0]>=0) & (warp_keypoints0[:,0]<w_kpt) & (warp_keypoints0[:,1]>=0) & (warp_keypoints0[:,1]<h_kpt)
        warp_rgb = False

    if padding:
        pad_to = max(h_new1, w_new1)
        keypoint_match_mask = np.zeros((4, pad_to, pad_to), dtype=np.float32)        
    else:
        keypoint_match_mask = np.zeros((4, h_new1, w_new1), dtype=np.float32)   
    
    def f(x, max):
        return math.floor(x) if math.floor(x)<max else max-1
    int_array = np.vectorize(f)
    
    # Create a mask of keypoints: keypoint_match_mask[0] for image 0, keypoint_match_mask[2] for image 1.
    # Pixels corresponding to keypoints are set to one or confidence when available, and all other pixels are set to zero.
    if conf is None:
        if np.sum(valid_warp_pts)>0:
            keypoint_match_mask[0, int_array(warp_keypoints0[valid_warp_pts,1], h_kpt), int_array(warp_keypoints0[valid_warp_pts,0], w_kpt)] = 1.
        if len(keypoints1[:,1])>0:
            keypoint_match_mask[2, int_array(keypoints1[:,1], h_new1), int_array(keypoints1[:,0], w_new1)] = 1.
    else:
        if np.sum(valid_warp_pts)>0:
            keypoint_match_mask[0, int_array(warp_keypoints0[valid_warp_pts,1], h_kpt), int_array(warp_keypoints0[valid_warp_pts,0], w_kpt)] = conf[valid_warp_pts]
        if len(keypoints1[:,1])>0:
            keypoint_match_mask[2, int_array(keypoints1[:,1], h_new1), int_array(keypoints1[:,0], w_new1)] = conf

    # Create a mask of matches: keypoint_match_mask[1] for image 0, keypoint_match_mask[3] for image 1.
    # Pixels corresponding to matches are set to one, and all other pixels are set to zero.
    if matches is not None:
        if np.sum(valid_warp_pts[matches[:, 0]]) > 0:
            match_y0 = int_array(warp_keypoints0[matches[:,0],1][valid_warp_pts[matches[:,0]]], h_kpt)
            match_x0 = int_array(warp_keypoints0[matches[:,0],0][valid_warp_pts[matches[:,0]]], w_kpt)    
            keypoint_match_mask[1,match_y0,match_x0] = 1.
        match_y1 = int_array(keypoints1[matches[:,1],1], h_new1)
        match_x1 = int_array(keypoints1[matches[:,1],0], w_new1)    
        keypoint_match_mask[3,match_y1,match_x1] = 1.


    # Concatenate keypoint and match masks with images.
    image1 = cv2.resize(image1, (w_new1, h_new1))
    image1 = np.transpose(image1, (2,0,1))        
    rgb_image = np.zeros((6, keypoint_match_mask.shape[1], keypoint_match_mask.shape[2]), dtype=np.float32)
    rgb_image[:3,:h_new1,:w_new1] = image1/255.
    if matches is not None and warp_rgb==True:
        warp_image0 = cv2.resize(image0, (w_new0, h_new0))
        warp_image0 = cv2.warpAffine(warp_image0, M, (w_new1,h_new1), flags=cv2.INTER_AREA)            
        warp_image0 = np.transpose(warp_image0, (2,0,1))        
        rgb_image[3:,:h_new1,:w_new1] = warp_image0/255.
    else:
        image0 = cv2.resize(image0, (w_new0, h_new0))
        image0 = np.transpose(image0, (2,0,1))
        rgb_image[3:,:h_new0,:w_new0] = image0/255.
    keypoint_match_image = np.concatenate((keypoint_match_mask, rgb_image), axis=0)
    
    return torch.from_numpy(keypoint_match_image) 
