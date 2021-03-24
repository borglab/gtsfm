import numpy as np
import cv2 
import glob

class DepthmapMetrics(object):
    def __call__(self, depthmaps, masks):
        pass 

class Accuracy(DepthmapMetrics):
    def __call__(self, depthmaps, masks):
        final_mask = masks[0] * masks[1]
        # final_mask = np.ones(masks[0].shape, dtype=bool)  # no mask
        dist = np.sqrt(np.square(depthmaps[0][final_mask] - depthmaps[1][final_mask]).mean()) 

        return dist

class Completeness(DepthmapMetrics):
    def __call__(self, depthmaps, masks):
        return masks[0].sum() / masks[1].sum()


if __name__=='__main__':
    from depthmap_reader import DepthmapReaderManager
    colmap_pattern = 'results_densify/depthmap_colmap/*.geometric.bin'
    output_root = 'results_densify/outputs'
    # output_root = 'results_densify/outputs_colmap_cams'
    # output_root = 'results_densify/outputs_gt_cams_modify'
    mvsnet_pattern = f'{output_root}/scan1/depth_img/depth_*.png'
    mvsnet_mask = f'{output_root}/scan1/mask/*_final.png'
    colmap_reader = DepthmapReaderManager.build_depthmap_reader(colmap_pattern, 'COLMAP')
    colmap_depthmap = colmap_reader.load()
    
    mvsnet_reader = DepthmapReaderManager.build_depthmap_reader(mvsnet_pattern, 'PNG')
    mvsnet_depthmap = mvsnet_reader.load()

    h, w = mvsnet_depthmap[0].shape 

    colmap_depthmap = [cv2.resize(depthmap, (w, h)) for depthmap in colmap_depthmap]
    colmap_mask = [depthmap > 10 for depthmap in colmap_depthmap]

    mask_files = sorted(glob.glob(mvsnet_mask))
    mvsnet_mask = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) > 0 for mask in mask_files]
    
    dense_metrics = np.zeros([2])
    for i in range(len(mvsnet_depthmap)):
        # print(np.mean(mvsnet_depthmap[i]))
        # print(np.mean(colmap_depthmap[i]))
        dense_metrics[0] += Accuracy()([mvsnet_depthmap[i], colmap_depthmap[i]], [mvsnet_mask[i], colmap_mask[i]])
        dense_metrics[1] += Completeness()([mvsnet_depthmap[i], colmap_depthmap[i]], [mvsnet_mask[i], colmap_mask[i]])
    dense_metrics /= len(mvsnet_depthmap)

    print(mvsnet_depthmap[4].mean(), colmap_depthmap[4].mean())
    print(dense_metrics)
