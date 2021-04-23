"""MVSNets Parser class 
    parse GtsfmData to fit input datatype of mvsnets

Authors: Ren Liu
"""
from typing import Dict, List, Any

import math
import numpy as np 

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.densify.mvsnets.mvsUtills import Math

class Parser(object):
    """Parser class for parsing GtsfmData to fit MVSNets """
    
    @classmethod
    def parse_camera_matrix(cls, sfmData: GtsfmData) -> List:
        """Parse camera extrinsics and intrinsics from GtsfmData
        
        Args:
            sfm_result: pre-computed GtsfmData
        
        Returns: 
            List of camera parameters for MVSNets, the length is the number of cameras
                each entrance camera[i] contains a 3x3 intrinsic matrix and a 4x4 extrinsic matrix
        """

        cn = sfmData.number_images()

        cameras = []

        for ci in range(cn):
            intrinsics_i = sfmData.get_camera(ci).calibration().K()

            extrinsics_i = np.linalg.inv(sfmData.get_camera(ci).pose().matrix())

            cameras.append([intrinsics_i, extrinsics_i])

        return cameras

    @classmethod
    def parse_sparse_point_cloud(cls, sfmData: GtsfmData, cameras: List) -> (np.ndarray, np.ndarray):
        """ parse pair distances and depth ranges for each camera

        Args:
            sfm_result: pre-computed GtsfmData,
            camera: List of camera parameters for MVSNets, the length is the number of cameras
                each entrance camera[i] contains a 3x3 intrinsic matrix and a 4x4 extrinsic matrix
        
        Returns: 
            pairs: a np.ndarray of shape [N, N], which calculates the pair distances between each view pairs,
            depth_range: a np.ndarray of shape [N, 3], which calculates the minimum depth, maximum depth, and the number of virtual depth layers for each view
        
        """
        
        cn = sfmData.number_images()
        tn = sfmData.number_tracks()
        pairs = np.zeros([cn, cn])

        depth_array_cam = [[] for i in range(cn)]

        for ci in range(cn):
            for cj in range(ci+1, cn):
                pairs[ci, cj] = 0
                for ti in range(tn):
                    track_i = sfmData.get_track(ti)
                    mn = track_i.number_measurements()
                    idx_m = [-1, -1]
                    for mi in range(mn):
                        if track_i.measurement(mi)[0] == ci:
                            idx_m[0] = mi
                        elif track_i.measurement(mi)[0] == cj:
                            idx_m[1] = mi
                    if idx_m[0] >= 0 and idx_m[1] >= 0: #both cameras have measurements on this track
                        p =track_i.point3()

                        pi = Math.to_cam_coord(p, cameras[ci][1])
                        pj = Math.to_cam_coord(p, cameras[cj][1])

                        depth_array_cam[ci].append(pi[-1])
                        depth_array_cam[cj].append(pj[-1])
                        
                        score_ij = Math.piecewiseGaussian(pi, pj)
                        pairs[ci, cj] += score_ij
                        pairs[cj, ci] += score_ij

        min_depth = [ np.floor(np.mean(depth_array_cam[i]) - np.std(depth_array_cam[i]) ) for i in range(cn)] 
        max_depth = [  np.ceil(np.mean(depth_array_cam[i]) + np.std(depth_array_cam[i]) ) for i in range(cn)] 
        
        depth_layer_numer = [ 192 for i in range(cn) ]

        depth_range = np.array([min_depth, max_depth, depth_layer_numer])

        return pairs, depth_range

    @classmethod
    def to_mvsnets_data(
        cls, 
        images: np.ndarray, 
        sfmData: GtsfmData
    ) -> Dict[str, Any]:

        """a combination parsing functions to parse images and GtsfmData to fit MVSNets 
        
        Args:
            images: np.ndarray list of images, the shape is [N, H, W]
            sfm_result: object containing camera parameters and the optimized point cloud.

        Returns:
            a dictionary includes necessary information for MVSNets that parsed from GtsfmData
        
        """
    
        cameras = cls.parse_camera_matrix(sfmData)
   
        pairs, depthRange = cls.parse_sparse_point_cloud(sfmData, cameras)

        return {
            'images': images,
            'cameras': cameras,
            'pairs': pairs,
            'depthRange': depthRange
        }