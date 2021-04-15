"""MVSNets Parser class 
    parse GtsfmData to fit input datatype of mvsnets

Authors: Ren Liu
"""

import math
import numpy as np 
from gtsfm.densify.mvsnets.mvsUtills import Math

class Parser(object):
    @classmethod
    def parse_sparse_point_cloud(cls, sfmData, cameras):
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
        
        depth_layer_numer = 192

        depth_range = [min_depth, max_depth, depth_layer_numer]

        return pairs, depth_range
    
    @classmethod
    def parse_camera_matrix(cls, sfmData):
        cn = sfmData.number_images()

        cameras = []

        for ci in range(cn):
            intrinsics_i = sfmData.get_camera(ci).calibration().K()

            extrinsics_i = np.linalg.inv(sfmData.get_camera(ci).pose().matrix())

            cameras.append([intrinsics_i, extrinsics_i])

        return cameras
    

    @classmethod
    def to_mvsnets_data(cls, images, sfmData, labeled_cameras = None):
    
        if labeled_cameras:
            cameras = labeled_cameras
        else:
            cameras = cls.parse_camera_matrix(sfmData)
   
        pairs, depthRange = cls.parse_sparse_point_cloud(sfmData, cameras)

        return {
            'images': images,
            'cameras': cameras,
            'pairs': pairs,
            'depthRange': depthRange
        }