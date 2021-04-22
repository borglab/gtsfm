"""MVSNets utils class for gtsfm

Authors: Ren Liu
"""
from typing import Dict, Any

import math
import numpy as np 

class Math(object):
    """Some mathematic methods for MVSNets"""
    
    @classmethod
    def theta_ij(cls, pi: np.ndarray, pj: np.ndarray) -> float:
        """calculate the angle between vector pi and pj

        Args:
            pi: vector in np.ndarray of [3, ] shape
            pj: vector in np.ndarray of [3, ] shape

        Returns:
            angle in degree with float type
        """
        return (180.0 / math.pi) * math.acos(np.dot(pi, pj) / np.linalg.norm(pi) / np.linalg.norm(pj))
    
    @classmethod
    def piecewiseGaussian(
        cls, 
        pi: np.ndarray, 
        pj: np.ndarray, 
        theta_0: float = 5, 
        sigma_1: float = 1, 
        sigma_2: float = 10
        ) -> float:
        """calculate the piecewise gaussian value as pair distances
            reference: https://arxiv.org/abs/1804.02505

            Args:
                pi: pose vector in np.ndarray of [3, ] shape,
                pj: pose vector in np.ndarray of [3, ] shape,
                theta_0: float parameter,
                sigma_1: float parameter,
                sigma_2: float parameter
            
            Returns:
                float piecewice gaussian value
        """
        theta = cls.theta_ij(pi, pj)
        if theta <= theta_0:
            return math.exp(-(theta-theta_0)**2 / (2*sigma_1**2))
        else:
            return math.exp(-(theta-theta_0)**2 / (2*sigma_2**2))
    
    @classmethod
    def to_cam_coord(cls, p: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """convert world coordinates to camera coordinates

            Args:
                p: pose vector in np.ndarray of [3, ] shape,
                extrinsics: target camera extrinsics, a 4x4 np.ndarray
            
            Returns:
                pose vector in np.ndarray of [3, ] shape in target camera perspective
        """
        homo_p = np.ones([4])
        homo_p[:3] = p
        cam_p = np.linalg.inv(extrinsics) @ homo_p.reshape([4, 1])
        cam_p /= cam_p[3,0]

        return cam_p.reshape([4,])[:3]                       


class MVSNetsModelManager(object):
    """Model manager class for mvsnets, and call each specific eval functions according to args"""

    @classmethod
    def test(cls, method: str, args: Dict[str, Any]) -> np.ndarray:
        """call evaluation functions according to the method

            Args:
                method: string method which decides the method,
                args: an integrated dictionary with all necessary parameters for MVSNets

            Results:
                Dense point cloud, as an array of shape (N,3)  
        """

        model_func = None 
        if method.lower() == 'PatchmatchNet'.lower():
            from gtsfm.densify.mvsnets.methods.PatchmatchNet.eval_gtsfm import eval_function
            model_func = eval_function

        return model_func(args)