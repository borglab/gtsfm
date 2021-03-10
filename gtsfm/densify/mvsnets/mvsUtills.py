import math
import numpy as np 

class Math(object):
    
    @classmethod
    def theta_ij(cls, pi, pj):
        return (180.0 / math.pi) * math.acos(np.dot(pi, pj) / np.linalg.norm(pi) / np.linalg.norm(pj))
    
    @classmethod
    def piecewiseGaussian(cls, pi, pj, theta_0=5, sigma_1=1, sigma_2=10):
        theta = cls.theta_ij(pi, pj)
        if theta <= theta_0:
            return math.exp(-(theta-theta_0)**2 / (2*sigma_1**2))
        else:
            return math.exp(-(theta-theta_0)**2 / (2*sigma_2**2))
    
    @classmethod
    def to_cam_coord(cls, p, extrinsics):
        homo_p = np.ones([4])
        homo_p[:3] = p
        cam_p = np.linalg.inv(extrinsics) @ homo_p.reshape([4, 1])
        cam_p /= cam_p[3,0]

        return cam_p.reshape([4,])[:3]                       




class MVSNetsModelManager(object):

    @classmethod
    def test(cls, method, args):
        model = None 
        if method.lower() == 'PatchmatchNet'.lower():
            from .source.PatchmatchNet.eval_gtsfm import eval_function

        eval_function(args)

        return model