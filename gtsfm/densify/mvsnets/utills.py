import math
import numpy as np 

class Math(object):
    
    @classmethod
    def theta_ij(cls, ci, cj, p):
        return (180.0 / math.pi) * math.acos(np.dot(ci-p, cj-p))
    
    @classmethod
    def piecewiseGaussian(cls, ci, cj, p, theta_0 = 5, sigma_1=1, sigma_2=10):
        theta = cls.theta_ij(ci, cj, p)
        if theta <= theta_0:
            return math.exp(-(theta-theta_0)**2 / (2*sigma_1**2))
        else:
            return math.exp(-(theta-theta_0)**2 / (2*sigma_2**2))
    

def pair_score(ci, cj, tracks):
    
    return NotImplemented

