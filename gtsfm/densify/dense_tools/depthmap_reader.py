import argparse
import numpy as np
import os
import struct
import glob
import cv2

class DepthmapReader(object):
    def __init__(self, pattern):
        self.files = glob.glob(pattern)
        self.files = sorted(self.files)
    
    def parse_single_file(self, f):
        pass 

    def load(self):
        depthmaps = []
        for file in self.files:
            depthmaps.append(self.parse_single_file(file))
            
        return depthmaps            

class ColmapReader(DepthmapReader):
    def __init__(self, pattern):
        super(ColmapReader, self).__init__(pattern)
    
    def parse_single_file(self, file):
        f = open(file, "rb")
        width, height, channels = np.genfromtxt(f, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        f.seek(0)
        num_delimiter = 0
        byte = f.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = f.read(1)
        array = np.fromfile(f, np.float32)
        array = array.reshape((width, height, channels), order="F")
        f.close()
        return np.transpose(array, (1, 0, 2)).squeeze()
    
class PfmReader(DepthmapReader):
    def __init__(self, pattern):
        super(PfmReader, self).__init__(pattern)
    
    def parse_single_file(self, f):
        return NotImplemented  

class PngReader(DepthmapReader):
    def __init__(self, pattern):
        super(PngReader, self).__init__(pattern)
    
    def parse_single_file(self, f):
        values = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        t = np.load(f + '.npy')
        values = values * t[0] + t[1]

        return values  
    
class DepthmapReaderManager(object):
        
    @classmethod
    def build_depthmap_reader(cls, pattern, depthmap_type="pfm"):
        reader = None 
        if depthmap_type.strip().lower() == 'colmap':
            reader = ColmapReader(pattern)
        if depthmap_type.strip().lower() == 'pfm':
            reader = PfmReader(pattern)
        if depthmap_type.strip().lower() == 'png':
            reader = PngReader(pattern)
        
        return reader

if __name__=='__main__':
    reader = DepthmapReaderManager.build_depthmap_reader('results_densify/depthmap_colmap/*.geometric.bin', 'COLMAP')
    # reader = DepthmapReaderManager.build_depthmap_reader('results_densify/outputs/scan1/depth_img/depth_*.png', 'PNG')
    depthmap = reader.load()
    print(depthmap[4].shape)
    depthmap[4][depthmap[4]==0] = depthmap[4].mean()
    depthmap[4] = cv2.resize((depthmap[4] - depthmap[4].min()) / (depthmap[4].max()-depthmap[4].min()), (640, 960))
    cv2.imshow("test", depthmap[4])
    cv2.waitKey(0)