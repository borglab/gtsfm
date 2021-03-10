class Parser(object):
    @classmethod
    def parse_sparse_point_cloud(cls, sfmData):
        return NotImplemented
    
    @classmethod
    def parse_camera_matrix(cls, sfmData):
        return NotImplemented
    
    @classmethod
    def to_mvsnets_data(cls, images, cameras, )