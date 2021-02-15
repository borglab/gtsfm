from view_frustum import ViewFrustum
from scipy.spatial.transform import Rotation
from se3 import SE3
import numpy as np

sampleFrustum = ViewFrustum(fx=1532.24, img_w=1920, img_h=1080)
rotation = Rotation.from_quat(
    [-0.0659275, 0.011254, 0.0217253, 0.997524]).as_matrix()
translation = np.array([0.13125, -0.708675, -2.39795])
wTc = SE3(rotation, translation)
verts_worldfr = sampleFrustum.get_mesh_edges_worldframe(wTc)
print(verts_worldfr)
