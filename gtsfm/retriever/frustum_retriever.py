""" Camera frustum-based retriever, that uses intersections of camera frustums to propose image pairs.

Only useful for datasets with pose estimates / priors.

Authors: Jon Womack
"""
from typing import List, Tuple

from Geometry3D import *
import math
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()


class FrustumRetriever(RetrieverBase):
    def __init__(self, max_frame_lookahead: int, intersection_threshold: int) -> None:
        """
        Args:
            max_frame_lookahead: maximum number of consecutive frames to consider for matching/co-visibility.
        """
        self._max_frame_lookahead = max_frame_lookahead
        self._intersection_threshold = intersection_threshold

    def Rx(theta):
        return np.matrix([[1, 0, 0],
                          [0, math.cos(theta), -math.sin(theta)],
                          [0, math.sin(theta), math.cos(theta)]])

    def Ry(theta):
        return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                          [0, 1, 0],
                          [-math.sin(theta), 0, math.cos(theta)]])

    def Rz(theta):
        return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                          [math.sin(theta), math.cos(theta), 0],
                          [0, 0, 1]])

    def fov_and_depth_to_pentahedron(fovx, fovy, depth):
        # Calculate width
        fovx_radians = np.radians(fovx)
        width = 2 * depth * np.tan(fovx_radians / 2)
        # Calculate height
        fovy_radians = np.radians(fovy)
        height = 2 * depth * np.tan(fovy_radians / 2)
        return width, height

    def make_pentahedron(pinhole, width, height, depth, roll, pitch, yaw):
        e1 = [pinhole[0], pinhole[1], pinhole[2]]
        a1 = [width / 2, height / 2, depth]
        b1 = [width / 2, -height / 2, depth]
        c1 = [- width / 2, height / 2, depth]
        d1 = [- width / 2, -height / 2, depth]
        # rotated
        rotate_all = np.matmul(np.matmul(Rx(roll), Ry(pitch)), Ry(yaw))
        a1r = np.matmul(rotate_all, np.asarray(a1))
        b1r = np.matmul(rotate_all, np.asarray(b1))
        c1r = np.matmul(rotate_all, np.asarray(c1))
        d1r = np.matmul(rotate_all, np.asarray(d1))
        # translated
        a1t = np.squeeze(a1r + e1)
        b1t = np.squeeze(b1r + e1)
        c1t = np.squeeze(c1r + e1)
        d1t = np.squeeze(d1r + e1)

        a1t = np.asarray(a1t).flatten()
        b1t = np.asarray(b1t).flatten()
        c1t = np.asarray(c1t).flatten()
        d1t = np.asarray(d1t).flatten()

        a = Point(a1t)
        b = Point(b1t)
        c = Point(c1t)
        d = Point(d1t)
        e = Point(e1)

        face1 = ConvexPolygon((a, b, c, d))
        face2 = ConvexPolygon((a, b, e))
        face3 = ConvexPolygon((c, a, e))
        face4 = ConvexPolygon((c, d, e))
        face5 = ConvexPolygon((a, b, e))
        pentahedron = ConvexPolyhedron((face1, face2, face3, face4, face5))
        return pentahedron

    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        num_images = len(loader)

        pairs = []
        for i1 in range(num_images):
            frustum1 = self.make_pentahedron()
            max_i2 = min(i1 + self._max_frame_lookahead + 1, num_images)
            for i2 in range(i1 + 1, max_i2):
                frustum2 = self.make_pentahedron()
                if intersection(frustum1, frustum2) is not None: #TODO: Find threshold like intersection > 25% union (frustum volume)
                    pairs.append((i1, i2))

        logger.info("Found %d pairs from the FrustumRetriever", len(pairs))
        return pairs


