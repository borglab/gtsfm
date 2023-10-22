"""Spanning-tree-based global rotation estimator."""
from typing import Dict, List, Optional, Tuple

import networkx as nx
from gtsam import Rot3

from gtsfm.common.pose_prior import PosePrior
import gtsfm.utils.graph as graph_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase


class SpanningTreeRotationEstimator(RotationAveragingBase):

    def run_rotation_averaging(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
    ) -> List[Optional[Rot3]]:
        """Estimates rotations by greedily assembling a spanning tree and composing Rot(3) measurements.

        NOTE: this is not a minimum spanning tree.

        Args:
            num_images: Number of images. Since we have one pose per image, it is also the number of poses.
            i2Ri1_dict: Relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.
            i1Ti2_priors: Priors on relative poses.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
            `num_images`. The list may contain `None` where the global rotation could not be computed (either
            underconstrained system or ill-constrained system), or where the camera pose had no valid observation
            in the input to run_rotation_averaging().
        """
        edges = i2Ri1_dict.keys()

        num_nodes = max([max(i1, i2) for i1, i2 in edges]) + 1

        # Find the largest connected component.
        cc_nodes = graph_utils.get_nodes_in_largest_connected_component(edges)
        cc_nodes = sorted(cc_nodes)

        wRi_list = [None] * num_nodes
        # Choose origin node.
        origin_node = cc_nodes[0]
        wRi_list[origin_node] = Rot3()

        G = nx.Graph()
        G.add_edges_from(edges)

        # Ignore 0th node, as we already set its global pose as the origin.
        for dst_node in cc_nodes[1:]:
            # Determine the path to this node from the origin. ordered from [origin_node,...,dst_node]
            path = nx.shortest_path(G, source=origin_node, target=dst_node)

            wRi = Rot3()
            for i1, i2 in zip(path[:-1], path[1:]):
                # NOTE: i1, i2 may not be in sorted order here. May need to reverse ordering.
                if i1 < i2:
                    i1Ri2 = i2Ri1_dict[(i1, i2)].inverse()
                else:
                    i1Ri2 = i2Ri1_dict[(i2, i1)]

                # wRi = wR0 * 0R1
                wRi = wRi.compose(i1Ri2)

            wRi_list[dst_node] = wRi

        return wRi_list
