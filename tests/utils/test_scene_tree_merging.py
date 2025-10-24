"""
Unit tests for io utility functions.
Authors: Frank Dellaert.
"""

import pickle
import random
import unittest
from pathlib import Path

import numpy as np
from gtsam import Similarity3  # type: ignore

import gtsfm.products.scene_tree as scene_tree
import gtsfm.utils.merging as merging_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.types import create_camera
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.scene_tree import LocalScene, SceneTree
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.utils.tree import Tree


def merge(a: GtsfmData, b: GtsfmData) -> GtsfmData:
    """Merge another GtsfmData into this one in-place.

    Args:
        b: GtsfmData to merge into this one.

    Returns:
        Merged GtsfmData, a new object. Coordinate frame A is used as prime.
    """
    ap = a.poses()
    bp = b.poses()
    print(f"merge A:{a}, {sorted(ap.keys())[:3]}")
    print(f"merge B:{b}, {sorted(bp.keys())[:3]}")
    try:
        aTb = merging_utils.calculate_transform(ap, bp)
        print(f"Calculated aTb: {aTb}")
    except ValueError:
        print("Cannot merge GtsfmData objects.")
        return a

    # poor man scale estimation
    common_keys = [i for i in ap if i in bp]

    # For some random pair in common_keys, print out ratio of distances before and after
    i, j = common_keys[0], common_keys[-1]
    b_norm = np.linalg.norm(bp[i].translation() - bp[j].translation())
    a_norm = np.linalg.norm(ap[i].translation() - ap[j].translation())
    scale = float(a_norm / b_norm)
    print(f"Distance ratio (b/a) for {i}, {j}: {scale}")

    aSb1 = Similarity3(aTb.rotation().matrix(), aTb.translation(), scale)
    print(f"Calculated aSb1: {aSb1}")

    pose_pairs = [(ap[i], bp[i]) for i in common_keys]
    aSb2 = Similarity3.Align(pose_pairs)
    print(f"Calculated aSb2 from Similarity3.Align: {aSb2}")

    aSb = aSb2

    # Create merged cameras with updated poses. Only b-poses need to be updated.
    merged_cameras = a.cameras().copy()
    for i, cam in b.cameras().items():
        # TODO: what to do if we have conflicting calibrations?
        if i not in merged_cameras:
            bTi = cam.pose()
            merged_cameras[i] = create_camera(aSb.transformFrom(bTi), cam.calibration())

    # Create merged tracks
    merged_tracks = a.tracks().copy()
    # For all b_tracks, update the point by multiplying with aTb:
    for track in b.tracks():
        track.p = aSb.transformFrom(track.p)  # from b to a
        merged_tracks.append(track)

    # Create merged GtsfmData
    # Hack: don't remap tracks for now, just copy them over.
    merged_data = GtsfmData(len(merged_cameras), merged_cameras, [])
    merged_data._tracks = merged_tracks

    print(f"Merged data:{merged_data}, {sorted(merged_data.poses().keys())[:3]}\n")
    return merged_data


def merge_colmap_tree(tree: Tree[LocalScene]) -> GtsfmData:
    """Merge a cluster tree of ColmapScenes into a single GtsfmData.

    Args:
        tree: Tree whose values are (Path, GtsfmData) tuples.

    Returns:
        Merged GtsfmData.
    """

    def f(path_scene: LocalScene, merged_children) -> GtsfmData:
        path, scene = path_scene
        if scene is None:
            raise ValueError(f"merge_colmap_tree: scene at path {path} is None.")
        if len(merged_children) == 0:
            return scene
        print(f"Merging scene at path: {path}")
        merged_scene = scene
        for child in merged_children:
            merged_scene = merge(merged_scene, child)
        return merged_scene

    return tree.fold(f)


TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# Test for cluster tree IO
class TestClusterTreeIO(unittest.TestCase):
    def setUp(self) -> None:
        # Base directory containing the COLMAP hierarchy.
        self.base_dir = TEST_DATA_ROOT / "palace"

    def test_whole_enchilada(self) -> None:
        """Test reading the COLMAP hierarchy as a tree of (Path, GtsfmData)."""
        # TODO(frank): we need somehow to have images names and shapes if we want to save.
        # Add into GtsfmData?
        palace_metis = Path(__file__).resolve().parent.parent.parent / "palace_metis"
        print("Reading COLMAP tree from", palace_metis)
        tree = scene_tree.read_colmap(str(palace_metis), name="sparse")
        # tree = scene_tree.read_colmap(str(self.base_dir))
        assert tree is not None

        pseudo_color = False
        down_sample = True

        if down_sample:
            seed = 42
            fraction_points_to_keep = 0.1
            tree = scene_tree.downsample(tree, fraction_points_to_keep=fraction_points_to_keep, seed=seed)

        with open(self.base_dir / "cluster_tree.pkl", "rb") as f:
            cluster_tree: ClusterTree = pickle.load(f)
        print("Loaded cluster_tree from", self.base_dir / "cluster_tree.pkl")

        def reorder(vg_path_scene):
            vg, (path, scene) = vg_path_scene
            keys = sorted(visibility_graph_keys(vg))
            new_cameras = {k: scene.get_camera(i) for i, k in enumerate(keys)}
            new_data = GtsfmData(len(new_cameras), new_cameras, [])
            # HACK(Frank): don't remap tracks for now, just copy them over.
            new_data._tracks = scene.tracks().copy()
            if pseudo_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                print(f"\033[38;2;{r};{g};{b}mReordering scene at {path}, color=({r},{g},{b})\033[0m")
                for track in new_data.tracks():
                    track.r = r
                    track.g = g
                    track.b = b
            return path, new_data

        print(cluster_tree)
        random.seed(0)  # For reproducibility
        reordered_tree: SceneTree = Tree.zip(cluster_tree, tree).map(reorder)
        self.assertEqual(scene_tree.number_tracks(reordered_tree), scene_tree.number_tracks(tree))

        # Given a cluster tree with *local* VGGT results, we want to output a single GtsfmData that is
        # the merger of all the cluster results.
        merged_scene = merge_colmap_tree(reordered_tree)
        self.assertIsInstance(merged_scene, GtsfmData)
        print(merged_scene)

        # TODO(frank): these images names and shapes should be real.
        image_filenames = [f"San_Francisco_{i:04d}.jpg" for i in range(281)]  # Dummy filenames
        image_shapes = [(1936, 1296)] * len(image_filenames)  # Dummy shapes
        merged_scene.export_as_colmap_text(
            TEST_DATA_ROOT / "palace_merged_colmap" / "color-10%",
            image_filenames=image_filenames,
            image_shapes=image_shapes,
        )

        self.assertEqual(merged_scene.number_tracks(), scene_tree.number_tracks(reordered_tree))


if __name__ == "__main__":
    unittest.main()
