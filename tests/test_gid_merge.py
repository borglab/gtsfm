"""Unit tests for global-track-ID merge anchoring (gid-index sidecar).

Two clusters that triangulated the SAME global tracks from DISJOINT camera sets must be matchable by
global identity (the shared-camera matcher finds nothing there), and the MERGE_GUARD must drop a large
child whose ID-matched correspondences are too few to constrain its 7-DoF Sim3 seat.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmTrack

from gtsfm.cluster_merging import (
    _SCENE_GID_INDEX_ATTR,
    _lookup_gid,
    _select_gid_point_correspondences,
    _track_gid,
    _union_gid_indices,
    annotate_scene_with_metadata,
    build_measurement_gid_arrays,
    merge_scenes_with_sim3_nonlinear,
    slice_gid_index,
)
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.scene_optimizer import _dlt_triangulate, _ransac_dlt_resect

NUM_IMAGES = 64
PARENT_CAMS = [0, 1, 2]
CHILD_CAMS = [10, 11, 12]  # DISJOINT from parent — no shared cameras anywhere.


def _uv(gid: int, cam: int) -> np.ndarray:
    """Deterministic per-(track, camera) pixel; same values used for global tracks and scene tracks."""
    return np.array([13.0 * gid + cam + 0.25, 7.0 * gid + 2 * cam + 0.75])


def _global_tracks(num: int) -> list[SfmTrack2d]:
    """Global 2D tracks spanning BOTH camera sets (the cross-cluster edges made them)."""
    tracks = []
    for gid in range(num):
        meas = [SfmMeasurement(c, _uv(gid, c)) for c in PARENT_CAMS + CHILD_CAMS]
        tracks.append(SfmTrack2d(measurements=meas))
    return tracks


def _scene(cams: list[int], gids: list[int], point_offset: float) -> GtsfmData:
    """A reconstruction over ``cams`` containing one 3D track per gid (measurements = the global uvs)."""
    data = GtsfmData(number_images=NUM_IMAGES)
    for k, c in enumerate(cams):
        pose = Pose3(Rot3(), np.array([k, 0.0, 0.0]))
        data.add_camera(c, PinholeCameraCal3Bundler(pose, Cal3Bundler()))
    for gid in gids:
        track = SfmTrack(Point3(float(gid), point_offset, 0.0))
        for c in cams:
            track.addMeasurement(c, _uv(gid, c))
        assert data.add_track(track)
    return data


class TestGidIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks_2d = _global_tracks(60)
        self.arrays = build_measurement_gid_arrays(self.tracks_2d)

    def test_pack_slice_lookup_roundtrip(self) -> None:
        """Every measurement resolves to its gid through a camera-restricted slice."""
        parent_index = slice_gid_index(*self.arrays, set(PARENT_CAMS))
        child_index = slice_gid_index(*self.arrays, set(CHILD_CAMS))
        self.assertIsNotNone(parent_index)
        for gid in (0, 7, 59):
            self.assertEqual(_lookup_gid(parent_index, 0, _uv(gid, 0)), gid)
            self.assertEqual(_lookup_gid(child_index, 11, _uv(gid, 11)), gid)
        # A camera outside the slice must not resolve.
        self.assertEqual(_lookup_gid(parent_index, 10, _uv(3, 10)), -1)

    def test_correspondences_across_disjoint_cameras(self) -> None:
        """Same global tracks triangulated from disjoint camera sets match by identity."""
        parent = _scene(PARENT_CAMS, list(range(60)), point_offset=0.0)
        child = _scene(CHILD_CAMS, list(range(60)), point_offset=100.0)
        parent_index = slice_gid_index(*self.arrays, set(PARENT_CAMS))
        child_index = slice_gid_index(*self.arrays, set(CHILD_CAMS))

        pairs = _select_gid_point_correspondences(parent, child, parent_index, child_index, max_correspondences=100)
        self.assertEqual(len(pairs), 60)
        # Pairs must join the SAME physical point: x-coords encode the gid on both sides.
        for p_pt, c_pt in pairs:
            self.assertAlmostEqual(p_pt[0], c_pt[0])
            self.assertAlmostEqual(p_pt[1] + 100.0, c_pt[1])

    def test_track_gid_and_union(self) -> None:
        parent_index = slice_gid_index(*self.arrays, set(PARENT_CAMS))
        child_index = slice_gid_index(*self.arrays, set(CHILD_CAMS))
        union = _union_gid_indices(parent_index, child_index)
        parent = _scene(PARENT_CAMS, [5], point_offset=0.0)
        child = _scene(CHILD_CAMS, [5], point_offset=100.0)
        # Both scenes' copies of global track 5 resolve to the same gid through the union index.
        self.assertEqual(_track_gid(parent.get_track(0), union), 5)
        self.assertEqual(_track_gid(child.get_track(0), union), 5)

    def test_track_gid_majority_vote(self) -> None:
        """A track whose FIRST measurement's pixel bin collides to a different gid still resolves to the
        majority gid (the old first-hit identity returned the collided gid -> phantom correspondences)."""
        tracks_2d = [
            # gid 0: a single measurement in camera 0 whose bin (10, 20) the test track will collide into.
            SfmTrack2d(measurements=[SfmMeasurement(0, np.array([10.25, 20.75]))]),
            # gid 1: measurements in cameras 1 and 2 — the test track's TRUE identity.
            SfmTrack2d(
                measurements=[
                    SfmMeasurement(1, np.array([30.25, 40.75])),
                    SfmMeasurement(2, np.array([50.25, 60.75])),
                ]
            ),
        ]
        index = slice_gid_index(*build_measurement_gid_arrays(tracks_2d), {0, 1, 2})

        track = SfmTrack(Point3(0.0, 0.0, 0.0))
        track.addMeasurement(0, np.array([10.9, 20.1]))  # floors to bin (10, 20) -> collides to gid 0
        track.addMeasurement(1, np.array([30.5, 40.9]))  # -> gid 1
        track.addMeasurement(2, np.array([50.7, 60.2]))  # -> gid 1
        # Majority (2 votes gid 1 vs 1 collided vote gid 0) wins; first-hit would have returned 0.
        self.assertEqual(_track_gid(track, index), 1)

        # And a track with no indexed measurements still resolves to -1.
        unknown = SfmTrack(Point3(0.0, 0.0, 0.0))
        unknown.addMeasurement(0, np.array([9999.5, 9999.5]))
        self.assertEqual(_track_gid(unknown, index), -1)

    def test_merge_guard_drops_large_weak_child(self) -> None:
        """A >=30-camera child with too few ID correspondences is dropped (returns parent unchanged)."""
        parent = _scene(PARENT_CAMS, list(range(60)), point_offset=0.0)
        big_cams = list(range(20, 55))  # 35 cameras, disjoint from parent
        # Child shares only 5 global tracks with the parent -> 5 ID correspondences << 50 floor.
        child = GtsfmData(number_images=NUM_IMAGES)
        for k, c in enumerate(big_cams):
            child.add_camera(c, PinholeCameraCal3Bundler(Pose3(Rot3(), np.array([k, 0.0, 0.0])), Cal3Bundler()))
        for gid in range(5):
            track = SfmTrack(Point3(float(gid), 100.0, 0.0))
            for c in big_cams[:3]:
                track.addMeasurement(c, _uv(gid, c))
            self.assertTrue(child.add_track(track))

        # Global tracks for the child's cameras exist too (so its index resolves).
        tracks_2d = _global_tracks(60)
        for gid in range(5):
            for c in big_cams[:3]:
                tracks_2d[gid].measurements.append(SfmMeasurement(c, _uv(gid, c)))
        arrays = build_measurement_gid_arrays(tracks_2d)
        annotate_scene_with_metadata(parent, None, None, slice_gid_index(*arrays, set(PARENT_CAMS)))
        annotate_scene_with_metadata(child, None, None, slice_gid_index(*arrays, set(big_cams)))

        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        # Guard dropped the only child -> parent returned as-is.
        self.assertEqual(merged.number_tracks(), parent.number_tracks())
        self.assertEqual(set(merged.get_valid_camera_indices()), set(PARENT_CAMS))

    def test_no_sidecar_runs_legacy_path(self) -> None:
        """Without gid sidecars (enable_gid_merge_anchoring=False), the merge is the R3-baseline legacy
        path: shared-camera matching, no corr-floor guard — a large low-corr child is KEPT, not dropped."""
        shared = list(range(30, 46))
        parent = _scene(PARENT_CAMS + shared, list(range(30)), point_offset=0.0)
        child_cams = shared + list(range(46, 62))  # 32 cams, low legacy correspondences
        child = _scene(child_cams, list(range(60)), point_offset=0.0)
        # No annotate_scene_with_metadata -> no sidecars -> id_mode False everywhere.
        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        self.assertTrue(set(child_cams).issubset(set(merged.get_valid_camera_indices())))

    def test_structureless_child_dropped_any_size(self) -> None:
        """A 0-track child is never merged, even with shared cameras (nothing can anchor or verify it)."""
        parent = _scene(PARENT_CAMS, list(range(20)), point_offset=0.0)
        child = GtsfmData(number_images=NUM_IMAGES)
        for k, c in enumerate(PARENT_CAMS + [20, 21]):  # shares ALL parent cameras, but zero tracks
            child.add_camera(c, PinholeCameraCal3Bundler(Pose3(Rot3(), np.array([k, 0.0, 0.0])), Cal3Bundler()))
        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        self.assertEqual(set(merged.get_valid_camera_indices()), set(PARENT_CAMS))

    def test_overlap_escape_keeps_large_child_with_strong_camera_anchor(self) -> None:
        """A large child with >=15 shared cameras is KEPT despite low ID-corr (measured: 21-shared seats fine).

        Reproduces the gid-run regression where a 50-cam child with 21 shared cameras and 25 correspondences
        was dropped, costing Nc with no accuracy gain.
        """
        shared = list(range(30, 46))  # 16 shared cameras
        parent = _scene(PARENT_CAMS + shared, list(range(30)), point_offset=0.0)
        child_cams = shared + list(range(46, 62))  # 32 cams total, 16 shared
        child = _scene(child_cams, list(range(5)), point_offset=0.0)  # only 5 MATCHING tracks (< 50 corr floor)
        # Give the child real structure (>=50 tracks for the escape's structure floor) whose measurements
        # do NOT resolve in the gid index (offset uvs) -> correspondences stay at 5.
        for extra in range(60):
            track = SfmTrack(Point3(float(extra), -50.0, 5.0))
            for c in child_cams[:4]:
                track.addMeasurement(c, _uv(extra, c) + np.array([5000.0, 5000.0]))
            self.assertTrue(child.add_track(track))

        # Global tracks must SPAN this test's cameras so both slices resolve (id_mode active).
        tracks_2d = []
        for gid in range(30):
            meas = [SfmMeasurement(c, _uv(gid, c)) for c in PARENT_CAMS + shared + list(range(46, 62))]
            tracks_2d.append(SfmTrack2d(measurements=meas))
        arrays = build_measurement_gid_arrays(tracks_2d)
        annotate_scene_with_metadata(parent, None, None, slice_gid_index(*arrays, set(PARENT_CAMS + shared)))
        annotate_scene_with_metadata(child, None, None, slice_gid_index(*arrays, set(child_cams)))

        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        # Escape clause: child kept (its cameras present in the merged scene).
        self.assertTrue(set(child_cams).issubset(set(merged.get_valid_camera_indices())))


class TestBoundaryRecoveryGeometry(unittest.TestCase):
    """Unit tests for the boundary-recovery DLT helpers (ports of the offline-validated exp5 recipe)."""

    def test_dlt_triangulate_recovers_known_point(self) -> None:
        """3 posed cameras observing a known 3D point: DLT recovers it to <1e-6 with ~0 reprojection."""
        f, px, py = 500.0, 320.0, 240.0
        x_true = np.array([0.5, 0.5, 10.0])
        centers = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        posed, intrinsics, measurements = {}, {}, []
        for i, center in enumerate(centers):
            r_w2c = np.eye(3)
            t_w2c = -r_w2c @ center
            posed[i] = (r_w2c, t_w2c)
            intrinsics[i] = (f, 0.0, 0.0, px, py)
            p = r_w2c @ x_true + t_w2c
            measurements.append((i, p[0] / p[2] * f + px, p[1] / p[2] * f + py))

        x_rec, err = _dlt_triangulate(measurements, posed, intrinsics)
        self.assertIsNotNone(x_rec)
        np.testing.assert_allclose(x_rec, x_true, atol=1e-6)
        self.assertLess(err, 1e-6)

    def test_ransac_dlt_resect_recovers_known_camera(self) -> None:
        """Noise-free observations of 60 known 3D points: resection recovers the exact w2c pose."""
        rng = np.random.default_rng(42)
        struct = {g: rng.uniform([-2.0, -2.0, 4.0], [2.0, 2.0, 8.0]) for g in range(60)}
        theta = 0.3
        r_true = np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ]
        )
        c_true = np.array([0.5, -0.3, -2.0])
        t_true = -r_true @ c_true
        f, px, py = 600.0, 400.0, 300.0
        observations = []
        for g, point in struct.items():
            p = r_true @ point + t_true
            observations.append((g, p[0] / p[2] * f + px, p[1] / p[2] * f + py))

        result = _ransac_dlt_resect(observations, (f, 0.0, 0.0, px, py), struct)
        self.assertIsNotNone(result)
        r_rec, t_rec, c_rec, num_inliers = result
        self.assertEqual(num_inliers, 60)
        np.testing.assert_allclose(r_rec, r_true, atol=1e-6)
        np.testing.assert_allclose(t_rec, t_true, atol=1e-6)
        np.testing.assert_allclose(c_rec, c_true, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
