"""Run the Zhu et al. motion-averaging baseline from a verified COLMAP database."""

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import NamedTuple

import networkx as nx
import numpy as np
import pycolmap
from gtsam import Pose3, Rot3
from scipy.sparse import diags
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial.transform import Rotation

from gtsfm.averaging.distributed_motion_averaging import (
    MotionState,
    RelativeDisplacements,
    RelativeRotations,
    average_motions,
)
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.loader.yfcc_imb_loader import YfccImbLoader
from gtsfm.utils.align import sim3_from_Pose3_maps
from gtsfm.utils.metrics import compute_ba_pose_metrics


def normalized_cut(graph: nx.Graph, max_cameras: int) -> list[set[int]]:
    """Recursively split the inlier-weighted graph with a normalized-cut sweep.

    Args:
        graph: Connected camera graph with inlier counts as edge weights.
        max_cameras: Maximum size of a primary partition.

    Returns:
        Disjoint primary partitions. Boundary cameras are added during mapping.
    """
    if len(graph) <= max_cameras:
        return [set(graph)]
    ids = sorted(graph)
    weights = nx.to_scipy_sparse_array(graph, nodelist=ids, dtype=float, format="csr")
    degrees = np.asarray(weights.sum(axis=1))
    if np.any(degrees == 0):
        raise ValueError("Normalized cut requires a connected subgraph")
    normalized = laplacian(weights, normed=True).asformat("csr")
    values, vectors = eigsh(normalized, k=2, which="SM", v0=np.arange(1.0, len(ids) + 1))
    arrangement = np.argsort((diags(degrees**-0.5) @ vectors)[:, np.argsort(values)[1]], kind="stable")
    total = np.sum(degrees)
    volume, crossing = 0.0, 0.0
    in_left = np.zeros(len(ids), dtype=bool)
    best_score, best_split = np.inf, None
    for position, index in enumerate(arrangement, 1):
        start, stop = weights.indptr[index : index + 2]
        neighbors = weights.indices[start:stop]
        crossing += degrees[index] - 2 * weights.data[start:stop][in_left[neighbors]].sum()
        volume += degrees[index]
        in_left[index] = True
        if min(position, len(ids) - position) < 3:
            continue
        score = crossing * (1 / volume + 1 / (total - volume))
        if score < best_score:
            best_score, best_split = score, position
    if best_split is None:
        raise ValueError("The graph is too small for two incremental SfM partitions of three or more cameras")
    left = [ids[i] for i in arrangement[:best_split]]
    right = [ids[i] for i in arrangement[best_split:]]
    result = []
    for part in (left, right):
        for component in nx.connected_components(graph.subgraph(part)):
            if len(component) < 3:
                raise ValueError(f"Normalized cut left a component too small for local SfM: {sorted(component)}")
            result.extend(normalized_cut(graph.subgraph(component), max_cameras))
    return sorted(result, key=min)


def _camera_to_world(image: pycolmap.Image) -> Pose3:
    """Convert a registered COLMAP image to GTSfM world-from-camera convention.

    Args:
        image: Registered COLMAP image.

    Returns:
        Its camera-to-world pose.
    """
    pose = image.cam_from_world.inverse()
    return Pose3(Rot3(pose.rotation.matrix()), pose.translation)


def _map_partitions(
    database: Path, images: Path, output: Path, graph: nx.Graph, names: dict[int, str], groups: list[set[int]]
) -> list[pycolmap.Reconstruction]:
    """Seed each partition by incremental SfM with its adjacent separator cameras.

    Args:
        database: Shared verified frontend database.
        images: Source image directory.
        output: Run artifact directory.
        graph: Verified inlier-weighted view graph.
        names: COLMAP image identifiers to filenames.
        groups: Disjoint camera partitions.

    Returns:
        One incremental reconstruction per partition.
    """
    models = []
    for p, group in enumerate(groups):
        halo = set().union(*(set(graph[i]) for i in group))
        destination = output / f"local-{p}"
        destination.mkdir()
        options = pycolmap.IncrementalPipelineOptions(
            image_names={names[i] for i in group | halo}, multiple_models=False, extract_colors=False, num_threads=2
        )
        candidates = pycolmap.incremental_mapping(str(database), str(images), str(destination), options)
        if not candidates:
            raise RuntimeError(f"Incremental mapping failed for partition {p}")
        model = max(candidates.values(), key=lambda candidate: candidate.num_reg_images())
        missing = group - set(model.reg_image_ids())
        if missing:
            raise RuntimeError(f"Partition {p} lost primary cameras: {[names[i] for i in sorted(missing)]}")
        models.append(model)
    return models


class _Point(NamedTuple):
    """A local triangulated landmark supported by registered observations."""

    identifier: int
    position: np.ndarray
    observations: list[tuple[int, int]]


def _pick_points(model: pycolmap.Reconstruction, primary: set[int], target: int = 20) -> list[_Point]:
    """Greedily cover primary cameras with the widest available local tracks.

    Args:
        model: The partition's incremental SfM result.
        primary: Cameras owned by this partition.
        target: Target number of selected landmark observations per primary camera.

    Returns:
        Selected geometric constraints with their original 2D identities.
    """
    points = [
        _Point(k, point.xyz, [(e.image_id, e.point2D_idx) for e in point.track.elements])
        for k, point in model.points3D.items()
    ]
    counts = dict.fromkeys(primary, 0)
    selected = []
    for point in sorted(points, key=lambda item: (-len(item.observations), item.identifier)):
        if not any(i in counts and counts[i] < target for i, _ in point.observations):
            continue
        selected.append(point)
        for i, _ in point.observations:
            if i in counts:
                counts[i] += 1
        if min(counts.values()) >= target:
            break
    return selected


def _initial_similarities(poses: list[dict[int, Pose3]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align local seeds by common estimated cameras without reference poses.

    Args:
        poses: Independently estimated world-from-camera poses per partition.

    Returns:
        Partition rotation vectors, offsets, and scales in the first model's frame.
    """
    num_models = len(poses)
    rotation = np.zeros((num_models, 3))
    offset = np.zeros_like(rotation)
    scale = np.ones(num_models)
    world = poses[0].copy()
    unseen = set(range(1, num_models))
    while unseen:
        p = max(sorted(unseen), key=lambda k: len(poses[k].keys() & world.keys()))
        shared = sorted(poses[p].keys() & world.keys())
        if len(shared) < 3:
            raise RuntimeError(f"Partition {p} has fewer than three common cameras for similarity initialization")
        transform = sim3_from_Pose3_maps({i: world[i] for i in shared}, {i: poses[p][i] for i in shared})
        rotation[p] = Rotation.from_matrix(transform.rotation().matrix()).as_rotvec()
        scale[p] = transform.scale()
        # GTSAM Similarity3 represents s * (R * x + t).
        offset[p] = transform.transformFrom(np.zeros(3))
        if not np.isfinite(scale[p]) or scale[p] <= 0:
            raise RuntimeError(f"Partition {p} has invalid initial scale {scale[p]}")
        for i, wTi in poses[p].items():
            if i not in world:
                world[i] = transform.transformFrom(wTi)
        unseen.remove(p)
    return rotation, offset, scale


def _create_problem(
    graph: nx.Graph, groups: list[set[int]], models: list[pycolmap.Reconstruction]
) -> tuple[list[int], MotionState, RelativeRotations, RelativeDisplacements]:
    """Construct shared separators and scale-bearing measurements from local models.

    Args:
        graph: Verified camera-pair graph supplied to the baseline.
        groups: Primary camera partitions.
        models: Independently estimated local reconstructions.

    Returns:
        Camera IDs in solver order, initial state, rotations, and displacements.
    """
    camera_ids = sorted(graph)
    row_by_id = {i: row for row, i in enumerate(camera_ids)}
    partition_by_id = {i: p for p, group in enumerate(groups) for i in group}
    poses = [{i: _camera_to_world(model.image(i)) for i in model.reg_image_ids()} for model in models]
    rotation, offset, scale = _initial_similarities(poses)
    frames = Rotation.from_rotvec(rotation)
    separator_ids = {i for i in graph if any(partition_by_id[i] != partition_by_id[j] for j in graph[i])}
    camera_owner = np.array([-1 if i in separator_ids else partition_by_id[i] for i in camera_ids])
    entity_owner = list(camera_owner)
    camera_rotation = []
    positions = []
    for i in camera_ids:
        p = partition_by_id[i]
        local = poses[p][i]
        orientation = Rotation.from_matrix(local.rotation().matrix())
        center = local.translation()
        if i in separator_ids:
            orientation = frames[p] * orientation
            center = frames[p].apply(center) * scale[p] + offset[p]
        camera_rotation.append(orientation.as_rotvec())
        positions.append(center)

    selected = [_pick_points(model, group) for model, group in zip(models, groups)]
    observation_owner = {}
    tracks = nx.utils.UnionFind()
    for p, points in enumerate(selected):
        for point in points:
            key = (p, point.identifier)
            tracks[key]
            for observation in point.observations:
                other = observation_owner.setdefault(observation, key)
                tracks.union(other, key)
    points_by_key = {(p, point.identifier): point for p, points in enumerate(selected) for point in points}
    landmark_rows = {}
    for matched in tracks.to_sets():
        matched = sorted(matched)
        # A chain matching two distinct points in one model is ambiguous.
        valid = len({p for p, _ in matched}) == len(matched)
        for linked in [matched] if valid else [[key] for key in matched]:
            p, k = linked[0]
            point = points_by_key[(p, k)]
            crosses = len(linked) > 1 or any(i in separator_ids for i, _ in point.observations)
            row = len(positions)
            positions.append(frames[p].apply(point.position) * scale[p] + offset[p] if crosses else point.position)
            entity_owner.append(-1 if crosses else p)
            landmark_rows.update(dict.fromkeys(linked, row))

    rotation_pairs = []
    rotations = []
    displacement_pairs = []
    sources = []
    translations = []
    for p, group in enumerate(groups):
        local = poses[p]
        for i, j in sorted(tuple(sorted(edge)) for edge in graph.subgraph(local).edges):
            if i not in group and j not in group:
                continue
            iTj = local[i].between(local[j])
            pair = (row_by_id[i], row_by_id[j])
            rotation_pairs.append(pair)
            rotations.append(iTj.rotation().inverse().matrix())
            displacement_pairs.append(pair)
            sources.append(p)
            translations.append(iTj.translation())
        for point in selected[p]:
            for i, _ in point.observations:
                displacement_pairs.append((row_by_id[i], landmark_rows[(p, point.identifier)]))
                sources.append(p)
                translations.append(local[i].transformTo(point.position))

    retained_graph = nx.Graph()
    retained_graph.add_nodes_from(range(len(camera_ids)))
    retained_graph.add_edges_from(rotation_pairs)
    if not nx.is_connected(retained_graph):
        raise RuntimeError("Scale-bearing camera measurements do not connect all camera partitions")
    state = MotionState(
        camera_owner,
        np.array(entity_owner),
        np.array(camera_rotation),
        np.array(positions),
        rotation,
        offset,
        scale,
    )
    rotation_i, rotation_j = np.array(rotation_pairs).T
    displacement_i, displacement_j = np.array(displacement_pairs).T
    return (
        camera_ids,
        state,
        RelativeRotations(rotation_i, rotation_j, Rotation.from_matrix(rotations)),
        RelativeDisplacements(displacement_i, displacement_j, np.array(sources), np.array(translations)),
    )


def _prepare_frontend(source: Path, database: Path, min_inliers: int) -> tuple[dict[int, str], nx.Graph]:
    """Freeze verified inputs and load their largest connected view graph.

    Args:
        source: Original COLMAP database with keypoints and verified pairs.
        database: Path to the filtered snapshot for reconstruction.
        min_inliers: Minimum inlier count required to retain a verified pair.

    Returns:
        Camera names by database ID and the weighted view graph.
    """
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(database)) as snapshot:
            original.backup(snapshot)
            with snapshot:
                snapshot.execute("DELETE FROM two_view_geometries WHERE rows < ?", (min_inliers,))
    with pycolmap.Database(str(database)) as db:
        names = {image.image_id: image.name for image in db.read_all_images()}
        pair_ids, counts = db.read_two_view_geometry_num_inliers()
    graph = nx.Graph()
    graph.add_nodes_from(names)
    for pair_id, count in zip(pair_ids, counts):
        graph.add_edge(*pycolmap.Database.pair_id_to_image_pair(pair_id), weight=count)
    if graph.number_of_edges() == 0:
        raise ValueError("The frontend produced no usable verified pairs")
    largest = max(nx.connected_components(graph), key=lambda component: (len(component), -min(component)))
    return names, graph.subgraph(largest)


def _estimated_calibrations(
    groups: list[set[int]],
    models: list[pycolmap.Reconstruction],
) -> dict[int, pycolmap.Camera]:
    """Carry local intrinsics through to triangulation and final bundle adjustment.

    Args:
        groups: Primary image ids per partition.
        models: Incremental SfM estimates per partition.

    Returns:
        One fitted camera per database camera id. Ties favor the earliest partition.
    """
    calibrations = {}
    support = {}
    for group, model in zip(groups, models):
        local_support: dict[int, int] = {}
        for i in group:
            camera_id = model.image(i).camera_id
            local_support[camera_id] = local_support.get(camera_id, 0) + model.image(i).num_points3D
        for camera_id, count in local_support.items():
            if count > support.get(camera_id, -1):
                support[camera_id] = count
                calibrations[camera_id] = model.camera(camera_id)
    return calibrations


def _refine(
    state: MotionState,
    camera_ids: list[int],
    calibrations: dict[int, pycolmap.Camera],
    database: Path,
    images: Path,
    output: Path,
    fix_intrinsics: bool,
) -> tuple[pycolmap.Reconstruction, dict[int, Pose3]]:
    """Triangulate verified tracks and run final sparse BA with COLMAP.

    Args:
        state: Optimized motion averaging result.
        camera_ids: Database image ids in solver order.
        calibrations: Local estimates keyed by database camera id.
        database: The verified database shared with local mapping.
        images: Source image directory.
        output: Directory where the final COLMAP model will be written.
        fix_intrinsics: Hold the local calibration estimates fixed during final BA.

    Returns:
        Refined reconstruction and poses after triangulation but before final BA.
    """
    reconstruction = pycolmap.Reconstruction()
    rotations = state.world_rotations().as_matrix()
    locations = state.world_positions()
    with pycolmap.Database(str(database)) as db:
        for camera in calibrations.values():
            reconstruction.add_camera(camera)
        for row, i in enumerate(camera_ids):
            original = db.read_image(i)
            camera_from_world = rotations[row].T
            pose = pycolmap.Rigid3d(np.column_stack((camera_from_world, -camera_from_world @ locations[row])))
            image = pycolmap.Image(
                name=original.name,
                keypoints=db.read_keypoints(i)[:, :2].astype(float),
                cam_from_world=pose,
                camera_id=original.camera_id,
                id=i,
            )
            reconstruction.add_image(image)
            reconstruction.register_image(i)
    output.mkdir()
    options = pycolmap.IncrementalPipelineOptions(num_threads=2, extract_colors=False)
    reconstruction = pycolmap.triangulate_points(
        reconstruction, str(database), str(images), str(output), options=options
    )
    if reconstruction.num_points3D() == 0:
        raise RuntimeError("Triangulation produced no sparse points")
    triangulated = {i: _camera_to_world(reconstruction.image(i)) for i in reconstruction.reg_image_ids()}
    options = pycolmap.BundleAdjustmentOptions(
        refine_focal_length=not fix_intrinsics, refine_extra_params=not fix_intrinsics, print_summary=False
    )
    options.solver_options.num_threads = 2
    pycolmap.bundle_adjustment(reconstruction, options)
    reconstruction.write(str(output))
    return reconstruction, triangulated


def _score(reference: dict[int, Pose3], estimates: dict[int, Pose3]) -> dict[str, float]:
    """Compute the standard GTSfM pose AUC with a fixed reference denominator.

    Args:
        reference: Reference world-from-camera poses for this scene.
        estimates: Estimated world-from-camera poses keyed by database id.

    Returns:
        Pose AUC scalar values at GTSfM's published thresholds.
    """
    alignment = sim3_from_Pose3_maps(reference, estimates)
    group = compute_ba_pose_metrics(reference, {i: alignment.transformFrom(pose) for i, pose in estimates.items()})
    return {metric.name: float(metric.data) for metric in group.metrics if metric.name.startswith("pose_auc_")}


def _load_reference(names: dict[int, str], olsson: Path | None, yfcc_imb: Path | None) -> dict[int, Pose3]:
    """Match reference cameras to database images by the dataset's filename convention.

    Args:
        names: Image paths relative to the source images directory, by database ID.
        olsson: Optional root of the Olsson scene.
        yfcc_imb: Optional root of the Image Matching Benchmark scene.

    Returns:
        Reference poses keyed by database ID.
    """
    if olsson is not None:
        loader = OlssonLoader(str(olsson))
        ids = {str(PurePosixPath(name)): i for i, name in names.items()}
    elif yfcc_imb is not None:
        loader = YfccImbLoader(str(yfcc_imb), co_visibility_threshold=0.0)
        ids = {}
        for i, name in names.items():
            # IMB's visibility list stores paths without image extensions.
            key = str(PurePosixPath(name).with_suffix(""))
            if key in ids:
                raise ValueError(f"IMB reference name is ambiguous for database images: {key}")
            ids[key] = i
    else:
        return {}
    reference = {
        ids[key]: pose
        for i, name in enumerate(loader.image_filenames())
        if (key := str(PurePosixPath(name))) in ids and (pose := loader.get_camera_pose(i)) is not None
    }
    if len(reference) < 3:
        raise ValueError("Need at least three matching reference cameras for pose evaluation")
    return {i: reference[i] for i in sorted(reference)}


def main() -> None:
    """Run and score a sparse reconstruction from cached verified image pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    reference_options = parser.add_mutually_exclusive_group()
    reference_options.add_argument("--olsson-reference", type=Path)
    reference_options.add_argument("--yfcc-imb-reference", type=Path)
    parser.add_argument("--max-cameras", type=int, default=150)
    parser.add_argument("--min-pair-inliers", type=int, default=15)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--fix-final-intrinsics", action="store_true")
    args = parser.parse_args()
    if args.max_cameras < 3 or args.min_pair_inliers < 1 or args.max_rounds < 1:
        parser.error("Need at least three cameras per partition, one inlier per pair, and one round")
    if not args.database.is_file() or not args.images.is_dir():
        parser.error("The database and image directory must exist")
    args.output.mkdir(parents=True)
    started = perf_counter()
    pycolmap.set_random_seed(0)
    pycolmap.logging.minloglevel = pycolmap.logging.ERROR

    database = args.output / "frontend.db"
    names, graph = _prepare_frontend(args.database, database, args.min_pair_inliers)
    groups = normalized_cut(graph, args.max_cameras)
    if len(groups) < 2:
        parser.error("Choose --max-cameras below the scene size to exercise partitioned averaging")
    partitioned = perf_counter()
    models = _map_partitions(database, args.images, args.output, graph, names, groups)
    camera_ids, state, rotations, displacements = _create_problem(graph, groups, models)
    initialized = perf_counter()
    seed_rotations, seed_centers = state.world_rotations().as_matrix(), state.world_positions()
    trace = average_motions(state, rotations, displacements, max_rounds=args.max_rounds)
    averaged = perf_counter()
    final, triangulated = _refine(
        state,
        camera_ids,
        _estimated_calibrations(groups, models),
        database,
        args.images,
        args.output / "reconstruction",
        args.fix_final_intrinsics,
    )
    finished = perf_counter()

    # PyCOLMAP can update SQLite header counters while opening the database.
    with database.open("rb") as snapshot_file:
        frontend_sha256 = hashlib.file_digest(snapshot_file, "sha256").hexdigest()
    num_camera_pairs = len(rotations.first)
    summary = {
        "frontend_sha256": frontend_sha256,
        "input_cameras": len(names),
        "outside_largest_component": [names[i] for i in sorted(names.keys() - graph.nodes)],
        "verified_pairs": graph.number_of_edges(),
        "min_pair_inliers": args.min_pair_inliers,
        "camera_count": len(camera_ids),
        "separator_cameras": int((state.camera_owner < 0).sum()),
        "partitions": [[names[i] for i in sorted(group)] for group in groups],
        "scales": state.partition_scale.tolist(),
        "camera_relative_measurements": num_camera_pairs,
        "landmark_relative_measurements": len(displacements.first) - num_camera_pairs,
        "final_intrinsics_fixed": args.fix_final_intrinsics,
        "registered_after_refinement": final.num_reg_images(),
        "final_points": final.num_points3D(),
        "final_mean_reprojection_error_px": final.compute_mean_reprojection_error(),
        "stages_seconds": {
            "partitioning": partitioned - started,
            "local_reconstruction_and_initialization": initialized - partitioned,
            "motion_averaging": averaged - initialized,
            "triangulation_and_final_ba": finished - averaged,
        },
        "residuals": trace,
    }
    reference = _load_reference(names, args.olsson_reference, args.yfcc_imb_reference)
    if reference:
        averaged_poses = {
            i: Pose3(Rot3(r), t)
            for i, r, t in zip(camera_ids, state.world_rotations().as_matrix(), state.world_positions())
        }
        summary["reference_cameras"] = len(reference)
        summary["pose_auc_before_averaging"] = _score(
            reference, {i: Pose3(Rot3(r), t) for i, r, t in zip(camera_ids, seed_rotations, seed_centers)}
        )
        summary["pose_auc_after_averaging"] = _score(reference, averaged_poses)
        summary["pose_auc_after_triangulation"] = _score(reference, triangulated)
        summary["pose_auc_after_refinement"] = _score(
            reference, {i: _camera_to_world(final.image(i)) for i in final.reg_image_ids()}
        )
    result = json.dumps(summary, indent=2, allow_nan=False)
    (args.output / "metrics.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
