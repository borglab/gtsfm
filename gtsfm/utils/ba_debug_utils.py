"""Debug/visualization hooks for BundleAdjustmentOptimizer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, cast

import gtsam
import numpy as np
from gtsam import NonlinearFactorGraph, Values

from gtsfm.bundle.bundle_adjustment_with_hooks import BundleAdjustmentHooks
from gtsfm.utils.logger import get_logger

logger = get_logger()


def _factor_at(graph: NonlinearFactorGraph, index: int):
    try:
        return graph.at(index)
    except AttributeError:
        return graph[index]


def _format_key(key: int) -> str:
    try:
        return cast(str, gtsam.Symbol(key).string())
    except Exception:
        return str(key)


def _write_graph_print(graph: NonlinearFactorGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        output_fd = output_file.fileno()
        stdout_fd = os.dup(1)
        try:
            os.dup2(output_fd, 1)
            graph.print("NonlinearFactorGraph")
        finally:
            os.dup2(stdout_fd, 1)
            os.close(stdout_fd)


def _write_factor_errors(graph: NonlinearFactorGraph, values: Values, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        output_fd = output_file.fileno()
        stdout_fd = os.dup(1)
        try:
            os.dup2(output_fd, 1)
            graph.printErrors(values)
        finally:
            os.dup2(stdout_fd, 1)
            os.close(stdout_fd)


def _write_factor_error_stats(graph: NonlinearFactorGraph, values: Values, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [1e3, 1e4]
    total_factors = int(graph.size())
    errors: list[tuple[int, float, list[str]]] = []
    key_counts_by_threshold: dict[float, dict[str, int]] = {t: {} for t in thresholds}
    key_type_counts_by_threshold: dict[float, dict[str, int]] = {t: {} for t in thresholds}
    key_error_sums: dict[str, float] = {}

    for i in range(total_factors):
        factor = _factor_at(graph, i)
        if factor is None:
            continue
        try:
            error = float(factor.error(values))
        except RuntimeError:
            continue
        try:
            keys = [int(k) for k in factor.keys()]
        except Exception:
            keys = []
        key_strs = [_format_key(k) for k in keys]
        errors.append((i, error, key_strs))
        for key_str in key_strs:
            key_error_sums[key_str] = key_error_sums.get(key_str, 0.0) + error
        for threshold in thresholds:
            if error >= threshold:
                for key_str in key_strs:
                    key_counts_by_threshold[threshold][key_str] = (
                        key_counts_by_threshold[threshold].get(key_str, 0) + 1
                    )
                    key_type = key_str[0] if key_str else "?"
                    key_type_counts_by_threshold[threshold][key_type] = (
                        key_type_counts_by_threshold[threshold].get(key_type, 0) + 1
                    )

    errors_sorted = sorted(errors, key=lambda item: item[1], reverse=True)
    with output_path.open("w") as output_file:
        output_file.write(f"total_factors: {total_factors}\n")
        if not errors_sorted:
            output_file.write("no_errors_recorded\n")
            return

        all_errors = [e for _, e, _ in errors_sorted]
        output_file.write(f"min_error: {min(all_errors):.6f}\n")
        output_file.write(f"max_error: {max(all_errors):.6f}\n")
        output_file.write(f"median_error: {np.median(all_errors):.6f}\n")
        output_file.write(f"mean_error: {np.mean(all_errors):.6f}\n")
        output_file.write(f"p95_error: {np.percentile(all_errors, 95):.6f}\n")
        output_file.write(f"p99_error: {np.percentile(all_errors, 99):.6f}\n")

        for threshold in thresholds:
            count = sum(1 for _, e, _ in errors_sorted if e >= threshold)
            fraction = count / total_factors if total_factors else 0.0
            output_file.write(f"\nerrors_ge_{int(threshold)}: {count} ({fraction:.4%})\n")
            key_counts = key_counts_by_threshold[threshold]
            key_type_counts = key_type_counts_by_threshold[threshold]
            if key_type_counts:
                output_file.write("key_type_counts:\n")
                for key_type, kt_count in sorted(key_type_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
                    output_file.write(f"  {key_type}: {kt_count}\n")
            if key_counts:
                output_file.write("top_keys_by_category:\n")
                for category in ["x", "p", "k", "?"]:
                    category_keys = {
                        key_str: key_count
                        for key_str, key_count in key_counts.items()
                        if (key_str[0] if key_str else "?") == category
                    }
                    if not category_keys:
                        continue
                    output_file.write(f"  category {category}:\n")
                    for key_str, key_count in sorted(category_keys.items(), key=lambda item: item[1], reverse=True)[:20]:
                        output_file.write(f"    {key_str}: {key_count}\n")

        if key_error_sums:
            output_file.write("\nkey_error_sums_by_category:\n")
            for category in ["x", "p", "k", "?"]:
                category_sums = {
                    key_str: total_error
                    for key_str, total_error in key_error_sums.items()
                    if (key_str[0] if key_str else "?") == category
                }
                if not category_sums:
                    continue
                output_file.write(f"  category {category}:\n")
                for key_str, total_error in sorted(category_sums.items(), key=lambda item: item[1], reverse=True)[:20]:
                    output_file.write(f"    {key_str}: {total_error:.6f}\n")

        output_file.write("\nworst_factors:\n")
        for factor_index, error, key_strs in errors_sorted[:50]:
            key_list = ", ".join(key_strs)
            output_file.write(f"  factor {factor_index}: error={error:.6f} keys={{ {key_list} }}\n")


def _write_factor_error_visualizations(
    graph: NonlinearFactorGraph, values: Values, camera_output_path: Path, point_output_path: Path
) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
        import plotly.io as pio  # type: ignore
    except Exception:
        logger.warning("Plotly not available, skipping factor error visualizations.")
        return

    camera_output_path.parent.mkdir(parents=True, exist_ok=True)
    point_output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_error_lists: dict[int, list[float]] = {}
    point_error_lists: dict[int, list[float]] = {}

    for i in range(int(graph.size())):
        factor = _factor_at(graph, i)
        if factor is None:
            continue
        try:
            error = float(factor.error(values))
        except RuntimeError:
            continue
        try:
            keys = [int(k) for k in factor.keys()]
        except Exception:
            keys = []
        for key in keys:
            key_str = _format_key(key)
            key_type = key_str[0] if key_str else "?"
            if key_type == "x":
                camera_error_lists.setdefault(key, []).append(error)
            elif key_type == "p":
                point_error_lists.setdefault(key, []).append(error)

    if camera_error_lists:
        camera_means = {key: float(np.mean(errors)) for key, errors in camera_error_lists.items()}
        cam_keys = [_format_key(key) for key in camera_means]
        cam_errors = [camera_means[key] for key in camera_means]
        fig = go.Figure(data=[go.Bar(x=cam_keys, y=cam_errors)])
        fig.update_layout(
            title="Camera error means by key",
            xaxis_title="Camera key",
            yaxis_title="Mean factor error",
            xaxis_tickangle=45,
            margin=dict(l=40, r=20, t=40, b=120),
        )
        pio.write_html(fig, file=str(camera_output_path), auto_open=False)

    if point_error_lists:
        points_xyz = []
        point_errors = []
        point_labels = []
        point_means = {key: float(np.mean(errors)) for key, errors in point_error_lists.items()}
        for key, error_mean in point_means.items():
            try:
                point = values.atPoint3(key)
            except Exception:
                continue
            points_xyz.append(point)
            point_errors.append(error_mean)
            point_labels.append(_format_key(key))

        if points_xyz:
            xyz = np.asarray(points_xyz)
            errors_np = np.asarray(point_errors, dtype=float)
            percentiles = [50, 60, 70, 80, 90, 95, 98, 99]
            clip_values = {p: float(np.percentile(errors_np, p)) for p in percentiles}

            def _clipped_colors(clip_max: float):
                return np.minimum(errors_np, clip_max)

            initial_percentile = 80
            initial_clip = clip_values[initial_percentile]
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=xyz[:, 0],
                        y=xyz[:, 1],
                        z=xyz[:, 2],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=_clipped_colors(initial_clip),
                            colorscale="Viridis",
                            opacity=0.8,
                            colorbar=dict(title="Mean factor error"),
                        ),
                        text=point_labels,
                    )
                ]
            )

            steps = []
            for percentile in percentiles:
                clip_max = clip_values[percentile]
                steps.append(
                    dict(
                        method="update",
                        label=f"p{percentile}",
                        args=[
                            {"marker.color": [_clipped_colors(clip_max)]},
                            {"title": f"Point error colormap (clipped at p{percentile}={clip_max:.3f})"},
                        ],
                    )
                )

            fig.update_layout(
                title=f"Point error colormap (mean errors, clipped at p{initial_percentile}={initial_clip:.3f})",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                margin=dict(l=0, r=0, t=40, b=0),
                sliders=[
                    dict(
                        active=percentiles.index(initial_percentile),
                        currentvalue={"prefix": "Clip: "},
                        pad={"t": 30},
                        steps=steps,
                    )
                ],
            )
            pio.write_html(fig, file=str(point_output_path), auto_open=False)


def _write_iteration_visualization(values_trace: list[Values], initial_values: Values, output_path: Path) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
        import plotly.io as pio  # type: ignore
        import visu3d as v3d  # type: ignore
    except Exception:
        logger.warning("Plotly or visu3d not available, skipping optimization visualization.")
        return

    if not values_trace:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _symbol_char(key: int) -> str:
        try:
            return chr(gtsam.Symbol(key).chr())
        except Exception:
            return "?"

    def _symbol_index(key: int) -> int:
        try:
            return int(gtsam.Symbol(key).index())
        except Exception:
            return -1

    def _point_keys(values: Values) -> list[int]:
        return [int(k) for k in values.keys() if _symbol_char(int(k)) == "p"]

    def _camera_keys(values: Values) -> list[int]:
        keys = [int(k) for k in values.keys() if _symbol_char(int(k)) == "x"]
        return sorted(keys, key=_symbol_index)

    point_keys = _point_keys(initial_values)
    camera_keys = _camera_keys(initial_values)
    if not camera_keys:
        logger.warning("No camera keys found for optimization visualization.")
        return

    max_points = 20000
    if len(point_keys) > max_points:
        indices = np.linspace(0, len(point_keys) - 1, max_points, dtype=int)
        point_keys = [point_keys[i] for i in indices]

    def _pose_to_v3d_matrix(pose: gtsam.Pose3):
        matrix = np.concatenate([0.1 * pose.rotation().matrix(), pose.translation()[:, None]], axis=-1)
        matrix = np.concatenate([matrix, np.array([[0, 0, 0, 1]], dtype=np.float64)], axis=0)
        return matrix.astype(np.float32)

    def _extract_frame(values: Values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = []
        for key in point_keys:
            try:
                points.append(values.atPoint3(key))
            except Exception:
                points.append([np.nan, np.nan, np.nan])
        camera_matrices = []
        for key in camera_keys:
            try:
                pose = values.atPose3(key)
                camera_matrices.append(_pose_to_v3d_matrix(pose))
            except Exception:
                camera_matrices.append(np.full((4, 4), np.nan, dtype=np.float32))
        first_camera_matrix = camera_matrices[0:1]
        return (
            np.asarray(points, dtype=float),
            np.asarray(camera_matrices, dtype=np.float32),
            np.asarray(first_camera_matrix, dtype=np.float32),
        )

    initial_points, initial_camera_matrices, initial_first_camera = _extract_frame(values_trace[0])
    camera_traces = v3d.make_fig(
        [
            v3d.Transform.from_matrix(initial_camera_matrices),
            v3d.Transform.from_matrix(initial_first_camera),
        ]
    )
    fig = go.Figure(data=list(camera_traces.data))
    fig.add_trace(
        go.Scatter3d(
            x=initial_points[:, 0],
            y=initial_points[:, 1],
            z=initial_points[:, 2],
            mode="markers",
            marker=dict(size=2, color="rgba(31, 119, 180, 0.35)"),
            name="Points",
        )
    )

    frames = []
    for idx, values in enumerate(values_trace):
        points, camera_matrices, first_camera_matrix = _extract_frame(values)
        camera_frame = v3d.make_fig(
            [
                v3d.Transform.from_matrix(camera_matrices),
                v3d.Transform.from_matrix(first_camera_matrix),
            ]
        )
        frame_traces = list(camera_frame.data)
        frame_traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2, color="rgba(31, 119, 180, 0.35)"),
                name="Points",
            )
        )
        frames.append(go.Frame(data=frame_traces, name=str(idx)))

    fig.frames = frames
    fig.update_layout(
        title="BA optimization progress (points + cameras)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.05,
                x=1.0,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 200, "redraw": False}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Iter: "},
                pad={"t": 30},
                steps=[
                    dict(method="animate", label=str(i), args=[[str(i)], {"mode": "immediate"}])
                    for i in range(len(values_trace))
                ],
            )
        ],
    )
    pio.write_html(fig, file=str(output_path), auto_open=False)


def build_ba_debug_hooks(
    metrics_dir: Path,
    *,
    save_factor_graph: bool,
    save_iteration_visualization: bool,
    factor_graph_filename: str = "factor_graph.txt",
) -> BundleAdjustmentHooks:
    metrics_dir.mkdir(parents=True, exist_ok=True)

    def _on_graph_constructed(graph: NonlinearFactorGraph) -> None:
        if save_factor_graph:
            _write_graph_print(graph, metrics_dir / factor_graph_filename)

    def _on_before(graph: NonlinearFactorGraph, initial_values: Values, verbose: bool, save_iter: bool) -> None:
        if not verbose:
            return
        _write_factor_errors(graph, initial_values, metrics_dir / "ba_factor_errors_initial.txt")
        _write_factor_error_stats(graph, initial_values, metrics_dir / "ba_factor_error_stats_initial.txt")
        _write_factor_error_visualizations(
            graph,
            initial_values,
            metrics_dir / "ba_camera_error_initial.html",
            metrics_dir / "ba_point_error_colormap_initial.html",
        )

    def _on_after(
        graph: NonlinearFactorGraph,
        initial_values: Values,
        result_values: Values,
        values_trace: Optional[list[Values]],
        verbose: bool,
        save_iter: bool,
    ) -> None:
        if not verbose:
            return
        _write_factor_errors(graph, result_values, metrics_dir / "ba_factor_errors_final.txt")
        _write_factor_error_stats(graph, result_values, metrics_dir / "ba_factor_error_stats_final.txt")
        _write_factor_error_visualizations(
            graph,
            result_values,
            metrics_dir / "ba_camera_error_final.html",
            metrics_dir / "ba_point_error_colormap_final.html",
        )
        if save_iteration_visualization and save_iter and values_trace is not None:
            _write_iteration_visualization(values_trace, initial_values, metrics_dir / "ba_optimization_progress.html")

    return BundleAdjustmentHooks(
        on_graph_constructed=_on_graph_constructed,
        on_before_optimization=_on_before,
        on_after_optimization=_on_after,
    )
