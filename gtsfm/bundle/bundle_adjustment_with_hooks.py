"""Bundle adjustment wrapper with optional hook callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from gtsam import NonlinearFactorGraph, Values
from gtsam.symbol_shorthand import K, P, X  # type: ignore

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils.logger import get_logger

logger = get_logger()


@dataclass
class BundleAdjustmentHooks:
    """Optional callbacks to observe BA lifecycle events."""

    on_graph_constructed: Optional[Callable[[NonlinearFactorGraph], None]] = None
    on_before_optimization: Optional[Callable[[NonlinearFactorGraph, Values, bool, bool], None]] = None
    on_after_optimization: Optional[
        Callable[[NonlinearFactorGraph, Values, Values, Optional[List[Values]], bool, bool], None]
    ] = None


class BundleAdjustmentWithHooks(BundleAdjustmentOptimizer):
    """BundleAdjustmentOptimizer with external lifecycle hooks."""

    def __init__(
        self, save_iteration_visualization: bool = False, *args, hooks: Optional[BundleAdjustmentHooks] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._hooks = hooks
        self._save_iteration_visualization = save_iteration_visualization

    def run_simple_ba(self, initial_data: GtsfmData, verbose: bool = True) -> Tuple[GtsfmData, float]:
        cameras_to_model = sorted(initial_data.get_valid_camera_indices())
        graph, cameras_without_tracks = (
            self._BundleAdjustmentOptimizer__construct_simple_factor_graph(  # noqa: SLF001  # type: ignore[attr-defined]
                cameras_to_model, initial_data
            )
        )

        if self._hooks is not None and self._hooks.on_graph_constructed is not None:
            self._hooks.on_graph_constructed(graph)
        initial_values = initial_data.to_values(shared_calib=self._shared_calib)
        if self._hooks is not None and self._hooks.on_before_optimization is not None:
            self._hooks.on_before_optimization(
                graph,
                initial_values,
                verbose,
                self._save_iteration_visualization,
            )

        result_values, values_trace = (
            self._BundleAdjustmentOptimizer__optimize_factor_graph(  # noqa: SLF001  # type: ignore[attr-defined]
                graph,
                initial_values,
                self._ordering_type if not cameras_without_tracks else "COLAMD",
            )
        )
        final_error = graph.error(result_values)
        if verbose:
            logger.info("initial error: %.2f", graph.error(initial_values))
            logger.info("final error: %.2f", final_error)
        optimized_data = GtsfmData.from_values(result_values, initial_data, self._shared_calib)

        if self._hooks is not None and self._hooks.on_after_optimization is not None:
            self._hooks.on_after_optimization(
                graph,
                initial_values,
                result_values,
                values_trace,
                verbose,
                self._save_iteration_visualization,
            )

        final_T_i0 = result_values.atPose3(X(cameras_to_model[0]))
        init_camera_i0 = initial_data.get_camera(cameras_to_model[0])
        assert init_camera_i0 is not None
        init_T_i0 = init_camera_i0.pose()
        init_T_final = init_T_i0.compose(final_T_i0.inverse())
        transformed_values = Values()

        for c in cameras_to_model:
            optimized_camera = optimized_data.get_camera(c)
            assert optimized_camera is not None
            transformed_values.insert(X(c), init_T_final.compose(optimized_camera.pose()))
        for t in range(optimized_data.number_tracks()):
            transformed_values.insert(P(t), init_T_final.transformFrom(optimized_data.get_track(t).point3()))
        camera_ids = [cameras_to_model[0]] if self._shared_calib else cameras_to_model
        for c in camera_ids:
            optimized_camera = optimized_data.get_camera(c)
            assert optimized_camera is not None
            transformed_values.insert(
                K(
                    self._BundleAdjustmentOptimizer__map_to_calibration_variable(c)
                ),  # noqa: SLF001  # type: ignore[attr-defined]
                optimized_camera.calibration(),
            )
        for i, cam in cameras_without_tracks.items():
            optimized_data.add_camera(i, cam)
        return optimized_data, final_error
