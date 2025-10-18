"""Implements a base class for ViewGraph estimation.

Estimating the ViewGraph can be done trivially by adding all the two-view estimates into a ViewGraph data structure.
The purpose of this class, however, is to define an API for more sophisticated methods for estimating a ViewGraph
that include filtering or optimizing the two-view estimates.

Authors: Akshay Krishnan, Ayush Baid, John Lambert
"""

import abc
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from dask.delayed import Delayed, delayed
from gtsam import Cal3Bundler, Rot3, Unit3  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.products.visibility_graph import AnnotatedGraph, ImageIndexPairs
from gtsfm.two_view_estimator import TwoViewEstimationReport
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

PLOT_BASE_PATH = Path(__file__).resolve().parent.parent.parent / "plots"

# threshold for evaluation w.r.t. GT
MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0
METRIC_GROUP = "view_graph"

logger = logger_utils.get_logger()


class ViewGraphEstimatorBase(GTSFMProcess):
    """Base class for ViewGraph estimation.

    A ViewGraphEstimator aggregates two-view estimates into a ViewGraph.
    It could also improve the two-view estimates using filtering or optimization techniques.
    """

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="View-Graph Estimator",
            input_products=(
                "Optimized Relative Rotation",
                "Optimized Relative Translation",
                "Camera Intrinsics",
                "Inlier Correspondences",
                "Keypoints",
            ),
            output_products=(
                "View-Graph Relative Rotations",
                "View-Graph Relative Translations",
                "View-Graph Correspondences",
            ),
            parent_plate="Sparse Reconstruction",
        )

    @abc.abstractmethod
    def run(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: AnnotatedGraph[TwoViewEstimationReport],
    ) -> Set[Tuple[int, int]]:
        """Estimates the view graph, needs to be implemented by the derived class.

        The input rotation and unit translation dicts are guaranteed to be valid, i.e., i1 < i2 and
        neither i2Ri1 nor i2Ui1 are None.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints: keypoints for each images.
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.

        Returns:
            Edges of the view-graph, which are the subset of the image pairs in the input args.
        """

    def _get_valid_input_edges(
        self, i2Ri1_dict: Dict[Tuple[int, int], Rot3], i2Ui1_dict: Dict[Tuple[int, int], Unit3]
    ) -> ImageIndexPairs:
        """Gets the input edges (i1, i2):
        1. i1 < i2
        2. i2Ri1 and i2Ui1 are both not None.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to unit translation of i1 with respect to i2.

        Returns:
            List of valid edge indices.
        """
        valid_edges = []
        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            if i1 >= i2:
                logger.error("Incorrectly ordered edge indices found in cycle consistency for (%d, %d)", i1, i2)
                continue
            if i2Ri1 is None:
                continue  # edge was previously discarded for insufficient support
            if (i1, i2) not in i2Ui1_dict:
                logger.error("Found edge (%d, %d) in rotations dict but not in unit translations", i1, i2)
                continue
            if i2Ui1_dict[(i1, i2)] is None:
                continue
            valid_edges.append((i1, i2))

        return valid_edges

    def _filter_with_edges(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        two_view_reports: AnnotatedGraph[TwoViewEstimationReport],
        edges_to_select: Set[Tuple[int, int]],
    ) -> Tuple[
        Dict[Tuple[int, int], Rot3],
        Dict[Tuple[int, int], Unit3],
        Dict[Tuple[int, int], np.ndarray],
        AnnotatedGraph[TwoViewEstimationReport],
    ]:
        """Filters the dictionaries of 2-view results with the image-pair edges.

        Note: (key,value) pairs are preserved only if the key (i1,i2) corresponds to an edge
        that was deemed to be accurate with high probability by the ViewGraphEstimator.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.
            edges_to_select: edges to select (tuple of image pair indices).

        Returns:
            Subset of i2Ri1_dict.
            Subset of i2Ui1_dict
            Subset of corr_idxs_i1i2.
            Subset of two_view_reports.
        """
        return (
            {edge: i2Ri1_dict[edge] for edge in edges_to_select},
            {edge: i2Ui1_dict[edge] for edge in edges_to_select},
            {edge: corr_idxs_i1i2[edge] for edge in edges_to_select},
            {edge: two_view_reports[edge] for edge in edges_to_select},
        )

    def compute_metrics(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        two_view_reports: AnnotatedGraph[TwoViewEstimationReport],
        view_graph_edges: ImageIndexPairs,
        plots_output_dir: Path = PLOT_BASE_PATH,
    ) -> GtsfmMetricsGroup:
        """Metric computation for the view optimizer by selecting a subset of two-view reports for the pairs which
        are the edges of the view-graph. This can be overrode by implementations to define custom metrics.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.
            view_graph_edges: edges of the view-graph.

        Returns:
            Metrics for the view graph estimation, as a GtsfmMetricsGroup.
        """
        # pylint: disable=unused-argument

        # Case of missing ground truth.
        if len(two_view_reports) == 0:
            return GtsfmMetricsGroup(name="rotation_cycle_consistency_metrics", metrics=[])

        input_i1_i2 = i2Ri1_dict.keys()
        inlier_i1_i2 = view_graph_edges
        outlier_i1_i2 = list(set(input_i1_i2) - set(inlier_i1_i2))

        try:
            graph_utils.draw_view_graph_topology(
                edges=list(input_i1_i2),
                two_view_reports=two_view_reports,
                title="ViewGraphEstimator input",
                save_fpath=str(plots_output_dir / "view_graph_estimator_input_topology.jpg"),
                cameras_gt=None,
            )
            graph_utils.draw_view_graph_topology(
                edges=view_graph_edges,
                two_view_reports=two_view_reports,
                title="ViewGraphEstimator output",
                save_fpath=str(plots_output_dir / "view_graph_estimator_output_topology.jpg"),
                cameras_gt=None,
            )
        except Exception as e:
            # drawing the topology can fail in case of too many cameras
            logger.info(e)

        inlier_R_angular_errors = []
        outlier_R_angular_errors = []
        inlier_U_angular_errors = []
        outlier_U_angular_errors = []

        for (i1, i2), report in two_view_reports.items():
            if report is None:
                logger.error("TwoViewEstimationReport is None for ({}, {})".format(i1, i2))
            if report.R_error_deg is not None:
                if (i1, i2) in inlier_i1_i2:
                    inlier_R_angular_errors.append(report.R_error_deg)
                else:
                    outlier_R_angular_errors.append(report.R_error_deg)
            if report.U_error_deg is not None:
                if (i1, i2) in inlier_i1_i2:
                    inlier_U_angular_errors.append(report.U_error_deg)
                else:
                    outlier_U_angular_errors.append(report.U_error_deg)

        R_precision, R_recall = metrics_utils.get_precision_recall_from_errors(
            inlier_R_angular_errors, outlier_R_angular_errors, MAX_INLIER_MEASUREMENT_ERROR_DEG
        )

        U_precision, U_recall = metrics_utils.get_precision_recall_from_errors(
            inlier_U_angular_errors, outlier_U_angular_errors, MAX_INLIER_MEASUREMENT_ERROR_DEG
        )
        view_graph_metrics = [
            GtsfmMetric("num_input_measurements", len(input_i1_i2)),
            GtsfmMetric("num_inlier_measurements", len(inlier_i1_i2)),
            GtsfmMetric("num_outlier_measurements", len(outlier_i1_i2)),
            GtsfmMetric("R_precision", R_precision),
            GtsfmMetric("R_recall", R_recall),
            GtsfmMetric("U_precision", U_precision),
            GtsfmMetric("U_recall", U_recall),
            GtsfmMetric("inlier_R_angular_errors_deg", np.array(inlier_R_angular_errors)),
            GtsfmMetric("outlier_R_angular_errors_deg", np.array(outlier_R_angular_errors)),
            GtsfmMetric("inlier_U_angular_errors_deg", np.array(inlier_U_angular_errors)),
            GtsfmMetric("outlier_U_angular_errors_deg", np.array(outlier_U_angular_errors)),
        ]
        return GtsfmMetricsGroup("view_graph_estimation_metrics", view_graph_metrics)

    def create_computation_graph(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: Sequence[Optional[gtsfm_types.CALIBRATION_TYPE]],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        debug_output_dir: Optional[Path] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, Delayed, Delayed]:
        """Create the computation graph for ViewGraph estimation and metric evaluation.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints: keypoints for each image.
            two_view_reports: Dict from (i1, i2) to TwoViewEstimationReport that contains metrics.
            debug_output_dir: Path to directory where outputs for debugging will be saved.

        Returns:
            Tuple of the following 5 elements, all wrapped as Delayed:
            - Dict of i2Ri1 in the view graph
            - Dict of i2Ui1 in the view graph
            - Dict of corr_idxs_i1i2 in the view graph
            - Dict of two_view_reports in the view graph
            - GtsfmMetricsGroup with the view graph estimation metrics
        """

        # create debug directory for cycle_consistency
        plot_cycle_consist_path = None
        if debug_output_dir:
            plot_cycle_consist_path = debug_output_dir / "cycle_consistency"
            os.makedirs(plot_cycle_consist_path, exist_ok=True)

        # Remove all invalid edges in the input dicts.
        # TODO(Frank): This should be true by construction
        valid_edges = delayed(self._get_valid_input_edges)(
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
        )
        i2Ri1_valid_dict, i2Ui1_valid_dict, corr_idxs_i1i2_valid, two_view_reports_valid = delayed(
            self._filter_with_edges, nout=4
        )(
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            corr_idxs_i1i2=corr_idxs_i1i2,
            two_view_reports=two_view_reports,
            edges_to_select=valid_edges,
        )

        # Run view graph estimation.
        view_graph_edges = delayed(self.run)(
            i2Ri1_dict=i2Ri1_valid_dict,
            i2Ui1_dict=i2Ui1_valid_dict,
            calibrations=calibrations,
            corr_idxs_i1i2=corr_idxs_i1i2_valid,
            keypoints=keypoints,
            two_view_reports=two_view_reports_valid,
            output_dir=plot_cycle_consist_path,
        )

        # Remove all edges that are not in the view graph.
        i2Ri1_filtered, i2Ui1_filtered, corr_idxs_i1i2_filtered, two_view_reports_filtered = delayed(
            self._filter_with_edges, nout=4
        )(
            i2Ri1_dict=i2Ri1_valid_dict,
            i2Ui1_dict=i2Ui1_valid_dict,
            corr_idxs_i1i2=corr_idxs_i1i2_valid,
            two_view_reports=two_view_reports_valid,
            edges_to_select=view_graph_edges,
        )

        view_graph_estimation_metrics = delayed(self.compute_metrics)(
            i2Ri1_dict=i2Ri1_valid_dict,
            i2Ui1_dict=i2Ui1_valid_dict,
            calibrations=calibrations,
            two_view_reports=two_view_reports_valid,
            view_graph_edges=view_graph_edges,
            plots_output_dir=plot_cycle_consist_path,
        )

        return (
            i2Ri1_filtered,
            i2Ui1_filtered,
            corr_idxs_i1i2_filtered,
            two_view_reports_filtered,
            view_graph_estimation_metrics,
        )
