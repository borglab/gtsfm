from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class TwoViewBundleAdjustment(GTSFMProcess, BundleAdjustmentOptimizer):
    """Adds UI metadata for two-view bundle adjustment to BundleAdjustmentOptimizer."""

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Two-View Bundle Adjustment",
            input_products=(
                "Relative Rotation",
                "Relative Translation",
                "Triangulated Points",
                "Relative Pose Priors",
                "Camera Intrinsics",
                "Keypoints",
                "Verified Correspondences",
            ),
            output_products=("Optimized Relative Rotation", "Optimized Relative Translation", "Inlier Correspondences"),
            parent_plate="Two-View Estimator",
        )
