# Two View Estimator

![Alt text](gtsfm-overview-two-view-estimator.svg?raw=true)

- [Loader](assets/LOADER.md)
- [Image Pairs Generator](assets/IMAGE_PAIRS_GENERATOR.md)
- [Correspondence Generator](assets/CORRESPONDENCE_GENERATOR.md)
- **Two View Estimator**
- [Multiview Optimizer](assets/MULTIVIEW_OPTIMIZER.md)

## What is a Two-View Estimator?


Two-View Estimator (TVE) takes information about two images and tries to determine their relative pose (how one camera is positioned and oriented with respect to the other one). It also generates correspondences between keypoint in the images.

As seen in the diagram above, TVE sits between [Correspondence Generator](assets/CORRESPONDENCE_GENERATOR.md) and [Multiview Optimizer](assets/MULTIVIEW_OPTIMIZER.md). TVE does three things in GTSfM:  

1) It filters incorrect matches received from correspondence generator by removing outliers.

2) It provides initial "seed" relative pose estimates that are passed on to Multi View Optimizer (MVO) to start the global optimization process. A good initial estimate is important for MVO to be able to converge to a good solution.  

3) It also provides data to be used by view graph estimator to improve robustness in a global coordinate frame.


## Relevant Files:

- [gtsfm/two_view_estimator.py](https://github.com/borglab/gtsfm/blob/master/gtsfm/two_view_estimator.py): Core implementation of the `TwoViewEstimator` class, contains logic for two view estimation, bundle adjustment and evaluation. 
```python
class TwoViewEstimator:

"""Wrapper for running two-view relative pose estimation on image pairs in the dataset."""

def __init__(
self,
verifier: VerifierBase,
inlier_support_processor: InlierSupportProcessor,
bundle_adjust_2view: bool,
eval_threshold_px: float,
triangulation_options: TriangulationOptions,
bundle_adjust_2view_maxiters: int = 100,
ba_reproj_error_thresholds: List[Optional[float]] = [0.5],
allow_indeterminate_linear_system: bool = False,
) -> None:

"""Initializes the two-view estimator from verifier.

  

Args:

verifier: Verifier to use.

inlier_support_processor: Post-processor that uses information about RANSAC support to filter out pairs.

bundle_adjust_2view: Boolean flag indicating if bundle adjustment is to be run on the 2-view data.

eval_threshold_px: Distance threshold for marking a correspondence pair as inlier during evaluation

(not during estimation).

bundle_adjust_2view_maxiters (optional): Max number of iterations for 2-view BA. Defaults to 100.

ba_reproj_error_thresholds (optional): Reprojection thresholds used to filter features after each stage of

2-view BA. The length of this list decides the number of BA stages. Defaults to [0.5] (single stage).

allow_indeterminate_linear_system: Reject a two-view measurement if an indeterminate linear system is

encountered during marginal covariance computation after 2-view bundle adjustment.

"""
```

- [gtsfm/two_view_estimator_cacher.py](https://github.com/borglab/gtsfm/blob/master/gtsfm/two_view_estimator_cacher.py): Provides a caching implementation to speed up two view estimation process when the same image pairs is processed a second time.

- [gtsfm/common/two_view_estimation_report.py](https://github.com/borglab/gtsfm/blob/master/gtsfm/common/two_view_estimation_report.py): Defines the `TwoViewEstimationReport` dataclass used to store information and metrics about the two-view estimation result.

- [gtsfm/configs/* .yaml](https://github.com/borglab/gtsfm/blob/master/gtsfm/configs/): Configuration files that specify the parameters and components to be used for the two-view estimator.

 - [gtsfm/frontend/verifier/](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/verifier/): Verifiers (RANSAC, OpenCV, etc) used in `TwoViewEstimator` to improve the robustness of estimation.