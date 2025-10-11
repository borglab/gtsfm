# GTSFM - Front End

This is the front end system for GTSFM

The front end accepts input images and produces geometrically verified feature point matches for each pair of input images. The complete output is encapsulated in a [`TwoViewResult`](../products/two_view_result.py) dataclass.

## TwoViewResult Lifecycle

📖 **[The Lifecycle of a TwoViewResult](two_view_result_lifecycle.md)** - Comprehensive documentation of how `TwoViewResult` objects are created, processed, and consumed throughout the GTSFM pipeline.

The `TwoViewResult` dataclass is defined in [`gtsfm.products.two_view_result`](../products/two_view_result.py) and contains:

- **Pose estimates**: Relative rotation (`i2Ri1`) and translation (`i2Ui1`) 
- **Correspondences**: Verified correspondence indices (`v_corr_idxs`)
- **Processing reports**: Metrics and statistics from each stage (pre-BA, post-BA, post-ISP)

This structured output enables downstream multi-view optimization components to access both the final estimates and intermediate results for quality assessment and debugging.

# Organization


+-- `detector`<br>
|      +-- `detector_base.py` (```DetectorBase```)<br>
|      +-- `detector_from_joint_detector_descriptor`<br>
+-- `descriptor`<br>
|      +-- `descriptor_base.py` (```DescriptorBase```)<br>
+-- `detector_descriptor`<br>
|      +-- `detector_descriptor_base.py` (```DetectorDescriptorBase```)<br>
|      +-- `combination_detector_descriptor.py`<br>
+-- `matcher`<br>
|      +-- `matcher_base.py` (```MatcherBase```)<br>
+-- `verifier`<br>
|      +-- `verified_base.py` (`VerifierBase`)<br>


## Detector+Descriptor 
Produces feature points (detection) and their associated descriptor vectors for an input image

There are two ways to define this stage of the front end:
- Implement a class which inherits from ```DetectorDescriptorBase```
- Combined individual detector (inherits ```DetectorBase```) and descriptor (inherits ```DescriptorBase```) using ```CombinationDetectorDescriptor```

## Matcher
Produces matched features for each pair of image:

To implement a matcher, inherit ```MatcherBase```.

## Verification
This is the final stage of the front end, and performs geometric verification of the matcher output to produce feature matches and the fundamental matrix for each pair of image.

To implement a verifier, inherit ```VerifierBase```.


