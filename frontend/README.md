# GTSFM - Front End

This is the front end system for GTSFM

The front end accepts input images and produces geometrically verified feature point matches for each pair of input images.

# Organization

+-- `detector`
|   +-- `detector_base.py` (```DetectorBase```)
|   +-- `detector_from_joint_detector_descriptor`
+-- `descriptor`
|   +-- `descriptor_base.py` (```DescriptorBase```)
+-- `detector_descriptor`
|   +-- `detector_descriptor_base.py` (```DetectorDescriptorBase```)
|   +-- `combination_detector_descriptor.py`
+-- `matcher`
|   +-- `matcher_base.py` (```MatcherBase```)
+-- `verifier`
|   +-- `verified_base.py` (```VerifierBase```)

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


