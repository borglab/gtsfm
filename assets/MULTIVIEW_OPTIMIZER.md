# Multi-View Optimizer

![Alt text](gtsfm-overview-two-view-estimator.svg?raw=true)

- [Loader](assets/LOADER.md)
- [Image Pairs Generator](assets/IMAGE_PAIRS_GENERATOR.md)
- [Correspondence Generator](assets/CORRESPONDENCE_GENERATOR.md)
- [Two View Estimator](assets/TWO_VIEW_ESTIMATOR.md)
- [**Multi-View Optimizer**](#what-is-a-multi-view-optimizer)

The multi-view optimizer aggregates the relative poses between camera pairs obtained from the two-view optimizer and optimizes for the camera poses (3D rotation and 3D translation) in a world frame along with the 3D locations of the landmarks obtained from 2D tracks. It comprises the following modules: 

- [Multi-View Optimizer](#multi-view-optimizer)
    - [View graph estimation](#view-graph-estimation)
    - [Rotation averaging](#rotation-averaging)
    - [Translation averaging](#translation-averaging)
    - [Data association](#data-association)
    - [Bundle adjustment](#bundle-adjustment)

### View graph estimation

This stage aggregates the relative poses from the two-view optimizer into a "view-graph" where the nodes are the cameras and the edges are two-view relative pose constraints between 2 cameras. In addition, we use a cycle-consistency check on the relative poses. The cycle-consistency check discards edges that have a mean cylic rotation error. The cylic rotation error is the difference of the composed 3D rotation along a cycle to an identity rotation. 

### Rotation averaging

The rotation averaging module initializes the absolute 3D rotation in the world frame for all cameras by solving an optimization problem. We use the [Shonan Rotation Averaging](https://dellaert.github.io/ShonanAveraging/) implementation from GTSAM. 

### Translation averaging

The translation averaging module initializes the absolute 3D translations of the cameras in the world frame by solving an optimization problem. We use a modified implementation of the [1DSfM](https://www.cs.cornell.edu/projects/1dsfm/) translation averaging formulation. 

We support an optional outlier rejection step that rejects two-view relative translations using the MFAS outlier rejection method proposed in 1DSfM. We also support estimating the 3D locations of the landmarks by optimizing jointly with the camera translations with camera-landmark direction constraints. 

### Data association

The data association step aggregates 2-view point correspondences into 2D multi-view tracks and triangulates them using the initialized camera poses. We discard tracks that have a high reprojection error post triangulation or a very low triangulation angle. 

### Bundle adjustment

This stage jointly optimizes the 6DoF camera poses and the 3D locations of the landmarks using the reprojection error. We do multiple iterations of bundle adjustment, rejecting landmarks that have a high reprojection error after each iteration, while progressively tightening the reprojection error threshold. 