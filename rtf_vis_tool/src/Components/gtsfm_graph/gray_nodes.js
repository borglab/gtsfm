/* List of all Gray Nodes in GTSFM Visualization Graph. Imported in LandingPageGraph.js.
Gray Nodes in the graph represent input/output data structures that are fed into or outputted
by various functions.

Author: Adi Singh
*/

const gray_node_list = [
    {
        text: 'Relative Pose Priors',
        topOffset: 7,
        leftOffset: 7
    },
    {
        text: 'Absolute Pose Priors',
        topOffset: 7,
        leftOffset: 14
    },
    {
        text: 'Camera Intrinsics',
        topOffset: 7,
        leftOffset: 21
    },
    {
        text: 'Images',
        topOffset: 7,
        leftOffset: 28
    },
    {
        text: 'Image Shapes',
        topOffset: 7,
        leftOffset: 32
    },
    {
        text: 'Image Pair Indices',
        topOffset: 16,
        leftOffset: 0
    },
    {
        text: 'Keypoints',
        topOffset: 7,
        leftOffset: 78
    },
    {
        text: 'Descriptors',
        topOffset: 7,
        leftOffset: 85
    },
    {
        text: 'Putative Correspondences',
        topOffset: 21,
        leftOffset: 82
    },
    {
        text: 'Verified Correspondences',
        topOffset: 21,
        leftOffset: 35
    },
    {
        text: 'Relative Rotation',
        topOffset: 21,
        leftOffset: 42
    },
    {
        text: 'Relative Translation',
        topOffset: 21,
        leftOffset: 49
    },
    {
        text: 'Inlier Ratio',
        topOffset: 21,
        leftOffset: 56
    },
    {
        text: 'Optimized Relative Rotation',
        topOffset: 40,
        leftOffset: 32
    },
    {
        text: 'Optimized Relative Translation',
        topOffset: 40,
        leftOffset: 42
    },
    {
        text: 'Inlier Correspondences',
        topOffset: 40,
        leftOffset: 49
    },
    {
        text: 'View-Graph Relative Rotations',
        topOffset: 40,
        leftOffset: 0
    },
    {
        text: 'View-Graph Relative Translations',
        topOffset: 40,
        leftOffset: 10
    },
    {
        text: 'Global Rotations',
        topOffset: 60,
        leftOffset: 0
    },
    {
        text: 'View-Graph Correspondences',
        topOffset: 60,
        leftOffset: 20
    },
    {
        text: 'Global Translations',
        topOffset: 65,
        leftOffset: 13
    },
    {
        text: 'Optimized Camera Poses',
        topOffset: 90,
        leftOffset: 17
    },
    {
        text: 'Dense Colored 3D Point Cloud',
        topOffset: 80,
        leftOffset: 50
    }
]

module.exports = gray_node_list;