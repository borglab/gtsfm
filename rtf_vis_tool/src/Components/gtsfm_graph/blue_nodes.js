/* List of all Blue Nodes in GTSFM Visualization Graph. Imported in LandingPageGraph.js. Blue Nodes
in the graph represent algorithms/functions that are called throughout the pipeline.

Author: Adi Singh
*/

const blue_node_list = [
    {
        text: 'Image Loader',
        topOffset: 0,
        leftOffset: 10,
    },
    {
        text: 'Image Retriever',
        topOffset: 7,
        leftOffset: 0,
    },
    {
        text: 'DetectorDescriptor',
        topOffset: 0,
        leftOffset: 80
    },
    {
        text: 'Matcher',
        topOffset: 14,
        leftOffset: 80
    },
    {
        text: 'Verifier',
        topOffset: 14,
        leftOffset: 45
    },
    {
        text: 'Two-View Bundle Adjustment',
        topOffset: 30,
        leftOffset: 40
    },
    {
        text: 'View-Graph Estimator',
        topOffset: 32,
        leftOffset: 16
    },
    {
        text: 'Rotation Averaging',
        topOffset: 50,
        leftOffset: 7
    },
    {
        text: 'Translation Averaging',
        topOffset: 65,
        leftOffset: 7
    },
    {
        text: 'Data Association',
        topOffset: 74,
        leftOffset: 8
    },
    {
        text: 'Global Bundle Adjustment',
        topOffset: 84,
        leftOffset: 5
    },
    {
        text: 'Multi-view Stereo',
        topOffset: 70,
        leftOffset: 52
    },
]

module.exports = blue_node_list;