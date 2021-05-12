/* List of all Blue Nodes in GTSFM Visualization Graph. Imported in LandingPageGraph.js. Blue Nodes
in the graph represent algorithms/functions that are called throughout the pipeline.

Author: Adi Singh
*/

const blue_node_list = [
    {
        text: 'Scene Looper',
        topOffset: 15,
        leftOffset: 0,
    },
    {
        text: 'Scene Image Directories',
        topOffset: 0,
        leftOffset: 0,
    },
    {
        text: 'Data Loader + Filter Invalid Edges',
        topOffset: 5,
        leftOffset: 20
    },
    {
        text: 'Detector',
        topOffset: 18,
        leftOffset: 30
    },
    {
        text: 'Keypoint Describer',
        topOffset: 10,
        leftOffset: 39
    },
    {
        text: 'Putative Matcher',
        topOffset: 18,
        leftOffset: 53
    },
    {
        text: 'Verifier',
        topOffset: 8,
        leftOffset: 69
    },
    {
        text: 'Post-Processor',
        topOffset: 8,
        leftOffset: 84
    },
    {
        text: 'Optimizer',
        topOffset: 39,
        leftOffset: 34
    },
    {
        text: 'File Writer',
        topOffset: 70,
        leftOffset: 30
    },
    {
        text: 'MVSNet',
        topOffset: 70,
        leftOffset: 52
    },
    {
        text: 'Triangulation',
        topOffset: 70,
        leftOffset: 65
    },
    {
        text: 'Aggregate',
        topOffset: 87,
        leftOffset: 72.5
    },
    {
        text: 'Data Association w/ Track Filtering',
        topOffset: 36,
        leftOffset: 47
    },
    {
        text: 'Bundler Calibrator',
        topOffset: 49,
        leftOffset: 53
    },
    {
        text: '1d-SfM',
        topOffset: 35,
        leftOffset: 70
    },
    {
        text: 'Largest Connected Component Extractor',
        topOffset: 33,
        leftOffset: 86
    },
    {
        text: 'Shonan',
        topOffset: 52,
        leftOffset: 76
    },
]

module.exports = blue_node_list;