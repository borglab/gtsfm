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
        topOffset: 20,
        leftOffset: 30
    },
    {
        text: 'Keypoint Describer',
        topOffset: 15,
        leftOffset: 40
    },
    {
        text: 'Putative Matcher',
        topOffset: 20,
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
        topOffset: 43,
        leftOffset: 28
    },
    {
        text: 'File Writer',
        topOffset: 70,
        leftOffset: 40
    },
    {
        text: 'MVSNet',
        topOffset: 78,
        leftOffset: 32
    },
    {
        text: 'Triangulation',
        topOffset: 78,
        leftOffset: 17
    },
    {
        text: 'Aggregate',
        topOffset: 83,
        leftOffset: 1
    },
    {
        text: 'Data Association w/ Track Filtering',
        topOffset: 40,
        leftOffset: 42
    },
    {
        text: 'Bundler Calibrator',
        topOffset: 53,
        leftOffset: 53
    },
    {
        text: '1d-SfM',
        topOffset: 43,
        leftOffset: 68
    },
    {
        text: 'Largest Connected Component Extractor',
        topOffset: 43,
        leftOffset: 86
    },
    {
        text: 'Shonan',
        topOffset: 65,
        leftOffset: 76
    },
]

module.exports = blue_node_list;