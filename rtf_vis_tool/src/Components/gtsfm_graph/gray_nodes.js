/* List of all Gray Nodes in GTSFM Visualization Graph. Imported in LandingPageGraph.js.
Gray Nodes in the graph represent input/output data structures that are fed into or outputted
by various functions.

Author: Adi Singh
*/

const gray_node_list = [
    {
        text: 'Scene Directory',
        topOffset: 15,
        leftOffset: 10
    },
    {
        text: 'DigiCamDB',
        topOffset: 8,
        leftOffset: 10.5
    },
    {
        text: 'Image i',
        topOffset: 10.5,
        leftOffset: 28
    },
    {
        text: 'Keypoints',
        topOffset: 18,
        leftOffset: 35
    },
    {
        text: 'Descriptors',
        topOffset: 18,
        leftOffset: 46
    },
    {
        text: 'Intrinsics',
        topOffset: 7,
        leftOffset: 45
    },
    {
        text: 'Image Pair Indices (i1,i2)',
        topOffset: 5,
        leftOffset: 54
    },
    {
        text: 'Putative Correspondence Indices',
        topOffset: 17,
        leftOffset: 60
    },
    {
        text: 'E matrix',
        topOffset: 8,
        leftOffset: 75
    },
    {
        text: 'Verified Correspondence Indices',
        topOffset: 16,
        leftOffset: 72
    },
    {
        text: 'relative Rs: i2Ri1',
        topOffset: 17,
        leftOffset: 79
    },
    {
        text: 'relative ts: i2ti1',
        topOffset: 17,
        leftOffset: 87
    },
    {
        text: 'Images',
        topOffset: 50,
        leftOffset: 21
    },
    {
        text: 'Output Directory',
        topOffset: 69,
        leftOffset: 22
    },
    {
        text: 'GtsfmData as files',
        topOffset: 69,
        leftOffset: 37
    },
    {
        text: 'Dense Point Cloud',
        topOffset: 70,
        leftOffset: 57
    },
    {
        text: 'Dense Mesh Reconstruction',
        topOffset: 70,
        leftOffset: 72
    },
    {
        text: 'Zipped Results for All Scenes',
        topOffset: 86,
        leftOffset: 60
    },
    {
        text: 'Bundler Pinhole Cameras',
        topOffset: 49,
        leftOffset: 45
    },
    {
        text: 'absolute ts',
        topOffset: 39,
        leftOffset: 64
    },
    {
        text: 'Pruned relative ts: i2ti1',
        topOffset: 33,
        leftOffset: 76
    },
    {
        text: 'absolute Rs',
        topOffset: 52,
        leftOffset: 65
    },
    {
        text: 'Pruned relative Rs (2): i2Ri1',
        topOffset: 50,
        leftOffset: 87
    }
]

module.exports = gray_node_list;