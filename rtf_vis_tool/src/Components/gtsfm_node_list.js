//List of all 43 Nodes in GTSFM Visualization Graph
//imported in DivGraph.js

const gtsfm_node_set = ['absolute Rs',
                        'Descriptors',
                        'DigiCamDB',
                        'E matrix', 
                        'relative Rs (2): i2_R_i1', 
                        'Images', 
                        'Post-Processor', 
                        'relative Rs: i2_r_i1', 
                        'LargestConnected Component Extractor', 
                        'Image Pair Indices (i1,i2)', 
                        'relative Ts: i2_t_i1',  
                        'Scene Directory',
                        'Dense Point Cloud', 
                        'Verified Correspondence Indices',
                        'Intrinsics',
                        'File Writer', 
                        'MVSNet', 
                        'Triangulation', 
                        'Optimizer', 
                        'Keypoint Describer',
                        'SfMData', 
                        'absolute Ts', 
                        'Output Directory', 
                        'Scene Image Directories', 
                        'Detector',
                        'Aggregate',
                        'Bundler Calibrator', 
                        'Keypoints',
                        'Shonan', 
                        'Zipped Results for All Scenes', 
                        'Verifier',
                        '1d-SfM',
                        'Bundler Pinhole Cameras', 
                        'Data Loader + Filter Invalid Edges', 
                        'Scene Looper',
                        'Putative Matcher', 
                        'Data Association w/ Track Filtering', 
                        'SFMResult as files', 
                        'Dense Mesh Reconstruction', 
                        'Putative Correspondence Indices', 
                        'relative Ts (2): i2_t_i1', 
                        'SFMResult (including Sparse Point Cloud, Optimized Intrinsics, absolute Rs, absolute Ts)',
                        'Image i']; 

var node_set_formatted = [];
for (var i = 0; i < gtsfm_node_set.length; i++) {
    node_set_formatted.push({id: gtsfm_node_set[i]});
}

module.exports = node_set_formatted;