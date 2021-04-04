//List of all 43 Nodes in GTSFM Visualization Graph
//imported in DivGraph.js

const gtsfm_node_set = ['absolute Rs',
                        'absolute Ts',
                        'Aggregate',
                        'Bundler Calibrator',
                        'Bundler Pinhole Cameras',
                        'Data Association w/ Track Filtering',
                        'Data Loader + Filter Invalid Edges',
                        'Dense Mesh Reconstruction',
                        'Dense Point Cloud',
                        'Descriptors',
                        'Detector',
                        'DigiCamDB',
                        'E matrix',
                        'File Writer',
                        'Image i', 
                        'Images', 
                        'Image Pair Indices (i1,i2)',
                        'Intrinsics',
                        'Keypoint Describer',
                        'Keypoints',
                        'LargestConnected Component Extractor',
                        'MVSNet', 
                        'Optimizer',
                        'Output Directory',
                        'Post-Processor',
                        'Putative Correspondence Indices',
                        'Putative Matcher',
                        'relative Rs: i2_r_i1',
                        'relative Rs (2): i2_R_i1',    
                        'relative Ts: i2_t_i1',  
                        'relative Ts (2): i2_t_i1',
                        'Scene Directory',
                        'Scene Image Directories',
                        'Scene Looper',
                        'SfMData',
                        'SFMResult as files',
                        'SFMResult (including Sparse Point Cloud, Optimized Intrinsics, absolute Rs, absolute Ts)',
                        'Shonan',
                        'Triangulation', 
                        'Verified Correspondence Indices',
                        'Verifier',        
                        'Zipped Results for All Scenes', 
                        '1d-SfM']; 

var node_set_formatted = [];
for (var i = 0; i < gtsfm_node_set.length; i++) {
    node_set_formatted.push({id: gtsfm_node_set[i]});
}

module.exports = node_set_formatted;