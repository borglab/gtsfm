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
                        'absolute ts(2)', 
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
                        'Optimized Intrinsics', 
                        'Keypoints', 
                        'absolute Rs(2)', 
                        'Sparse Point Cloud', 
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
                        'SFMResult', 
                        'Image i'];

var node_set_formatted = [];
for (var i = 0; i < gtsfm_node_set.length; i++) {
    node_set_formatted.push({id: gtsfm_node_set[i]});
}

//temporary
var node_sample = [{
    id: 'Scene Image Directories',
    x: 100,
    y: 100
}, {
    id: 'Scene Looper',
    x: 500,
    y: 200
}, {
    id: 'Scene Directory',
    x: 900,
    y:300
}]
//DELETE later

module.exports = node_set_formatted;