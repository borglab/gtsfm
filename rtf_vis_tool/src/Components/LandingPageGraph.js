import React, {useEffect, useState} from "react";
import Xarrow from "react-xarrows";
import '../stylesheets/LandingPageGraph.css'

//Loading Components
import Bundle_Adj_PC from './Bundle_Adj_PC';
import DivNode from './DivNode';
import EdgeList from './gtsfm_edge_list.js';
// import FrontendSummary from './FrontendSummary';         - add this in part 3
// import MVOSummary from './MVOSummary';                   - add this in part 3
//import RelativeRsDivNode from './relativeRsDivNode';      - add this in part 3
//import RotSuccessSummary from './RotSuccessSummary';      - add this in part 3
import SfMDataDivNode from './SfMDataDivNode';

//loading json files in result_metrics folders to display summary metrics
import data_association_json from '../result_metrics/data_association_metrics.json';
import frontend_summary_json from '../result_metrics/frontend_summary.json';
import multiview_optimizer_json from '../result_metrics/multiview_optimizer_metrics.json';

//Landing Page Component to display GTSFM graph (what user first sees)
const LandingPageGraph = (props) => {
    const [arrowList, setArrowList] = useState([]);
    const leftShift = 0;
    const topShift = 0;
    
    const aquaBlue = '#2255e0';
    const lightGray = '#dfe8e6';

    const [showFS, setShowFS] = useState(false);
    const [fs_json, setFS_JSON] = useState(null);
    const [showMVO, setShowMVO] = useState(false);
    const [mvo_json, setMVO_JSON] = useState(null);
    const [da_json, setDA_JSON] = useState(null);
    const [showDA_PC, setShowDA_PC] = useState(null);
    const [showRSS, setShowRSS] = useState(false);
    const [rotated_da_json, setRotatedDAJSON] = useState(null);

    //render all directed edges on the graph
    //save all the json resulting metrics in separate React variables
    useEffect(() => {
        var rawEdges = EdgeList
        var xArrows_formatted = [];

        for (var i = 0; i < rawEdges.length; i++) {
            const pair = rawEdges[i];
            xArrows_formatted.push(
                <Xarrow
                    start={pair[0]}
                    end={pair[1]}
                    color='gray'
                    strokeWidth='1.5'
                    path='straight'
                />)
        }

        setArrowList(xArrows_formatted);
        setFS_JSON(frontend_summary_json);
        setMVO_JSON(multiview_optimizer_json);
        setDA_JSON(data_association_json.points_3d);
        setRotatedDAJSON(data_association_json.rotated_points_3d);
    }, [])

    //Function to defines the percent offset each div node has from the top and left side of the screen
    //Used to make positioning of nodes more dynamic
    function formatPercent(shift, percent) {
        const str_percent = `${shift+percent}%`;
        return str_percent
    }

    //Functions to toggle the display of various pop ups on the screen
    //Like frontend metrics, multiview optimizer metrics, and data association point cloud
    const toggleFrontEndSummaryDisplay = (bool) => {setShowFS(bool)};
    const toggleMVOMetrics = (bool) => {setShowMVO(bool)};
    const toggleDataAssoc_PointCloud = (bool) => {setShowDA_PC(bool)};
    const toggleRotSummaryDisplay = (bool) => {setShowRSS(bool)};

    return (
        <div className="lp_graph_container">
            <div className="navbar">
                <h2 className="gtsfm_header">GTSFM Computational Graph Visualizer</h2>
            </div>

            {/* Render popups only when the respective node is clicked*/} 
            {showDA_PC && <Bundle_Adj_PC da_json={da_json} rotated_json={rotated_da_json} toggleDA_PC={toggleDataAssoc_PointCloud}/>}

            <div className="gtsfm_graph">
                {/* Render 43 Graph Nodes */}
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 0)} leftOffset={formatPercent(leftShift, 0)} text={'Scene Image Directories'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 0)} text={'Scene Looper'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 10)} text={'Scene Directory'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 10.5)} text={'DigiCamDB'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 5)} leftOffset={formatPercent(leftShift, 20)} text={'Data Loader + Filter Invalid Edges'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 12.5)} leftOffset={formatPercent(leftShift, 28)} text={'Image i'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 30)} text={'Detector'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 35)} text={'Keypoints'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 40)} text={'Keypoint Describer'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 46)} text={'Descriptors'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 9)} leftOffset={formatPercent(leftShift, 45)} text={'Intrinsics'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 53)} text={'Putative Matcher'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 5)} leftOffset={formatPercent(leftShift, 54)} text={'Image Pair Indices (i1,i2)'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 19)} leftOffset={formatPercent(leftShift, 60)} text={'Putative Correspondence Indices'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 69)} text={'Verifier'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 75)} text={'E matrix'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 84)} text={'Post-Processor'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 72)} text={'Verified Correspondence Indices'}/>
                <DivNode toggleRot={toggleRotSummaryDisplay} textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 79)} text={'relative Rs: i2Ri1'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 87)} text={'relative ts: i2ti1'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 50)} leftOffset={formatPercent(leftShift, 10)} text={'Images'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 28)} text={'Optimizer'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 40)} leftOffset={formatPercent(leftShift, 17)} text={'SFMResult (including Sparse Point Cloud, Optimized Intrinsics, absolute Rs, absolute Ts)'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 70)} leftOffset={formatPercent(leftShift, 40)} text={'File Writer'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 70)} leftOffset={formatPercent(leftShift, 47)} text={'Output Directory'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 80)} leftOffset={formatPercent(leftShift, 40)} text={'SFMResult as files'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 78)} leftOffset={formatPercent(leftShift, 32)} text={'MVSNet'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 78)} leftOffset={formatPercent(leftShift, 25)} text={'Dense Point Cloud'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 78)} leftOffset={formatPercent(leftShift, 17)} text={'Triangulation'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 78)} leftOffset={formatPercent(leftShift, 10)} text={'Dense Mesh Reconstruction'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 1)} text={'Aggregate'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 63)} leftOffset={formatPercent(leftShift, 1)} text={'Zipped Results for All Scenes'}/>
                <SfMDataDivNode json={da_json} toggleDA_PC={toggleDataAssoc_PointCloud} textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 40)} leftOffset={formatPercent(leftShift, 34)} text={'GtsfmData'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 40)} leftOffset={formatPercent(leftShift, 42)} text={'Data Association w/ Track Filtering'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 53)} leftOffset={formatPercent(leftShift, 45)} text={'Bundler Pinhole Cameras'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 53)} leftOffset={formatPercent(leftShift, 53)} text={'Bundler Calibrator'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 60)} text={'absolute ts'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 68)} text={'1d-SfM'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 76)} text={'Pruned relative ts: i2ti1'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 86)} text={'Largest Connected Component Extractor'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 63)} leftOffset={formatPercent(leftShift, 65)} text={'absolute Rs'}/>
                <DivNode textColor={'white'} backgroundColor={aquaBlue} topOffset={formatPercent(topShift, 65)} leftOffset={formatPercent(leftShift, 76)} text={'Shonan'}/>
                <DivNode textColor={'black'} backgroundColor={lightGray} topOffset={formatPercent(topShift, 62)} leftOffset={formatPercent(leftShift, 87)} text={'Pruned relative Rs (2): i2Ri1'}/>
        
                {/* Render Directed Edges */}
                {arrowList}


                {/* Render Plates */}
                <div className="scene_optimizer_plate">
                    <p className="plate_title">Scene Optimizer Scenes</p>
                </div>
                <div className="feature_extractor_plate">
                    <p className="plate_title">Feature Extractor Images</p>
                </div>
                <div className="two_view_estimator_plate" onClick={(fs_json) ? (() => toggleFrontEndSummaryDisplay(true)) : (null)}>
                    <p className="plate_title">TwoViewEstimator</p>
                </div>
                <div className="averaging_plate">
                    <p className="plate_title">Averaging</p>
                </div>
                <div className="sparse_multiview_optimizer_plate" onClick={(mvo_json) ? (() => toggleMVOMetrics(true)) : (null)}>
                    <p className="plate_title">Sparse Multiview Optimizer</p>
                </div>
                <div className="dense_multiview_optimizer_plate">
                    <p className="plate_title">Dense Multiview Optimizer</p>
                </div>
            </div>
        </div>
    )
}

export default LandingPageGraph;