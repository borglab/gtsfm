import React, {useEffect, useState} from "react";
import Xarrow from "react-xarrows";

import EdgeList from './gtsfm_edge_list.js';
import DivNode from './DivNode';
import PPDivNode from './PPDivNode';
import OptDivNode from './OptDivNode';
import DADivNode from './DADivNode';
import SFMResultDivNode from './SFMResultDivNode';
import FrontendSummary from './FrontendSummary';
import MVOSummary from './MVOSummary';
import Data_Association_PC from './Data_Association_PC';
import '../stylesheets/DivGraph.css'


import frontend_summary_json from '.././result_metrics/frontend_summary.json';
import multiview_optimizer_json from '.././result_metrics/multiview_optimizer_metrics.json';
import data_association_json from '.././result_metrics/data_association_metrics.json';

const DivGraph = (props) => {
    const [arrowList, setArrowList] = useState([]);
    const leftShift = 0;
    const topShift = 0;

    const [showFS, setShowFS] = useState(false);
    const [fs_json, setFS_JSON] = useState(null);
    const [showMVO, setShowMVO] = useState(false);
    const [mvo_json, setMVO_JSON] = useState(null);
    const [da_json, setDA_JSON] = useState(null);
    const [showDA_PC, setShowDA_PC] = useState(null);

    //render all edges and output files in graph
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
    }, [])

    function formatPercent(shift, percent) {
        const str_percent = `${shift+percent}%`;
        return str_percent
    }

    //Function
    const toggleFrontEndSummaryDisplay = (bool) => {setShowFS(bool)};
    const toggleMVOMetrics = (bool) => {setShowMVO(bool)};
    const toggleDataAssoc_PointCloud = (bool) => {setShowDA_PC(bool)};

    return (
        <div className="div_graph_container">
            <div className="navbar">
                <h2 className="gtsfm_header">GTSFM Pipeline Graph</h2>
            </div>

            {showFS && <FrontendSummary json={fs_json} toggleFS={toggleFrontEndSummaryDisplay}/>}
            {showMVO && <MVOSummary json={mvo_json} toggleMVO={toggleMVOMetrics}/>}
            {showDA_PC && <Data_Association_PC json={da_json} toggleDA_PC={toggleDataAssoc_PointCloud}/>}

            <div className="gtsfm_graph">
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 0)} leftOffset={formatPercent(leftShift, 0)} text={'Scene Image Directories'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 0)} text={'Scene Looper'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 10)} text={'Scene Directory'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 10.5)} text={'DigiCamDB'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 5)} leftOffset={formatPercent(leftShift, 20)} text={'Data Loader + Filter Invalid Edges'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 12.5)} leftOffset={formatPercent(leftShift, 28)} text={'Image i'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 30)} text={'Detector'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 35)} text={'Keypoints'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 40)} text={'Keypoint Describer'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 46)} text={'Descriptors'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 9)} leftOffset={formatPercent(leftShift, 45)} text={'Intrinsics'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 20)} leftOffset={formatPercent(leftShift, 53)} text={'Putative Matcher'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 5)} leftOffset={formatPercent(leftShift, 54)} text={'Image Pair Indices (i1,i2)'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 60)} text={'Putative Correspondence Indices'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 69)} text={'Verifier'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 75)} text={'E matrix'}/>
                <PPDivNode json={fs_json} toggleFS={toggleFrontEndSummaryDisplay} textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 84)} text={'Post-Processor'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 72)} text={'Verified Correspondence Indices'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 79)} text={'relative Rs: i2_r_i1'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 87)} text={'relative Ts: i2_t_i1'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 70)} leftOffset={formatPercent(leftShift, 10)} text={'Images'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 38)} leftOffset={formatPercent(leftShift, 8)} text={'Output Directory'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 45)} leftOffset={formatPercent(leftShift, 9)} text={'SFMResult as files'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 19)} text={'File Writer'}/>
                <OptDivNode json={mvo_json} toggleMVO={toggleMVOMetrics} textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 28)} text={'Optimizer'}/>
                <SFMResultDivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 53)} leftOffset={formatPercent(leftShift, 25)} text={'SFMResult (including Sparse Point Cloud, Optimized Intrinsics, absolute Rs, absolute Ts)'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 32)} text={'MVSNet'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 25)} text={'Dense Point Cloud'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 17)} text={'Triangulation'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 10)} text={'Dense Mesh Reconstruction'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 1)} text={'Aggregate'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 63)} leftOffset={formatPercent(leftShift, 1)} text={'Zipped Results for All Scenes'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 40)} leftOffset={formatPercent(leftShift, 34)} text={'SfMData'}/>
                <DADivNode json={da_json} toggleDA_PC={toggleDataAssoc_PointCloud} textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 40)} leftOffset={formatPercent(leftShift, 42)} text={'Data Association w/ Track Filtering'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 53)} leftOffset={formatPercent(leftShift, 45)} text={'Bundler Pinhole Cameras'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 53)} leftOffset={formatPercent(leftShift, 53)} text={'Bundler Calibrator'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 60)} text={'absolute Ts'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 68)} text={'1d-SfM'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 76)} text={'relative Ts (2): i2_t_i1'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 86)} text={'Largest Connected Component Extractor'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 63)} leftOffset={formatPercent(leftShift, 65)} text={'absolute Rs'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 65)} leftOffset={formatPercent(leftShift, 76)} text={'Shonan'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 65)} leftOffset={formatPercent(leftShift, 87)} text={'relative Rs (2): i2_R_i1'}/>
        
                {arrowList}
                <div className="scene_optimizer_plate">
                    <p style={{color: 'red', fontWeight: 'bold'}}>Scene Optimizer Scenes</p>
                </div>
                <div className="feature_extractor_plate">
                    <p style={{color: 'red', fontWeight: 'bold'}}>Feature Extractor Images</p>
                </div>
                <div className="two_view_estimator_plate">
                    <p style={{color: 'red', fontWeight: 'bold'}}>TwoViewEstimator</p>
                </div>
                <div className="multiview_optimizer_plate">
                    <p style={{color: 'red', fontWeight: 'bold'}}>MultiViewEstimator</p>
                </div>
            </div>
        </div>
    )
}

export default DivGraph;