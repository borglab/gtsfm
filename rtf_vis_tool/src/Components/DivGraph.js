import React, {useEffect, useState} from "react";
import Xarrow from "react-xarrows";

import EdgeList from './gtsfm_edge_list.js';
import DivNode from './DivNode';
import '../stylesheets/DivGraph.css'

const DivGraph = (props) => {
    const [arrowList, setArrowList] = useState([]);
    const leftShift = 0;
    const topShift = 0;

    //render all edges in graph
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
    }, [])

    function formatPercent(shift, percent) {
        const str_percent = `${shift+percent}%`;
        return str_percent
    }

    return (
        <div className="div_graph_container">
            <div className="navbar">
                <h2 className="gtsfm_header">GTSFM Pipeline Graph</h2>
            </div>

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
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 48)} text={'Descriptors'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 48)} text={'Intrinsics'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 53)} text={'Putative Matcher'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 0)} leftOffset={formatPercent(leftShift, 51)} text={'Image Pair Indices (i1,i2)'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 15)} leftOffset={formatPercent(leftShift, 60)} text={'Putative Correspondence Indices'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 69)} text={'Verifier'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 75)} text={'E matrix'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 8)} leftOffset={formatPercent(leftShift, 84)} text={'Post-Processor'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 72)} text={'Verified Correspondence Indices'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 79)} text={'relative Rs: i2_r_i1'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 21)} leftOffset={formatPercent(leftShift, 87)} text={'relative Ts: i2_t_i1'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 70)} leftOffset={formatPercent(leftShift, 0)} text={'Images'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 38)} leftOffset={formatPercent(leftShift, 8)} text={'Output Directory'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 45)} leftOffset={formatPercent(leftShift, 9)} text={'SFMResult as files'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 16)} text={'File Writer'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 50)} leftOffset={formatPercent(leftShift, 20)} text={'SFMResult'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 43)} leftOffset={formatPercent(leftShift, 25)} text={'Optimizer'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 61)} leftOffset={formatPercent(leftShift, 10)} text={'Sparse Point Cloud'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 68)} leftOffset={formatPercent(leftShift, 15)} text={'Optimized Intrinsics'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 68)} leftOffset={formatPercent(leftShift, 21)} text={'absolute Rs(2)'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 61)} leftOffset={formatPercent(leftShift, 26)} text={'absolute ts(2)'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 10)} text={'MVSNet'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 17)} text={'Dense Point Cloud'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 25)} text={'Triangulation'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 32)} text={'Dense Mesh Reconstruction'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 40)} text={'Aggregate'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 83)} leftOffset={formatPercent(leftShift, 47)} text={'Zipped Results for All Scenes'}/>
                <DivNode textColor={'black'} backgroundColor={'#dfe8e6'} topOffset={formatPercent(topShift, 40)} leftOffset={formatPercent(leftShift, 34)} text={'SfMData'}/>
                <DivNode textColor={'white'} backgroundColor={'#2255e0'} topOffset={formatPercent(topShift, 40)} leftOffset={formatPercent(leftShift, 42)} text={'Data Association w/ Track Filtering'}/>
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
            </div>
        </div>
    )
}

export default DivGraph;