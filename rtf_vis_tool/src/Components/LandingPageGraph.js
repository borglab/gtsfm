/* Landing Page Component to display GTSFM graph (what user first sees).

Authors: Adi Singh, Kevin Fu
*/
import React, {useEffect, useState} from "react";

// Third-Party Package Imports.
import Xarrow from "react-xarrows";  // Used to render directed edges.

// Local Imports.
import BlueNode from './BlueNode.js';
import BlueNodes from './gtsfm_graph/blue_nodes.js';
import EdgeList from './gtsfm_graph/edge_list.js';
import FrontendSummary from './FrontendSummary';
import GrayNode from './GrayNode';
import GrayNodes from './gtsfm_graph/gray_nodes.js';
import GtsfmNode from './GtsfmNode';
import MVOSummary from './MVOSummary';
import PCViewer from './PCViewer.js';
import ImageViewer from './ImageViewer.js';
import '../stylesheets/LandingPageGraph.css'

// JSON result_metrics data
import raw_frontend_summary_json from '../result_metrics/verifier_summary_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json';
import raw_rot_avg_json from '../result_metrics/rotation_averaging_metrics.json';
import raw_trans_avg_json from '../result_metrics/translation_averaging_metrics.json';

function LandingPageGraph() {
    /*
    Returns:
        The Landing Page Graph which the user can interact with.
    */

    const [arrowList, setArrowList] = useState([]); // Array storing all directed edges.
    const [grayNodesList, setGrayNodesList] = useState([]); // Array storing all gray nodes.
    const [blueNodesList, setBlueNodesList] = useState([]); // Array storing all bue nodes.
    
    const lightGray = '#dfe8e6';

    // Boolean variables indicating which pop ups to show
    const [showFS, setShowFS] = useState(false);
    const [showAveragingMetrics, setShowAveragingMetrics] = useState(false);
    const [showDA_PC, setShowDA_PC] = useState(false);
    const [showBA_PC, setShowBA_PC] = useState(false);
    const [showImageViewer, setShowImageViewer] = useState(false);
    
    // Variables storing JSON information from result_metrics directory.
    const [frontend_summary_json, setFSJSON] = useState(null);
    const [rotation_averaging_json, setRotationAveragingJSON] = useState(null);
    const [translation_averaging_json, setTranslationAveragingJSON] = useState(null);

    useEffect(() => {
        var rawEdges = EdgeList
        var xArrows_formatted = [];

        // Render all directed edges on the graph.
        for (var i = 0; i < rawEdges.length; i++) {
            const pair = rawEdges[i];
            xArrows_formatted.push(
                <Xarrow
                    start={pair[0]}
                    end={pair[1]}
                    color='gray'
                    strokeWidth={1.5}
                    path='straight'
                />)
        }
        setArrowList(xArrows_formatted);

        // Render all gray nodes in graph.
        var grayNodes = GrayNodes;
        var grayNodes_formatted = [];
        for (var j = 0; j < grayNodes.length; j++) {
            grayNodes_formatted.push(<GrayNode nodeInfo={grayNodes[j]}/>);
        }
        setGrayNodesList(grayNodes_formatted);
        
        // Render all blue nodes in graph.
        var blueNodes = BlueNodes;
        var blueNodes_formatted = [];
        for (var k = 0; k < blueNodes.length; k++) {
            blueNodes_formatted.push(<BlueNode nodeInfo={blueNodes[k]}/>)
        }
        setBlueNodesList(blueNodes_formatted);

        // Save all the json resulting metrics in separate React variables.
        setFSJSON(raw_frontend_summary_json.verifier_summary_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT);
        setRotationAveragingJSON(raw_rot_avg_json.rotation_averaging_metrics);
        setTranslationAveragingJSON(raw_trans_avg_json.translation_averaging_metrics);
    }, [])

    function toggleFrontEndSummaryDisplay(showDisplay) {
        /*Toggles the display of the frontend summary metrics.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
        */
        setShowFS(showDisplay);
    };

    function toggleAveragingMetrics(showDisplay) {
        /*Toggles the display of the multiview optimizer metrics.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
        */
        setShowAveragingMetrics(showDisplay);
    };

    function toggleDA_PointCloud(showDisplay) {
        /*Toggles the display of the Data Association Point Cloud.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
        */
        setShowDA_PC(showDisplay);
    };

    function toggleImageViewer(showDisplay) {
        /*Toggles the display of the Image Loader.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
        */
        setShowImageViewer(showDisplay);
    };

    function toggleBA_PointCloud(showDisplay) {
        /*Toggles the display of the Bundle Adjustment Point Cloud.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
        */
        setShowBA_PC(showDisplay);
    };

    return (
        <div className="lp_graph_container">
            <div className="navbar">
                <h2 className="gtsfm_header">GTSFM Computational Graph Visualizer</h2>
            </div>

            {/* Render popups only when the respective node is clicked. */} 
            {showDA_PC && <PCViewer title={'3D Tracks 1'} 
                                    togglePC={toggleDA_PointCloud} 
                                    pointCloudType={'ba_input'}/>}
            {showImageViewer && <ImageViewer title={'Images'} 
                                    togglePC={toggleImageViewer} 
                                    pointCloudType={'ba_input'}/>}
            {showBA_PC && <PCViewer title={'Optimized 3D Tracks'}
                                    togglePC={toggleBA_PointCloud}
                                    pointCloudType={'ba_output'}/>}
      
            {/* show the frontend summary post 2-view-estimator via:
              * result_metrics/verifier_summary_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json
              */}
            {showFS && <FrontendSummary frontend_summary={frontend_summary_json} toggleFS={toggleFrontEndSummaryDisplay}/>}

            {/* show averaging metrics for sparse multiview optimizer via: 
              * result_metrics/rotation_averaging_metrics
              * result_metrics/translation_averaging_metrics
              */}
            {showAveragingMetrics && <MVOSummary rotation_averaging_metrics={rotation_averaging_json} translation_averaging_metrics={translation_averaging_json} toggleMVO={toggleAveragingMetrics}/>}

            <div className="gtsfm_graph">

                {/* Render basic Gray and Blue Nodes (41). These 41 + 2 point cloud nodes yield 43 nodes total. */}
                {grayNodesList}
                {blueNodesList}

                {/* Render 2 more nodes which spawn point clouds (from ba_input and ba_output). */}
                <GtsfmNode 
                    onClickFunction={toggleDA_PointCloud}
                    funcParam={true}
                    textColor={'black'} 
                    backgroundColor={lightGray} 
                    topOffset={'76%'} 
                    leftOffset={'18%'} 
                    text={'3D Tracks'}/>
                
                <GtsfmNode 
                    onClickFunction={setShowImageViewer}
                    funcParam={true}
                    textColor={'black'} 
                    backgroundColor={lightGray} 
                    topOffset={'7%'} 
                    leftOffset={'28%'} 
                    text={'Images'}/>

                <GtsfmNode
                    onClickFunction={toggleBA_PointCloud}
                    funcParam={true}
                    textColor={'black'}
                    backgroundColor={lightGray}
                    topOffset={'90%'}
                    leftOffset={'10%'}
                    text={'Optimized 3D Tracks'}/>

                {/* Render Directed Edges. */}
                {arrowList}

                {/* Render Plates. */}
                <div className="loader_and_retriever">
                    <p className="plate_title">Loader and Retriever</p>
                </div>
                <div className="two_view_estimator_plate">
                    <p className="plate_title">Two-View Estimator</p>
                </div>
                <div className="correspondence_plate">
                    <p className="plate_title">DetDescCorrespondenceGenerator</p>
                </div>
                <div className="sparse_reconstruction_plate">
                    <p className="plate_title">Sparse Reconstruction</p>
                </div>
                <div className="sparse_multiview_optimizer_plate" 
                     onMouseEnter={(rotation_averaging_json, translation_averaging_json) ? (() => toggleAveragingMetrics(true)) : (null)}
                     onMouseLeave={(rotation_averaging_json, translation_averaging_json) ? (() => toggleAveragingMetrics(false)) : (null)}
                >
                    <p className="plate_title">Sparse Multiview Optimizer</p>
                </div>
            </div>
        </div>
    )
}

export default LandingPageGraph;
