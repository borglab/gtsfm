/* Landing Page Component to display GTSFM graph (what user first sees).

Author: Adi Singh
*/
import React, {useEffect, useState} from "react";

// Third-Party Package Imports.
import Xarrow from "react-xarrows";  // Used to render directed edges.

// Local Imports.
import BlueNode from './BlueNode.js';
import BlueNodes from './gtsfm_graph/blue_nodes.js';
import Bundle_Adj_PC from './Bundle_Adj_PC';
import data_association_json from '../result_metrics/data_association_metrics.json';
import EdgeList from './gtsfm_graph/edge_list.js';
import frontend_summary_json from '../result_metrics/frontend_summary.json';
import GrayNode from './GrayNode';
import GrayNodes from './gtsfm_graph/gray_nodes.js';
import multiview_optimizer_json from '../result_metrics/multiview_optimizer_metrics.json';
import GtsfmNode from './GtsfmNode';
import '../stylesheets/LandingPageGraph.css'

function LandingPageGraph() {
    /*
    Args:
        None
        
    Returns:
        The Landing Page Graph which the user can interact with.
    */

    const [arrowList, setArrowList] = useState([]); // Array storing all directed edges.
    const [grayNodesList, setGrayNodesList] = useState([]); // Array storing all gray nodes.
    const [blueNodesList, setBlueNodesList] = useState([]); // Array storing all bue nodes.
    
    const lightGray = '#dfe8e6';

    // Boolean variables indicating which pop ups to show
    const [showFS, setShowFS] = useState(false);
    const [showMVO, setShowMVO] = useState(false);
    const [showDA_PC, setShowDA_PC] = useState(null);
    const [showRSS, setShowRSS] = useState(false);

    // Variables storing JSON information from result_metrics directory.
    const [fs_json, setFS_JSON] = useState(null);
    const [mvo_json, setMVO_JSON] = useState(null);
    const [da_json, setDA_JSON] = useState(null);
    const [rotated_da_json, setRotatedDAJSON] = useState(null);

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
                    strokeWidth='1.5'
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
        setFS_JSON(frontend_summary_json);
        setMVO_JSON(multiview_optimizer_json);
        setDA_JSON(data_association_json.points_3d);
        setRotatedDAJSON(data_association_json.rotated_points_3d);
    }, [])

    function toggleFrontEndSummaryDisplay(showDisplay) {
        /*Toggles the display of the frontend summary metrics.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
            
        Returns:
            None.
        */
        setShowFS(showDisplay);
    };

    function toggleMVOMetrics(showDisplay) {
        /*Toggles the display of the multiview optimizer metrics.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
            
        Returns:
            None.
        */
        setShowMVO(showDisplay);
    };

    function toggleBundleAdj_PointCloud(showDisplay) {
        /*Toggles the display of the Bundle Adjustment Point Cloud.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
            
        Returns:
            None.
        */
        setShowDA_PC(showDisplay);
    };

    function toggleRotSummaryDisplay(showDisplay) {
        /*Toggles the display of the rotation averaging metrics.

        Args:
            showDisplay (boolean): Sets the display to be shown or not.
            
        Returns:
            None.
        */
        setShowRSS(showDisplay);
    };

    return (
        <div className="lp_graph_container">
            <div className="navbar">
                <h2 className="gtsfm_header">GTSFM Computational Graph Visualizer</h2>
            </div>

            {/* Render popups only when the respective node is clicked. */} 
            {showDA_PC && <Bundle_Adj_PC toggleBA_PC={toggleBundleAdj_PointCloud}/>}

            <div className="gtsfm_graph">

                {/* Render all Gray and Blue Nodes (43 combined). */}
                {grayNodesList}
                {blueNodesList}
                <GtsfmNode 
                    onClickFunction={toggleBundleAdj_PointCloud}
                    funcParam={true}
                    textColor={'black'} 
                    backgroundColor={lightGray} 
                    topOffset={'40%'} 
                    leftOffset={'34%'} 
                    text={'GtsfmData'}/>

                {/* Render Directed Edges. */}
                {arrowList}

                {/* Render Plates. */}
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