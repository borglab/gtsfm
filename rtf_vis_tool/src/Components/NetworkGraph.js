import React from "react";
import {Graph} from "react-d3-graph";
import '.././stylesheets/NetworkGraph.css';

import NodeList from './gtsfm_node_list.js';
import EdgeList from './gtsfm_edge_list.js';

const NetworkGraph = (props) => {

    const data = {
        nodes: NodeList,
        links: EdgeList,
        directed: true
    };

    const graphConfig = {
        // staticGraph: true,
        // staticGraphWithDragAndDrop: true,
        width: 5000,
        height: 2000,
        nodeHighlightBehavior: true,
        directed: true,
        node: {
          color: "#84b0f5",
          size: 1000,
          highlightColor: "red",
          fontSize: 12,
          labelPosition: "center",
          symbolType: "circle"
        },
        link: {
            color: "#929693",
            strokeWidth: 2
        },
        d3: {
            linkLength: 100,
            gravity: -350
        }
    };

    const onClickNode = function(nodeId) {
        alert(`Clicked node ${nodeId}`);
    };

    return (
        <Graph
            id="graph-id" // id is mandatory
            data={data}
            config={graphConfig}
            onClickNode={onClickNode}
        />
    )
}

export default NetworkGraph;