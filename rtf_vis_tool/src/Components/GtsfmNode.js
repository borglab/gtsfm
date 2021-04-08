/* Basic Div Component repeatedly rendered within LandingPageGraph.js. It is a div (a rectangular 
container to hold text) which serves as a node/vertex in a directed graph.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import '.././stylesheets/GtsfmNode.css';

function GtsfmNode(props) {
    /*
    Args:
        props.backgroundColor (string): Hex background color of string.
        props.textColor (string): Hex color of node text.
        props.leftOffset (string): Amount to offset from left of screen. 
        props.topOffset (string): Amount to offset from top of screen.
        props.text (string): Text to render in node.
        props.onClickFunction (function): Function to be called when node is clicked.
        props.funcParam (boolean): True/False value to feed into props.onClickFunction.
        
    Returns:
        A GtsfmNode component with custom styling for a blue node.
    */

    // Defines the stylings that depends on parameters passed in from LandingPageGraph.js.
    const propStyling = {
        backgroundColor: props.backgroundColor,
        color: props.textColor,
        left: props.leftOffset,
        top: props.topOffset
    }

    // For regular div nodes, simply just alert a message on the screen. 
    // This is just a placeholder for when actual information is to be displayed on click.
    return (
        <div id={props.text} 
            className="standard_div_node_style"
            style={propStyling} 
            onClick={() => props.onClickFunction(props.funcParam)}> 
            <p>{props.text}</p>
        </div>
    )
}

export default GtsfmNode;