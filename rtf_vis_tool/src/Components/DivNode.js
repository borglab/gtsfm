/* Basic Div Component repeatedly rendered which is rendered within LandingPageGraph.js. Called a 
"Div Node", primarily because in code, it is a div (a rectangular container to hold text), while in
concept, it serves as a node/vertex in a directed graph.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import '.././stylesheets/Div_Node.css';

const DivNode = (props) => {

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

export default DivNode;