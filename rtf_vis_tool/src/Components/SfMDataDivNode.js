import React from "react";
import '.././stylesheets/Div_Node.css';

//The Gray 'SfMData' Node Component 
//When clicked, it spawns the Bundle Adjustment point cloud (located in Bundle_Adj_PC.js)
const SfMDataDivNode = (props) => {

    // defines the stylings that depends on parameters passed in from LandingPageGraph.js
    const propStyling = {
        top: props.topOffset,
        left: props.leftOffset,
        backgroundColor: props.backgroundColor,
        color: props.textColor
    }

    return (
        <div id={props.text} 
            className="standard_div_node_style"
            style={propStyling} 
            onClick={(props.json) ? (() => props.toggleDA_PC(true)) : (null)}>     
            <p>{props.text}</p>
        </div>
    )
}

export default SfMDataDivNode;