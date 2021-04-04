import React from "react";
import '.././stylesheets/Div_Node.css';

//Basic Div Component repeatedly rendered in DivGraph.js
const DivNode = (props) => {

    // defines the stylings that depends on parameters passed in from LandingPageGraph.js
    const propStyling = {
        backgroundColor: props.backgroundColor,
        color: props.textColor,
        left: props.leftOffset,
        top: props.topOffset
    }

    //For regular div nodes, simply just alert a message on the screen. 
    //This is just a placeholder for when actual information is to be displayed on click.
    return (
        <div id={props.text} 
            className="standard_div_node_style"
            style={propStyling} 
            onClick={() => alert(`You Clicked ${props.text}`)}> 
            <p>{props.text}</p>
        </div>
    )
}

export default DivNode;