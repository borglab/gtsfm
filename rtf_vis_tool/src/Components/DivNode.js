import React from "react";

//Basic Div Component repeatedly rendered in DivGraph.js
const DivNode = (props) => {

    // a standard styling for each node in the graph
    const divNodeStyle = {
        position: 'absolute',
        top: props.topOffset,
        left: props.leftOffset,
        backgroundColor: props.backgroundColor,
        width: 'auto',
        height: 'auto',
        border: '1px solid black',
        borderRadius: '10px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        paddingRight: '5px',
        paddingLeft: '5px',
        cursor: 'pointer',
        maxWidth: '5%',
        zIndex: 2
    }

    //For regular div nodes, simply just alert a message on the screen. 
    //This is just a placeholder for when actual information is to be displayed on click.
    return (
        <div id={props.text} 
            style={divNodeStyle} 
            onClick={() => alert(`You Clicked ${props.text}`)}> 
            <p style={{fontSize: '75%', color: props.textColor}}>{props.text}</p>
        </div>
    )
}

export default DivNode;