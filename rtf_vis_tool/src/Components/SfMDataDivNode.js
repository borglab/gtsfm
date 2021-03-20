import React from "react";

const SfMDataDivNode = (props) => {
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
    }

    return (
        <div id={props.text} 
            style={divNodeStyle} 
            onClick={(props.json) ? (() => props.toggleDA_PC(true)) : (null)}>
            <p style={{fontSize: '75%', color: props.textColor}}>{props.text}</p>
        </div>
    )
}

export default SfMDataDivNode;