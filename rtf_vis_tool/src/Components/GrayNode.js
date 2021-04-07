/* Gray Node Component. Built on top of Node.js with styling properties specific to a Gray Node.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import Node from './Node';

const GrayNode = (props) => {
    const lightGray = '#dfe8e6';
    const nodeText = props.nodeInfo.text;
    const nodeTopOffset = props.nodeInfo.topOffset;
    const nodeLeftOffset = props.nodeInfo.leftOffset;

    return (
        <Node 
            textColor={'black'} 
            backgroundColor={lightGray} 
            topOffset={`${nodeTopOffset}%`} 
            leftOffset={`${nodeLeftOffset}%`}
            text={nodeText}/>
    )
}

export default GrayNode;