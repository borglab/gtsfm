/* Gray Node Component. Built on top of GtsfmNode.js with styling properties specific to a Gray Node.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import GtsfmNode from './GtsfmNode';

function GrayNode(props) {
    /*
    Args:
        props.nodeInfo.text (string): Text to display in node.
        props.nodeInfo.topOffset (int): Amount to offset from top of screen.
        props.nodeInfo.leftOffset (int): Amount to offset from left of screen.
        
    Returns:
        A GtsfmNode component with custom styling for a gray node.
    */

    const lightGray = '#dfe8e6';
    const nodeText = props.nodeInfo.text;
    const nodeTopOffset = props.nodeInfo.topOffset;
    const nodeLeftOffset = props.nodeInfo.leftOffset;

    return (
        <GtsfmNode 
            textColor={'black'} 
            backgroundColor={lightGray} 
            topOffset={`${nodeTopOffset}%`} 
            leftOffset={`${nodeLeftOffset}%`}
            text={nodeText}/>
    )
}

export default GrayNode;