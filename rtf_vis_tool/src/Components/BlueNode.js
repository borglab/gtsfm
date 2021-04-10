/* Blue Node Component. Built on top of GtsfmNode.js with styling properties specific to a Blue Node.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import GtsfmNode from './GtsfmNode';

function BlueNode(props) {
    /*
    Args:
        props.nodeInfo.text (string): Text to display in node.
        props.nodeInfo.topOffset (int): Amount to offset from top of screen.
        props.nodeInfo.leftOffset (int): Amount to offset from left of screen.
        
    Returns:
        A GtsfmNode component with custom styling for a blue node.
    */

    const aquaBlue = '#2255e0';
    const nodeText = props.nodeInfo.text;
    const nodeTopOffset = props.nodeInfo.topOffset;
    const nodeLeftOffset = props.nodeInfo.leftOffset;

    function defaultFunction(text) {
        alert(`You Clicked ${text}`);
    }

    return (
        <GtsfmNode 
            textColor={'white'} 
            backgroundColor={aquaBlue} 
            topOffset={`${nodeTopOffset}%`} 
            leftOffset={`${nodeLeftOffset}%`}
            text={nodeText}
            onClickFunction={defaultFunction}
            funcParam={nodeText}/>
    )
}

export default BlueNode;