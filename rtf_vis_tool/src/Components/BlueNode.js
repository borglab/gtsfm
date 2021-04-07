/* Blue Node Component. Built on top of GtsfmNode.js with styling properties specific to a Blue Node.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import GtsfmNode from './GtsfmNode';

const BlueNode = (props) => {
    const aquaBlue = '#2255e0';
    const nodeText = props.nodeInfo.text;
    const nodeTopOffset = props.nodeInfo.topOffset;
    const nodeLeftOffset = props.nodeInfo.leftOffset;

    return (
        <GtsfmNode 
            textColor={'white'} 
            backgroundColor={aquaBlue} 
            topOffset={`${nodeTopOffset}%`} 
            leftOffset={`${nodeLeftOffset}%`}
            text={nodeText}/>
    )
}

export default BlueNode;