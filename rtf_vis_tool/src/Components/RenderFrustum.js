import React from "react";
import LineMesh from './LineMesh'

//Function: swaps Y and Z coordinates of a point to suit RTF coordinate system
const swapYZ = (coords) => {
    const tempY = coords[1]
    coords[1] = coords[2]
    coords[2] = tempY
    return coords
}

//Component
const RenderFrustum = ({v0, v1, v2, v3, v4, width}) => {
    var v0_mod = swapYZ(v0);
    var v1_mod = swapYZ(v1);
    var v2_mod = swapYZ(v2);
    var v3_mod = swapYZ(v3);
    var v4_mod = swapYZ(v4);

    return (
      <mesh>
        <LineMesh points={[v0_mod,v1_mod]} color="#000000" width={width}/>
        <LineMesh points={[v0_mod,v2_mod]} color="#000000" width={width}/>
        <LineMesh points={[v0_mod,v3_mod]} color="#000000" width={width}/>
        <LineMesh points={[v0_mod,v4_mod]} color="#000000" width={width}/>
        <LineMesh points={[v1_mod,v2_mod]} color="#000000" width={width}/>
        <LineMesh points={[v2_mod,v3_mod]} color="#000000" width={width}/>
        <LineMesh points={[v3_mod,v4_mod]} color="#000000" width={width}/>
        <LineMesh points={[v4_mod,v1_mod]} color="#000000" width={width}/>
      </mesh>
    )
}

export default RenderFrustum;