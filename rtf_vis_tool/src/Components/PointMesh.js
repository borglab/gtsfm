/* Spherical Point Component. Defined by position vector, radius, and color.

Author: Adi Singh
*/
import React from "react";

const PointMesh = (props) => {
    return (
      <mesh position={props.position}>
        <sphereBufferGeometry attach='geometry' args={props.size} />
        <meshStandardMaterial attach='material' color={props.color}/>
      </mesh>
    )
}

export default PointMesh;