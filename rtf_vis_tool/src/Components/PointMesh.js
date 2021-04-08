/* Spherical Point Component. Defined by position vector, radius, and color.

Author: Adi Singh
*/
import React from "react";

function PointMesh(props) {
    /*
    Args:
        props.position (array): Array, size 3, representing the 3D coordinate location of point mesh.
        props.size (int): Radius for a point mesh.
        props.color (string): Hex string color of point mesh.
        
    Returns:
        A spherical point mesh.
    */
   
    return (
      <mesh position={props.position}>
        <sphereBufferGeometry attach='geometry' args={props.size} />
        <meshStandardMaterial attach='material' color={props.color}/>
      </mesh>
    )
}

export default PointMesh;