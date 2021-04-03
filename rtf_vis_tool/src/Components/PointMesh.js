import React from "react";

//Spherical Point Component
//Slower Rendering Speeds than a Sprite Component
const PointMesh = (props) => {
    return (
      <mesh position={props.position}>
        <sphereBufferGeometry attach='geometry' args={props.size} />
        <meshStandardMaterial attach='material' color={props.color}/>
      </mesh>
    )
}

export default PointMesh;