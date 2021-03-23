import React from "react";
import {DoubleSide} from 'three';

//Square Points - don't always face camera
const SquareMesh = (props) => {
    return (
      <mesh position={props.position}>
        <planeBufferGeometry attach='geometry' args={props.widthArgs}/> 
        <meshBasicMaterial attach='material' color={props.color} side={DoubleSide}/>
      </mesh>
    )
}

export default SquareMesh;