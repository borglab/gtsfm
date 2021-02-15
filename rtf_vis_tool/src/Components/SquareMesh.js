import React from "react";
import {DoubleSide} from 'three';

//Unused - Doesn't face Camera
const SquareMesh = (props) => {
    return (
      <mesh position={props.position}>
        <planeBufferGeometry attach='geometry' args={props.widthArgs}/> 
        <meshBasicMaterial attach='material' color={props.color} side={DoubleSide}/>
      </mesh>
    )
}

export default SquareMesh;