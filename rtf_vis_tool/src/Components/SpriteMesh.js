import React from "react";
import * as THREE from "three/build/three";

//Component
const SpriteMesh = (props) => {
    return (
      <sprite
        position={props.position} 
        material={new THREE.SpriteMaterial({color: props.color})}
        center={[0.8,0.8]} //rotates about the center
        scale={props.widthArgs} // [width, height]
        />
    )
}

export default SpriteMesh;