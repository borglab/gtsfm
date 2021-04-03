import React, {useRef} from "react";
import {useThree} from "react-three-fiber";

//Component which enables the user to pan and orbit around a point cloud
const OrbitControlsComponent = () => {
    const {
      camera,
      gl: {domElement}
    } = useThree();

    const controls = useRef();

    return (
      <>
        <orbitControls 
            args={[camera, domElement]} 
            ref={controls} 
            />
      </>
    )
}

export default OrbitControlsComponent;