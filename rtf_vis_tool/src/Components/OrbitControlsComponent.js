import React, {useRef} from "react";
import {useThree} from "react-three-fiber";

//Component
const OrbitControlsComponent = () => {
    const {
      camera,
      gl: {domElement}
    } = useThree();

    const controls = useRef();
      //Runs Every Frame
      // useFrame(({camera}) => {
      //   console.log([camera.position.x, camera.position.y, camera.position.z])
      //   camera.position.y += Math.random();
      // });

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