/* Component which enables the user to pan and orbit around a point cloud.

Author: Adi Singh
*/
import React, {useRef} from "react";

// Third-Party Package Imports.
import {useThree} from "@react-three/fiber"; // Used to define a three.js camera object.

function OrbitControlsComponent() {
    /*
    Returns:
        A component to allow for orbit functionality.
    */

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