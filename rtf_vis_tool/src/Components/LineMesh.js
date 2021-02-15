import React from "react";
import {Line} from "drei";

//Component
const LineMesh = (props) => {
    return (
      <mesh>
        <Line 
            points={props.points} 
            color={props.color}
            lineWidth={props.width}
          />
      </mesh>
    )
}

export default LineMesh;