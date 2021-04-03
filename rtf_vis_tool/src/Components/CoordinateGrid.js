import React from "react";
import {Line, Html} from "drei";

//Function that creates the grid lines along with xy plane
const getGridLines = () => {
    var coordPairSet = []
    //add x grid lines
    for (var z = -10; z <= 10; z++) {
      coordPairSet.push([[-10, 0, z], [10, 0, z]]);
    }
    //add z grid lines
    for (var x = -10; x <= 10; x++) {
      coordPairSet.push([[x, 0, -10], [x, 0, 10]]);
    }

    var finalLineGridSet = []
    coordPairSet.map(coordPair => {
      finalLineGridSet.push(<Line points={coordPair} color="black" position={[0,0,0]} lineWidth={0.1}/>);
    });
    return finalLineGridSet;
}

//Component which renders the x,y,z axes along with the grid lines along xy plane
const CoordinateGrid = () => { 
    return (
      <mesh>
          <Line 
            points={[[-10,0,0], [10,0,0]]} 
            color="red" 
            position={[0,0,0]}
            lineWidth={0.3}
          />
          <Html scaleFactor={50} position={[10,0,0]}>
            <div class="content">
              x
            </div>
          </Html>

          <Line 
            points={[[0,-10,0], [0,10,0]]} 
            color="blue" 
            position={[0,0,0]}
            lineWidth={0.3}
          />
          <Html scaleFactor={50} position={[0,10,0]}>
            <div class="content">
              z
            </div>
          </Html>

          <Line 
            points={[[0,0,-10], [0,0,10]]} 
            color="green" 
            position={[0,0,0]}
            lineWidth={0.3}
          />
          <Html scaleFactor={50} position={[0,0,10]}>
            <div class="content">
              y
            </div>
          </Html>

          {getGridLines()}
      </mesh>
    )
}

export default CoordinateGrid;