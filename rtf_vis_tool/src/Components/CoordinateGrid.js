import React from "react";
import {Line, Html} from "drei";

const lowerGridBound = -10;   //lower grid bound for coordinate axes
const upperGridBound = 10;    //upper grid bound for coordinate axes
const htmlSize = 50;          //size of the x,y,z axes labels

//Function that creates the grid lines along with xy plane
const getGridLines = () => {
    var coordPairSet = []

    //add grid lines parallel to x axis
    for (var z = lowerGridBound; z <= upperGridBound; z++) {
      coordPairSet.push([[lowerGridBound, 0, z], [upperGridBound, 0, z]]);
    }
    //add z grid lines
    for (var x = lowerGridBound; x <= upperGridBound; x++) {
      coordPairSet.push([[x, 0, lowerGridBound], [x, 0, upperGridBound]]);
    }

    var finalLineGridSet = []
    coordPairSet.map(coordPair => {
      finalLineGridSet.push(<Line points={coordPair} color="black" position={[0,0,0]} lineWidth={0.1}/>);
    });
    return finalLineGridSet;
}

// Component which renders the x,y,z axes along with the grid lines along the xy plane.
// Used within Data_Association_PC.js
//  This component returns the format JSX, which is essentially a modified version of HTML  
//  that is meant to be readable by React code. 
const CoordinateGrid = () => { 
    return (
      <mesh>
          <Line 
            points={[[lowerGridBound,0,0], [upperGridBound,0,0]]} 
            color="red" 
            lineWidth={0.3}
          />
          <Html scaleFactor={htmlSize} position={[upperGridBound,0,0]}>
            <div class="content">
              x
            </div>
          </Html>

          <Line 
            points={[[0,lowerGridBound,0], [0,upperGridBound,0]]} 
            color="blue" 
            lineWidth={0.3}
          />
          <Html scaleFactor={htmlSize} position={[0,upperGridBound,0]}>
            <div class="content">
              z
            </div>
          </Html>

          <Line 
            points={[[0,0,lowerGridBound], [0,0,upperGridBound]]} 
            color="green" 
            lineWidth={0.3}
          />
          <Html scaleFactor={htmlSize} position={[0,0,upperGridBound]}>
            <div class="content">
              y
            </div>
          </Html>

          {getGridLines()}
      </mesh>
    )
}

export default CoordinateGrid;