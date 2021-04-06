import React from "react";
import {Line, Html} from "drei";

const lowerGridBound = -10;   //lower grid bound for coordinate axes
const upperGridBound = 10;    //upper grid bound for coordinate axes
const htmlSize = 50;          //size of the x,y,z axes labels

//Function that creates the grid lines along with xy plane
const getGridLines = () => {
    var coordPairSet = []

    //add grid lines parallel to x axis
    for (var y = lowerGridBound; y <= upperGridBound; y++) {
      coordPairSet.push([[lowerGridBound, y, 0], [upperGridBound, y, 0]]);
    }
    //add z grid lines
    for (var x = lowerGridBound; x <= upperGridBound; x++) {
      coordPairSet.push([[x, lowerGridBound, 0], [x, upperGridBound, 0]]);
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
            color="green" 
            lineWidth={0.3}
          />
          <Html scaleFactor={htmlSize} position={[0,upperGridBound,0]}>
            <div class="content">
              y
            </div>
          </Html>

          <Line 
            points={[[0,0,lowerGridBound], [0,0,upperGridBound]]} 
            color="blue" 
            lineWidth={0.3}
          />
          <Html scaleFactor={htmlSize} position={[0,0,upperGridBound]}>
            <div class="content">
              z
            </div>
          </Html>

          {getGridLines()}
      </mesh>
    )
}

export default CoordinateGrid;