/* Component which renders the x,y,z axes along with the grid lines along the xy plane. Used within
Data_Association_PC.js.

Author: Adi Singh
*/
import React from "react";

// Third-Party Package Imports.
import {Line, Html} from "drei";  // Used to render axes lines and text.

// Create grid lines along the xy plane.
const getGridLines = (lowerGridBound = -10, upperGridBound = 10) => {
    var coordPairSet = []

    // Add grid lines parallel to x axis.
    for (var y = lowerGridBound; y <= upperGridBound; y++) {
      coordPairSet.push([[lowerGridBound, y, 0], [upperGridBound, y, 0]]);
    }
    // Add grid lines parallel to y axis.
    for (var x = lowerGridBound; x <= upperGridBound; x++) {
      coordPairSet.push([[x, lowerGridBound, 0], [x, upperGridBound, 0]]);
    }

    var finalLineGridSet = []
    coordPairSet.map(coordPair => {
      finalLineGridSet.push(<Line 
                              points={coordPair} 
                              color="black" 
                              position={[0,0,0]} 
                              lineWidth={0.1}/>);
    });
    return finalLineGridSet;
}

const CoordinateGrid = () => { 
  const lowerGridBound = -10;   // Lower grid bound for coordinate axes.
  const upperGridBound = 10;    // Upper grid bound for coordinate axes.
  const htmlSize = 50;          // Size of the x,y,z axes labels.
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