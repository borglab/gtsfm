import './App.css';
import React, {useEffect, useState} from "react";
import {Canvas} from "react-three-fiber";
import {OrbitControls, Line, Html} from "drei";

const App = (props) => {
  const [pointCloud, setPointCloud] = useState([]);

  //Function
  const getGridLines = () => {
    var coordPairSet = []
    //add x grid lines
    for (var z = -50; z <= 50; z++) {
      coordPairSet.push([[-50, 0, z], [50, 0, z]]);
    }
    //add z grid lines
    for (var x = -50; x <= 50; x++) {
      coordPairSet.push([[x, 0, -50], [x, 0, 50]]);
    }

    var finalLineGridSet = []
    coordPairSet.map(coordPair => {
      finalLineGridSet.push(<Line points={coordPair} color="black" position={[0,0,0]} lineWidth={0.1}/>);
    });
    return finalLineGridSet;
  }

  //Component
  const PointMesh = ({position, size, color, label}) => {
    return (
      <mesh position={position}>
        <sphereBufferGeometry attach='geometry' args={size} />
        <meshStandardMaterial attach='material' color={color}/>

        <Html scaleFactor={30} position={[1,0.5,0]}>
          <div class="content">
            {label}
          </div>
        </Html>

      </mesh>
    )
  }

  //Component
  const CoordinateGrid = () => { 
    return (
      <mesh>
          <Line 
            points={[[-50,0,0], [50,0,0]]} 
            color="red" 
            position={[0,0,0]}
            lineWidth={0.3}
          />
          <Html scaleFactor={200} position={[50,0,0]}>
            <div class="content">
              x
            </div>
          </Html>

          <Line 
            points={[[0,-50,0], [0,50,0]]} 
            color="blue" 
            position={[0,0,0]}
            lineWidth={0.3}
          />
          <Html scaleFactor={200} position={[0,50,0]}>
            <div class="content">
              z
            </div>
          </Html>

          <Line 
            points={[[0,0,-50], [0,0,50]]} 
            color="green" 
            position={[0,0,0]}
            lineWidth={0.3}
          />
          <Html scaleFactor={200} position={[0,0,50]}>
            <div class="content">
              y
            </div>
          </Html>

          {getGridLines()}
      </mesh>
    )
  }

  //Function
  const swapXY = (coords) => {
    const tempY = coords[1]
    coords[1] = coords[2]
    coords[2] = tempY
    return coords
  }

  //Function
  const readFile = (e) => {
    e.preventDefault();
    const reader = new FileReader();
    reader.readAsText(e.target.files[0])
    reader.onload = (e) => {
      const pointString = e.target.result;
      const arrStringPoints = pointString.split('\n');
      var pointArr = [];
      for (var i = 0; i < arrStringPoints.length; i++) {
        pointArr.push(swapXY(JSON.parse(arrStringPoints[i])));
      }

      var finalPointsJSX = [];
      pointArr.map(point => {
        finalPointsJSX.push(<PointMesh  position={point} size={[0.3,16,16]} color='rgb(255, 0, 0)'/>);
      })
      setPointCloud(finalPointsJSX);
    }
  }

  return (
    <div className="app-container">
      <div className="upload-container">
        <input type="file" name="Input Points" onChange={(e) => readFile(e)}/>
      </div>

      <Canvas colorManagement camera={{position: [30,30,30], fov: 100}}>
        <ambientLight intensity={0.5}/>
        <pointLight position={[0, 0, 20]} intensity={0.5}/> 
        <directionalLight 
          position={[0,10,0]} 
          intensity={1.5} 
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
          shadow-camera-far={50}
        />

        {pointCloud}

        <CoordinateGrid />

        <OrbitControls/>
      </Canvas>
    </div>
  )
}

export default App;
