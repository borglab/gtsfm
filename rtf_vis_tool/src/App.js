import React, {useEffect, useState, useRef} from "react";
import {Canvas, extend, useThree, useFrame, Renderer} from "react-three-fiber";
import {Line, Html} from "drei";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import './App.css';
import {DoubleSide, Scene} from 'three';
import {PointCloudOctree, Potree} from '@pnext/three-loader';

extend({OrbitControls})

const App = (props) => {
  const [rawFileString, setRawFileString] = useState("");
  const [pointCloud, setPointCloud] = useState([]);
  const canvasRef = useRef();
  const cameraRef = useRef();
  
  //Function
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
  const SquareMesh = ({position, widthArgs, color}) => {
    return (
      <mesh position={position}>
        <planeBufferGeometry attach='geometry' args={widthArgs}/> 
        <meshBasicMaterial attach='material' color={color} side={DoubleSide}/>
      </mesh>
    )
  }

  //Component
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

  //Function
  const swapYZ = (coords) => {
    const tempY = coords[1]
    coords[1] = coords[2]
    coords[2] = tempY
    return coords
  }

  //Function
  const readPLYFile = (e) => {
    e.preventDefault();
    const reader = new FileReader();
    reader.readAsText(e.target.files[0])
    reader.onload = (e) => {
      const fileString = e.target.result;
      setRawFileString(fileString);
    }
  }

  //Function
  const visualizeFile = () => {
      const startPointsIndex = rawFileString.indexOf("end_header")
      const condensedString = rawFileString.substring(startPointsIndex)
      var arrStringPoints = condensedString.split('\n')
      arrStringPoints.shift()
      console.log(`Length: ${arrStringPoints.length}`);

      var finalPointsJSX = [];  
      const scale = 1;   
      for (var i = 0; i < arrStringPoints.length; i += 500) {
        var pointArr = arrStringPoints[i].split(" ").map(Number);   
        finalPointsJSX.push(<SquareMesh  position={[scale*pointArr[0], scale*pointArr[1], scale*pointArr[2]]} 
          widthArgs={[0.05,0.05]} 
          color={`rgb(${pointArr[3]}, ${pointArr[4]}, ${pointArr[5]})`}/>);
      }
      setPointCloud(finalPointsJSX);
  }

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


  return (
    <div className="app-container">
      <div className="upload-container">
        <p className="ply-instructions">Please Upload a PLY File</p>

        <input type="file" name="ply-file-reader" onChange={(e) => readPLYFile(e)} accept=".ply"/>

        <button className="file-submit-btn" onClick={visualizeFile}>Visualize Point Cloud</button>
      </div>

      <Canvas colorManagement camera={{ fov: 30, position: [50, 50, 50]}} ref={canvasRef}>
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

        <OrbitControlsComponent />
      </Canvas>
    </div>
  )
}

export default App;