import './App.css';
import React, {useState} from "react";
import {Canvas} from "react-three-fiber";
import {OrbitControls, Line, Html} from "drei";
import FileReaderInput from 'react-file-reader-input';

const App = (props) => {
  const [rawFileString, setRawFileString] = useState("");
  const [pointCloud, setPointCloud] = useState([]);

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
  const readFile = (e) => {
    e.preventDefault();
    const reader = new FileReader();
    reader.readAsText(e.target.files[0])
    reader.onload = (e) => {
      const fileString = e.target.result;
      setRawFileString(fileString);
    }
  }

    //Function ... in progress 
    const readBinaryFile = (e) => {
      e.preventDefault();
      const reader = new FileReader();
      reader.readAsArrayBuffer(e.target.files[0])
      reader.onload = (e) => {
        const fileString = e.target.result;
        console.log(fileString)
      }
    }

  //Function
  const visualizeFile = () => {
      const startPointsIndex = rawFileString.indexOf("end_header")
      const condensedString = rawFileString.substring(startPointsIndex)
      var arrStringPoints = condensedString.split('\n')
      arrStringPoints.shift()
      console.log(arrStringPoints.length)

      var finalPointsJSX = []   
      const scale = 10;   
      for (var i = 0; i < arrStringPoints.length; i += 1000) {
        var arrStringElementsOfPoint = arrStringPoints[i].split(" ");
        var pointArr = arrStringElementsOfPoint.map(Number);
        pointArr = swapYZ(pointArr)
        
        const colorString = `rgb(${pointArr[3]}, ${pointArr[4]}, ${pointArr[5]})`
        finalPointsJSX.push(<PointMesh  position={[scale*pointArr[0], scale*pointArr[1], scale*pointArr[2]]} size={[0.05,16,16]} color={colorString}/>);
      }
      setPointCloud(finalPointsJSX);
  }


  return (
    <div className="app-container">
      <div className="upload-container">
        <p className="ply-instructions">Please Upload a PLY File</p>

        <input type="file" name="ply-file-reader" onChange={(e) => readFile(e)}/>

        <button className="file-submit-btn" onClick={visualizeFile}>Visualize Point Cloud</button>
      </div>

      <Canvas colorManagement camera={{position: [30,30,30], fov: 30}}>
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
