import React, {useEffect, useState, useRef} from "react";
import {Canvas, extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import '../stylesheets/Data_Association_PC.css';

import SpriteMesh from './SpriteMesh';
import CoordinateGrid from './CoordinateGrid';
import OrbitControlsComponent from './OrbitControlsComponent';
import PointMesh from './PointMesh';


extend({OrbitControls})

const Data_Association_PC = (props) => {
    const [pointCloud, setPointCloud] = useState([]);

    //render points from data association json file (passed in through props)
    useEffect(() => {
        var finalPointsJSX = [];
        const scale = 1;
        for (var i = 0; i < props.json.length; i += 1) {
            //const point = props.json[i];

            //swap Y and Z to convert file's Z coords into RTF's Y coords

            // finalPointsJSX.push(<SpriteMesh  position={[scale*point[0], scale*point[2], scale*point[1]]}  
            //     widthArgs={[0.1,0.1]} 
            //     color={`rgb(0, 0, 0)`} />);

            finalPointsJSX.push(<PointMesh 
                position={[scale*props.json[i][0], scale*props.json[i][2], scale*props.json[i][1]]} 
                color={`rgb(163, 168, 165)`}
                size={[0.35,8,8]}
            />)

            finalPointsJSX.push()
        }
        setPointCloud(finalPointsJSX);
    }, []);

    return (
        <div className="da-container">
            <h2>Data Association Point Cloud</h2>
            <Canvas colorManagement camera={{ fov: 20, position: [50, 50, 50]}}>
                <ambientLight intensity={0.5}/>
                {/* <pointLight position={[0, 0, 20]} intensity={0.5}/>  */}
                <pointLight position={[100, 100, 100]} intensity={1} castShadow />
                <pointLight position={[-100, -100, -100]} intensity={0.8}/>
                <directionalLight 
                    position={[0,20,0]} 
                    intensity={1.5} 
                    shadow-mapSize-width={1024}
                    shadow-mapSize-height={1024}
                    shadow-camera-far={50}
                />
        
                {pointCloud}
                <CoordinateGrid />
                <OrbitControlsComponent />
            </Canvas>

            <button className="da_go_back_btn" onClick={() => props.toggleDA_PC(false)}>Go Back</button>
        </div>
    )
}

export default Data_Association_PC;