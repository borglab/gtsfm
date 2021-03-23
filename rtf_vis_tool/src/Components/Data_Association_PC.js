import React, {useEffect, useState, useRef} from "react";
import {Canvas, extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import '../stylesheets/Data_Association_PC.css';

//Loading Components
import CoordinateGrid from './CoordinateGrid';
import OrbitControlsComponent from './OrbitControlsComponent';
import PointMesh from './PointMesh';
import SpriteMesh from './SpriteMesh';

//allows user to orbit and pan around the point cloud
extend({OrbitControls})

//point cloud renderer component in the 'SfMData' Node
const Data_Association_PC = (props) => {
    const [pointCloud, setPointCloud] = useState([]);

    //render points from data association json file (passed in through props)
    useEffect(() => {
        var finalPointsJSX = [];
        for (var i = 0; i < props.json.length; i += 1) {
            const x_angle = (Math.PI/180) * 55;
            const y_angle = (Math.PI/180) * 30;
            const x_mod = Math.cos(y_angle)*props.json[i][0] + (Math.sin(y_angle)*Math.sin(x_angle))*props.json[i][1] + (Math.sin(y_angle)*Math.cos(x_angle))*props.json[i][2]; 
            const y_mod = Math.cos(x_angle)*props.json[i][1] - Math.sin(x_angle)*props.json[i][2]
            const z_mod = -1*Math.sin(y_angle)*props.json[i][0] + (Math.cos(y_angle)*Math.sin(x_angle))*props.json[i][1] + (Math.cos(y_angle)*Math.cos(x_angle))*props.json[i][2]; 

            finalPointsJSX.push(<PointMesh 
                position={[x_mod, z_mod, y_mod]} 
                color={`rgb(163, 168, 165)`}
                size={[0.2,8,8]}
            />)
        }

            // end of test points

        setPointCloud(finalPointsJSX);
    }, []);

    return (
        <div className="da-container">
            <h2>Data Association Point Cloud</h2>
            <Canvas colorManagement camera={{ fov: 20, position: [50, 50, 50]}}>
                <ambientLight intensity={0.5}/>
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