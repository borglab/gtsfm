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
    const [isCentered, setIsCentered] = useState(false);
    const [showCoordGrid, setShowCoordGrid] = useState(true);

    //render points from data association json file (passed in through props)
    useEffect(() => {
        var finalPointsJSX = [];
        for (var i = 0; i < props.json.length; i += 1) {
            finalPointsJSX.push(<PointMesh 
                position={[props.json[i][0], -1*props.json[i][2], props.json[i][1]]} 
                color={`rgb(163, 168, 165)`}
                size={[0.2,8,8]}
            />)
        }

        setPointCloud(finalPointsJSX);
    }, []);

    //Function to center the point cloud with respect to the origin
    const centerPC = () => {
        if (isCentered) {
            alert('Point Cloud Already Centered');
            return;
        }
        setIsCentered(true);

        const length = props.json.length;
        var sumX = 0;
        var sumY = 0;
        var sumZ = 0;

        for (var i = 0; i < props.json.length; i += 1) {
            sumX += props.json[i][0];
            sumY += props.json[i][1];
            sumZ += props.json[i][2];
        }

        const meanX = sumX / length;
        const meanY = sumY / length;
        const meanZ = sumZ / length;

        var finalPointsJSX = [];
        for (var i = 0; i < props.json.length; i += 1) {
            finalPointsJSX.push(<PointMesh 
                position={[props.json[i][0] - meanX, props.json[i][2] - meanZ, props.json[i][1] - meanY]} 
                color={`rgb(163, 168, 165)`}
                size={[0.2,8,8]}
            />)
        }
        setPointCloud(finalPointsJSX);
    }

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
                {showCoordGrid && <CoordinateGrid />}
                <OrbitControlsComponent />
            </Canvas>

            <button className="da_go_back_btn" onClick={() => props.toggleDA_PC(false)}>Go Back</button>
            <button className="toggle_grid_btn" onClick={() => setShowCoordGrid(!showCoordGrid)}>Toggle Coordinate Grid</button>
            <button className="da_center_btn" onClick={centerPC}>Center</button>
        </div>
    )
}

export default Data_Association_PC;