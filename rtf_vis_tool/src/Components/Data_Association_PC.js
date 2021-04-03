import React, {useEffect, useState, useRef} from "react";
import {Canvas, extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import '../stylesheets/Data_Association_PC.css';

//Loading Components
import CoordinateGrid from './CoordinateGrid';
import OrbitControlsComponent from './OrbitControlsComponent';
import PointMesh from './PointMesh';

//allows user to orbit and pan around the point cloud
extend({OrbitControls})

//Point Cloud Renderer Component
//Spawned once the 'SfMData' node is clicked
const Data_Association_PC = (props) => {
    const [pointCloud, setPointCloud] = useState([]);
    const [isCentered, setIsCentered] = useState(false);
    const [showCoordGrid, setShowCoordGrid] = useState(true);
    const lightgray = `rgb(163, 168, 165)`;
    const pointSizeArr = [0.2,8,8];

    //render points_3d from data association json file (passed in through props)
    useEffect(() => {
        var finalPointsJSX = [];
        for (var i = 0; i < props.da_json.length; i += 1) {
            finalPointsJSX.push(<PointMesh 
                position={[props.da_json[i][0], props.da_json[i][2], props.da_json[i][1]]} 
                color={lightgray}
                size={pointSizeArr}
            />)
        }

        setPointCloud(finalPointsJSX);
    }, []);

    //Function to replace the original point cloud with a centered, aligned new point cloud
    //renders rotated_points_3d from data association file
    const alignPC = () => {
        var finalPointsJSX = [];
        for (var i = 0; i < props.rotated_json.length; i += 1) {
            finalPointsJSX.push(<PointMesh 
                position={[props.rotated_json[i][0], -1*props.rotated_json[i][2], props.rotated_json[i][1]]} 
                color={lightgray}
                size={pointSizeArr}
            />)
        }
        setPointCloud(finalPointsJSX);
    }

    //Function to center the point cloud with respect to the origin
    const centerPC = () => {
        if (isCentered) {
            alert('Point Cloud Already Centered');
            return;
        }
        setIsCentered(true);

        var arrX = [];
        var arrY = [];
        var arrZ = [];

        for (var i = 0; i < props.da_json.length; i += 1) {
            arrX.push(props.da_json[i][0]);
            arrY.push(props.da_json[i][1]);
            arrZ.push(props.da_json[i][2]);
        }

        const average = (array) => array.reduce((sum,el) => sum + el, 0) / array.length;
        const meanX = average(arrX);
        const meanY = average(arrY);
        const meanZ = average(arrZ);

        var finalPointsJSX = [];
        for (var i = 0; i < props.da_json.length; i += 1) {
            finalPointsJSX.push(<PointMesh 
                position={[props.da_json[i][0] - meanX, props.da_json[i][2] - meanZ, props.da_json[i][1] - meanY]} 
                color={lightgray}
                size={pointSizeArr}
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
            <button className="rotate_pc_btn" onClick={alignPC}>Align Point Cloud</button>
        </div>
    )
}

export default Data_Association_PC;