/* Component that renders the Point Cloud and Camera Frustums before Bundle Adjustment. Spawned 
once the 'SfMData' node is clicked.

Author: Adi Singh
*/
import React, {useEffect, useState} from "react";

// Third-Party Package Imports.
import {Canvas} from "react-three-fiber"; // Used to render canvas containing 3D elements

// Local Imports.
import CoordinateGrid from './CoordinateGrid';
import OrbitControlsComponent from './OrbitControlsComponent';
import PointMesh from './PointMesh';
import PointSizeSlider from './PointSizeSlider';
import '../stylesheets/PCViewer.css';

function PCViewer(props) {
    /*
    Args:
        props.title (string): Title of the Point Cloud Viewer Pop Up.
        props.toggleDA_PC (function): Toggles the display of the Bundle Adjustment Point Cloud.
        
    Returns:
        A component rendering the point cloud after bundle adjustment.
    */

    // Variables to store the point cloud information and toggle the coordinate grid display.
    const [pointCloudJSX, setPointCloudJSX] = useState([]);
    const [showCoordGrid, setShowCoordGrid] = useState(true);

    // Point size defined by radius, initialized as 0.15.
    const [pointRadius, setPointRadius] = useState(0.15);

    /* Render points3D.txt from COLMAP ba_input directory. Runs when component is first rendered and every time the
       pointRadius is updated.
    */
    useEffect(() => {
        // Fetch the COLMAP file from the public directory.
        fetch(props.filePath)
            .then((response) => {
                return response.text();
            })
            .then((data) => loadCOLMAPPointCloud(data))
    });

    function loadCOLMAPPointCloud(data) {
        /*Accepts the raw COLMAP points3D.txt file and converts it into an array of JSX formatted
        PointMeshes (which are then rendered on screen).

        Args:
            data (string): String of points directly from points3D.txt file. Each point is an 
                           array of length 7.
        */
        const arrStringPoints = data.split('\n');
        var finalPointsJSX = [];
                
        // Remove the first 3 commented lines of points3D.txt.
        for (var i = 0; i < 3; i++) {
            arrStringPoints.shift();
        }

        /* Variable arr_points is an (N x 6) array, with the first 3 entries as (x,y,z) and the last
           3 entries as (R,G,B). */
        const arr_points = arrStringPoints.map(point => point.split(" ").map(Number));

        // Loop through array. convert strings to numbers. Append to final point cloud.
        for (var index = 0; index < arr_points.length; index += 1) {
            var pointArr = arr_points[index];
                    
            finalPointsJSX.push(
                <PointMesh  
                    position={[pointArr[1], pointArr[2], pointArr[3]]}  
                    color={`rgb(${pointArr[4]}, ${pointArr[5]}, ${pointArr[6]})`} 
                    size={[pointRadius]}/>
                );
        }
        setPointCloudJSX(finalPointsJSX);
    }

    return (
        <div className="pc_container">
            <h2>{props.title}</h2>
            <Canvas colorManagement camera={{ fov: 20, position: [50, 50, 50], up: [0,0,1]}}>
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
        
                {pointCloudJSX}
                {showCoordGrid && <CoordinateGrid />}
                <OrbitControlsComponent />
            </Canvas>

            <button className="pc_go_back_btn" onClick={() => props.togglePC(false)}>Go Back</button>
            <button className="toggle_grid_btn" onClick={() => setShowCoordGrid(!showCoordGrid)}>
                Toggle Coordinate Grid
            </button>
            <PointSizeSlider pointRadius={pointRadius} setPointRadius={setPointRadius}/>
        </div>
    )
}

export default PCViewer;