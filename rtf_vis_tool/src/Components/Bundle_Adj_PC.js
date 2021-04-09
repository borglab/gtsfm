/* Component that renders the Point Cloud and Camera Frustums before Bundle Adjustment. Spawned 
once the 'SfMData' node is clicked.

Author: Adi Singh
*/
import React, {useEffect, useState} from "react";

// Third-Party Package Imports.
import {Canvas} from "react-three-fiber"; // Used to render canvas containing 3D elements

// Local Imports.
import AllFrustums from './AllFrustums';
import CoordinateGrid from './CoordinateGrid';
import OrbitControlsComponent from './OrbitControlsComponent';
import PointMesh from './PointMesh';
import PointSizeSlider from './PointSizeSlider';
import '../stylesheets/Bundle_Adj_PC.css';

function Bundle_Adj_PC(props) {
    /*
    Args:
        props.toggleBA_PC (function): toggles the display of the Bundle Adjustment Point Cloud.
        
    Returns:
        A component rendering the point cloud after bundle adjustment.
    */

    // Variables to store the point cloud information and toggle the coordinate grid display.
    const [pointCloudRaw, setPointCloudRaw] = useState([]);
    const [pointCloudJSX, setPointCloudJSX] = useState([]);
    const [showCoordGrid, setShowCoordGrid] = useState(true);

    // Point size defined by radius, initialized as 0.15.
    const [pointRadius, setPointRadius] = useState(0.15);
    const pointSizeArr = [pointRadius]; 

    // Render points3D.txt from COLMAP ba_input directory.
    useEffect(() => {
        // Fetch the COLMAP file from the public directory.
        fetch('results/ba_input/points3D.txt')
            .then((response) => {
                return response.text();
            })
            .then((data) => loadCOLMAPPointCloud(data))
    }, []);

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

        /* Variable pointCloudRaw is an (N x 6) array, with the first 3 entries as (x,y,z) and the last
           3 entries as (R,G,B). */
        const arrNumPoints = arrStringPoints.map(point => point.split(" ").map(Number));
        setPointCloudRaw(arrNumPoints);


        // Loop through array. convert strings to numbers. Append to final point cloud.
        for (var index = 0; index < arrNumPoints.length; index += 1) {
            var pointArr = arrNumPoints[index];
                    
            finalPointsJSX.push(
                <PointMesh  
                    position={[pointArr[1], pointArr[2], pointArr[3]]}  
                    color={`rgb(${pointArr[4]}, ${pointArr[5]}, ${pointArr[6]})`} 
                    size={pointSizeArr}/>
                );
        }
        setPointCloudJSX(finalPointsJSX);
    }

    function updatePointSizes(radius) {
        /*Updates the radius of all points within a point cloud. Called everytime the react 
        slider input is interacted with.

        Args:
            radius (int): New radius for all points in point cloud.
        */

        var finalPointsJSX = [];
        for (var i = 0; i < pointCloudRaw.length; i += 1) {
            var pointArr = pointCloudRaw[i];
            
            finalPointsJSX.push(
                <PointMesh  
                    position={[pointArr[1], pointArr[2], pointArr[3]]}  
                    color={`rgb(${pointArr[4]}, ${pointArr[5]}, ${pointArr[6]})`} 
                    size={[radius]}/>
            );
        }
        setPointCloudJSX(finalPointsJSX);
    }

    return (
        <div className="ba-container">
            <h2>Bundle Adjustment Point Cloud</h2>
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
                <AllFrustums/>
            </Canvas>

            <button className="ba_go_back_btn" onClick={() => props.toggleBA_PC(false)}>Go Back</button>
            <button className="toggle_grid_btn" onClick={() => setShowCoordGrid(!showCoordGrid)}>
                Toggle Coordinate Grid
            </button>
            <PointSizeSlider pointRadius={pointRadius} setPointRadius={setPointRadius} updatePointSizes={updatePointSizes}/>
        </div>
    )
}

export default Bundle_Adj_PC;