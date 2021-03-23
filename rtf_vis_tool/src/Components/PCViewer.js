import React, {useEffect, useState, useRef} from "react";
import {Canvas, extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import '.././stylesheets/PCViewer.css'

//Loading Components
import CoordinateGrid from './CoordinateGrid';
import OrbitControlsComponent from './OrbitControlsComponent';
import RenderFrustum from './RenderFrustum';
import SpriteMesh from './SpriteMesh';

//Loading SE3 classes for camera frustum rendering
var nj = require('numjs');
var Quaternion = require('quaternion');
var ViewFrustum = require('.././ViewFrustum_Ported/view_frustum.js');
var SE3 = require('.././ViewFrustum_Ported/se3.js');

extend({OrbitControls})

const PCViewer = (props) => {
    const [rawFileString, setRawFileString] = useState("");
    const [rawIntrinsicString, setRawIntrinsicString] = useState("");
    const [rawExtrinsicString, setRawExtrinsicString] = useState("");
    const [pointCloud, setPointCloud] = useState([]);
    const [frustums, setFrustums] = useState([]);
    const [numCams, setNumCams] = useState(0);

    //Function: reads an inputted ASCII point cloud file and saves its values
    const readPointsFile = (e) => {
        e.preventDefault();
        const reader = new FileReader();
        reader.readAsText(e.target.files[0])
        reader.onload = (e) => {
        const fileString = e.target.result;
        setRawFileString(fileString);
        }
    }

    //Function: reads a camera intrinsics file (from COLMAP) and saves its values
    const readIntrinsicsFile = (e) => {
        e.preventDefault();
        const reader = new FileReader();
        reader.readAsText(e.target.files[0]);
        reader.onload = (e) => {
        const fileString = e.target.result;
        setRawIntrinsicString(fileString);
        }
    }

    //Function: reads a camera extrinsics file (from COLMAP) and saves its values
    const readExtrinsicsFile = (e) => {
        e.preventDefault();
        const reader = new FileReader();
        reader.readAsText(e.target.files[0]);
        reader.onload = (e) => {
        const fileString = e.target.result;
        setRawExtrinsicString(fileString);
        }
    }

    //Function: uses intrinsics and extrinsics to render camera frustums in React Three Fiber
    const visualize_camera_frustums = () => {
        //load intrinsics file
        const startPointIndex = rawIntrinsicString.indexOf("# Number of cameras:");
        const condensedString = rawIntrinsicString.substring(startPointIndex);
        var in_cameraList = condensedString.split('\n');
        in_cameraList.shift();

        //load extriniscs file
        var ex_cameraList = rawExtrinsicString.split('\n');
        ex_cameraList = ex_cameraList.slice(4);
        ex_cameraList = ex_cameraList.filter((value, index) => (index % 2 == 0));

        var finalFrustumsJSX = [];
        // combine intrinsics + extrinsics to render frustums
        for (var i = 0; i < numCams; i++) {
            var inCamArray = in_cameraList[i].split(" ");
            var exCamArray = ex_cameraList[i].split(" ");

            //IMPORTANT VARIABLES
            var fx, img_w, img_h, qw, qx, qy, qz, tx, ty, tz;

            if (inCamArray[0] !== exCamArray[0]) {
                alert("Error: Cam IDs in Intrinsic and Extrinsic Files Don't Match.");
                break;
            }

            inCamArray = inCamArray.slice(2);
            inCamArray = inCamArray.map(Number);
            exCamArray = exCamArray.map(Number);

            img_w = inCamArray[0];
            img_h = inCamArray[1];
            fx = inCamArray[2];
            qw = exCamArray[1];
            qx = exCamArray[2];
            qy = exCamArray[3];
            qz = exCamArray[4];
            tx = exCamArray[5];
            ty = exCamArray[6];
            tz = exCamArray[7];

            //frustum + se3 creating and rendering
            var frustumObj = new ViewFrustum(fx, img_w, img_h);
            var q = new Quaternion(qw, qx, qy, qz) //w,x,y,z
            var rotation_matrix = q.toMatrix(true);
            var rotation = nj.array(rotation_matrix);
            var translation = nj.array([tx, ty, tz]);
            var wTc = new SE3(rotation, translation);
            var verts_worldfr = frustumObj.get_mesh_vertices_worldframe(wTc);
            
            finalFrustumsJSX.push(<RenderFrustum v0={[verts_worldfr[0].get(0,0),verts_worldfr[0].get(0,1),verts_worldfr[0].get(0,2)]} 
                v1={[verts_worldfr[1].get(0,0),verts_worldfr[1].get(0,1),verts_worldfr[1].get(0,2)]} 
                v2={[verts_worldfr[2].get(0,0),verts_worldfr[2].get(0,1),verts_worldfr[2].get(0,2)]} 
                v3={[verts_worldfr[3].get(0,0),verts_worldfr[3].get(0,1),verts_worldfr[3].get(0,2)]} 
                v4={[verts_worldfr[4].get(0,0),verts_worldfr[4].get(0,1),verts_worldfr[4].get(0,2)]}
                width={0.5}/>);
        }
        setFrustums(finalFrustumsJSX);
    }
    
    //Function: reads a point cloud file (from COLMAP) and saves its values
    const visualize_3D_Points_File = () => {
        var arrStringPoints = rawFileString.split('\n');
        
        var finalPointsJSX = [];
        const scale = 1;
        for (var i = 0; i < arrStringPoints.length; i += 1) {
        var pointArr = arrStringPoints[i].split(" ").map(Number);

        //swap Y and Z to convert file's Z coords into RTF's Y coords
        finalPointsJSX.push(<SpriteMesh  position={[scale*pointArr[1], scale*pointArr[3], scale*pointArr[2]]}  
            widthArgs={[0.1,0.1]} 
            color={`rgb(${pointArr[4]}, ${pointArr[5]}, ${pointArr[6]})`} />);
        }
        setPointCloud(finalPointsJSX);
    }

    return (
        <div className="app-container">
            <div className="upload-container">
  
            <div className="pointcloud-upload-container">
                <p className="ply-instructions">Point Cloud Upload</p>
                <input type="file" name="ply-file-reader" onChange={(e) => readPointsFile(e)}/>
                <button className="file-submit-btn" onClick={visualize_3D_Points_File}>Visualize Point Cloud</button>
            </div>
  
            <div className="frustum-upload-container">
                <p className="ply-instructions">Frustum Upload</p>
    
                <div style={{display:'flex', flexDirection: 'row', alignItems: 'center'}}>
                    <p style={{marginRight: '5px', fontSize: '90%'}}>Intrinsics</p>
                    <input type="file" onChange={(e) => readIntrinsicsFile(e)}/>
                </div>
                
                <div style={{display:'flex', flexDirection: 'row', alignItems: 'center'}}>
                    <p style={{marginRight: '5px', fontSize: '90%'}}>Extrinsics</p>
                    <input type="file" onChange={(e) => readExtrinsicsFile(e)}/>
                </div>
    
                <div className="num-cams-container">
                    <p style={{marginRight: '5px'}}>No. of Cameras:</p>
                    <input style={{height: '20px', width: '60px'}} type="number" value={numCams} onChange={(e) => setNumCams(e.target.value)}/>
                </div>
    
                <button className="frustums-submit-btn" onClick={visualize_camera_frustums}>
                    Visualize Camera Frustums
                </button>
            </div>
        </div>
  
        <Canvas colorManagement camera={{ fov: 30, position: [50, 50, 50]}}>
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
          {frustums}
          <CoordinateGrid />
          <OrbitControlsComponent />
        </Canvas>
      </div>
    )
}

export default PCViewer;