/* Component that renders all of the Frustums using the files: cameras.txt and images.txt. Will be rendered within
PCViewer.js

Author: Adi Singh
*/
import React, {useEffect, useState} from "react";
import Frustum from './Frustum';

// Third-Party Package Imports.
var nj = require('numjs');
var Quaternion = require('quaternion');

// Local Imports.
var SE3 = require('./frustum_classes/se3');
var ViewFrustum = require('./frustum_classes/view_frustum');

function AllFrustums(props) {
    /* 
    Returns:
        A component rendering all the camera frustums.
    */
    const [frustumsJSX, setFrustumsJSX] = useState([]);

    // Load camera intrinsics and extrinsics by fetching cameras.txt and images.txt respectively.
    useEffect(() => {
        
        // Fetch camera intrinsics from cameras.txt.
        fetch(`results/${props.pointCloudType}/cameras.txt`)
            .then((response) => response.text())
            .then((in_data) => {

                // Fetch camera extrinsics from images.txt.
                fetch(`results/${props.pointCloudType}/images.txt`)
                .then((response) => response.text())
                .then((ex_data) => {
                    visualize_camera_frustums(in_data, ex_data)
                });
            });
    });

    function visualize_camera_frustums(in_data, ex_data) {
        /* Given all the intrinsic and extrinsic data for all cameras, creates an array of frustums to be 
        rendered in react three fiber.

        Args:
            in_data (string): Raw string from cameras.txt, which contains instrinsic parameters of all reconstructed
                              cameras as per the COLMAP output convention. 
                              (https://colmap.github.io/format.html#cameras-txt)
            ex_data (string): Raw string from images.txt, which contains pose and keypoints of all reconstructed images
                              as per the COLMAP output convention.
                              (https://colmap.github.io/format.html#images-txt)
        */

        var in_cameraList = in_data.split('\n');
        in_cameraList = in_cameraList.slice(3);  // Remove the first 3 lines of comments in cameras.txt.
        in_cameraList.pop()  // Remove the last empty string from list.
        
        var ex_cameraList = ex_data.split('\n');
        ex_cameraList = ex_cameraList.slice(4); // Remove the 4 lines of comments in images.txt.
        ex_cameraList.pop() // remove the last empty string from list.
        ex_cameraList = ex_cameraList.filter(line => line !== "TODO"); // remove any lines that say 'TODO' from images.txt

        // remove any dummy lines from images.txt that contain the string "TODO", until gtsfm.utils.io.write_images is
        // updated
        ex_cameraList = ex_cameraList.filter(line => line !== "TODO"); 

        if (in_cameraList.length !== ex_cameraList.length) {
            alert('Camera count mismatch between images.txt and cameras.txt');
            return;
        }

        var finalFrustumsJSX = [];
        // Combine information from intrinsics and extrinsics to render frustums.
        for (var i = 0; i < in_cameraList.length; i++) {
            var inCamArray = in_cameraList[i].split(" ");
            var exCamArray = ex_cameraList[i].split(" ");

            if (inCamArray[0] !== exCamArray[0]) {
                alert("Error: Cam IDs in Intrinsic and Extrinsic Files Don't Match.");
                break;
            }

            // Set the important variables from inCamArray and exCamArray
            var fx, img_w, img_h;
            inCamArray = inCamArray.map(Number);
            exCamArray = exCamArray.map(Number);

            fx = inCamArray[4];
            img_w = inCamArray[2];
            img_h = inCamArray[3];

            const [qw, qx, qy, qz] = [exCamArray[1], exCamArray[2], exCamArray[3], exCamArray[4]];
            const [tx, ty, tz] = [exCamArray[5], exCamArray[6], exCamArray[7]]
            const DEFAULT_FRUSTUM_RAY_LENGTH = 0.5;  //meters, arbitrary

            // Use Frustum and SE3 Classes to obtain frustum vertices in world frame coordinates.
            const frustumObj = new ViewFrustum(fx, img_w, img_h, DEFAULT_FRUSTUM_RAY_LENGTH);
            const q = new Quaternion(qw, qx, qy, qz);
            const rotation_matrix = q.toMatrix(true);
            const rotation = nj.array(rotation_matrix);
            const translation = nj.array([tx,ty,tz]);
            const cTw = new SE3(rotation, translation);
            const wTc = cTw.inverse();
            var verts_worldframe = frustumObj.get_mesh_vertices_worldframe(wTc);

            finalFrustumsJSX.push(<Frustum 
                v0={verts_worldframe[0][0]}
                v1={verts_worldframe[1][0]}
                v2={verts_worldframe[2][0]}
                v3={verts_worldframe[3][0]}
                v4={verts_worldframe[4][0]}
                width={0.5}
                />)
        }
        setFrustumsJSX(finalFrustumsJSX);
    }

    return(frustumsJSX);
}

export default AllFrustums;