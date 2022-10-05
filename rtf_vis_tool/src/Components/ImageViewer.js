/*
Author: Hayk Stepanyan
*/
import React, {useEffect, useState} from "react";

import '../stylesheets/ImageViewer.css';
import imageShapes from '../images/image_shapes.json';


function ImageViewer(props) {

    function importAll(r) {
        let images = {};
        r.keys().map((item, index) => { images[item.replace('./', '')] = r(item); });
        return images;
      }
      
    const images = importAll(require.context('../images', false, /\.(png|jpe?g|svg)$/));
    const imageFileNames = Object.keys(images);

    const imageFiles = []
    for (let i = 0; i < imageFileNames.length; i++) {
        imageFiles.push(
            <div className="image">
                <img src={images[imageFileNames[i]]} />
                <p>Shape: ({imageShapes[i][0]}, {imageShapes[i][1]})</p>
            </div>
        )
    }

    return (
        <div className="pc_container">
            <h2>{props.title}</h2>
            {imageFiles}
            <button className="pc_go_back_btn" onClick={() => props.togglePC(false)}>Go Back</button>
        </div>
    )
}

export default ImageViewer;
