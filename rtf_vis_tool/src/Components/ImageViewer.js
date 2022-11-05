/*
Author: Hayk Stepanyan
*/
import React from "react";

import '../stylesheets/ImageViewer.css';
import imageShapes from '../images/image_shapes.json';


function ImageViewer(props) {

    // import image files
    function importAll(r) {
        let images = {};
        r.keys().map((item, index) => { images[item.replace('./', '')] = r(item); });
        return images;
      }
    const images = importAll(require.context('../images', false, /\.(png|jpe?g|svg)$/));
    const imageFileNames = Object.keys(images);

    // sort file names
    imageFileNames.sort((a, b) => {
        if (a < b)
            return -1;
        if (a > b)
            return 1;
        return 0;
    });

    const imageFiles = []
    for (let i = 0; i < imageFileNames.length; i++) {
        imageFiles.push(
            <div className="image-container">
                <img src={images[imageFileNames[i]]} alt=""/><br></br>
                Image {i}<br></br>
                Shape: ({imageShapes[i]["shape"][0]}, {imageShapes[i]["shape"][1]})<br></br>
                Focal length: {imageShapes[i]["focal_length"].toFixed(2)}
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
