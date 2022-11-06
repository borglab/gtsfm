/*
Provides a gallary to view all the images, corresponding image shapes, and focal lengths.

Author: Hayk Stepanyan
*/

import React, {useState} from "react";
import ReactPaginate from 'react-paginate';

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
        a = parseInt(a.split(".")[0])
        b = parseInt(b.split(".")[0])
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

    // show images in different pages if the dataset is large
    const [currentPage, setCurrentPage] = useState(0);
    const PER_PAGE = 10;
    const offset = currentPage * PER_PAGE;
    const currentPageData = imageFiles.slice(offset, offset + PER_PAGE)
    const pageCount = Math.ceil(imageFiles.length / PER_PAGE);

    function handlePageClick({ selected: selectedPage }) {
        setCurrentPage(selectedPage);
    }

    return (
        <div className="pc_container">
            <h2>{props.title}</h2>
            <ReactPaginate
                previousLabel={"← Previous"}
                nextLabel={"Next →"}
                pageCount={pageCount}
                onPageChange={handlePageClick}
                containerClassName={"pagination"}
                previousLinkClassName={"pagination__link"}
                nextLinkClassName={"pagination__link"}
                disabledClassName={"pagination__link--disabled"}
                activeClassName={"pagination__link--active"}
            />
            {currentPageData}
            <button className="pc_go_back_btn" onClick={() => props.togglePC(false)}>Go Back</button>
        </div>
    )
}

export default ImageViewer;
