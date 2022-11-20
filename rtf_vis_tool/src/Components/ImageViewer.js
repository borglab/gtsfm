/*
Provides a gallary to view all the images, corresponding image shapes, and focal lengths.

Author: Hayk Stepanyan
*/

import React, {useState, useEffect} from "react";
import ReactPaginate from 'react-paginate';

import '../stylesheets/ImageViewer.css';
import imageShapes from '../images/image_shapes.json';


function ImageViewer(props) {

    // window size info for image ratio calculation
    const [windowDimensions, setWindowDimensions] = useState(getWindowDimensions());
    const [showFullImage, setShowFullImage] = useState(false);
    const [fullScreenImageIndex, setFullScreenImageIndex] = useState(0);
    const windowRatio = windowDimensions.width / windowDimensions.height

    // show images in different pages if the dataset is large
    const [currentPage, setCurrentPage] = useState(0);
    const PER_PAGE = 25;
    const offset = currentPage * PER_PAGE;

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

    function getWindowDimensions() {
        const {innerWidth: width, innerHeight: height} = window;
        return {width, height};
    }

    // import image files
    function importAll(r) {
        let images = {};
        r.keys().map((item, index) => { images[item.replace('./', '')] = r(item); });
        return images;
    }

    // move to different pages
    function handlePageClick({ selected: selectedPage }) {
        setCurrentPage(selectedPage);
    }
    
    useEffect(() => {
        function handleResize() {
            setWindowDimensions(getWindowDimensions());
        }

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);
    
    const imageFiles = []
    var averageFocalLength = 0
    for (let i = 0; i < imageFileNames.length; i++) {
        var ratio = imageShapes[i]["shape"][1] / imageShapes[i]["shape"][0] // image width / height ratio
        imageFiles.push(
            <div className="image-container">
                <img 
                    src={images[imageFileNames[i]]} alt=""
                    width={100 / windowRatio * ratio + "%"}
                    height="100%"
                    onClick={() => {
                            setShowFullImage(true);
                            setFullScreenImageIndex(i);
                        }
                    }
                />
            </div>
        )
        averageFocalLength = averageFocalLength + imageShapes[i]["focal_length"];
    }
    averageFocalLength = averageFocalLength / imageFiles.length;

    const currentPageData = imageFiles.slice(offset, offset + PER_PAGE)
    const pageCount = Math.ceil(imageFiles.length / PER_PAGE);

    return (
        <div className="pc_container">
            <div className="header">
                <div className="title">
                    {/* {props.title} */}
                    {"Average Focal Length: " + averageFocalLength.toFixed(2)}
                </div>
                <div className="paginationContainer">
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
                </div>
            </div>
            <div className="content">
                {currentPageData}
            </div>
            <button className="pc_go_back_btn" onClick={() => props.togglePC(false)}>Go Back</button>
            {showFullImage && 
                <div className="fullImage">
                    <div className="modal-content">
                        <p>
                            Image: {fullScreenImageIndex}<br />
                            Focal length: {imageShapes[fullScreenImageIndex]["focal_length"].toFixed(2)}<br />
                            Image shape: ({imageShapes[fullScreenImageIndex]["shape"][0]}, {imageShapes[fullScreenImageIndex]["shape"][1]})
                        </p>
                        <img src={images[imageFileNames[fullScreenImageIndex]]} alt="" />
                        <span class="close" onClick={() => {setShowFullImage(false);}}>&times;</span>
                    </div>
                </div>
            }
        </div>
    )
}

export default ImageViewer;
