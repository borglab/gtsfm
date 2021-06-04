/* Component which adjusts the radius for all points within a point cloud. Used within 
Bundle_Adj_PC.js.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import '.././stylesheets/PointSizeSlider.css';

function PointSizeSlider(props) {
    /*
    Args:
        props.pointRadius (int): Radius for each point in point cloud.
        props.setPointRadius (function): Updates the state value's point radius. 
        props.updatePointSizes (function): Updates the point radius for all points in cloud.
        
    Returns:
        A point size slider component to dynamically change radius of point cloud points.
    */

    return (
        <div className="point_size_slider_container">
            <p>Adjust Point Radius:</p>
            <input
                type="range"
                min="0.05"
                max="0.25"
                value={props.pointRadius} 
                onChange={(e) => {
                    props.setPointRadius(e.target.value);
                }}
                step="0.05"/>
                
            <p className="slider_small_label">Small</p>
            <p className="slider_large_label">Large</p>
        </div>
    )
}

export default PointSizeSlider;