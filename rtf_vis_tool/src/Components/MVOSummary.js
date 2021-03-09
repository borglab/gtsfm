import React from "react";
import '../stylesheets/MVOSummary.css'

const MVOSummary = (props) => {
    return (
        <div className="mvo_container">
            <h3>MultiView Optimizer Metrics</h3>
            <div style={{position: "relative", padding: '0', width: '100%', height: '100%', display: 'flex', flexDirection: 'row'}}>
                <div className="metrics_container">
                    <p className="mvo-header">Rotation Averaging Angle Metrics</p>
                    <p className="mvo-text">Median Error: {props.json.rotation_averaging_angle_deg.median_error.toFixed(5)}</p>
                    <p className="mvo-text">Min Error: {props.json.rotation_averaging_angle_deg.min_error.toFixed(5)}</p>
                    <p className="mvo-text">Max Error: {props.json.rotation_averaging_angle_deg.max_error.toFixed(5)}</p>
                </div>

                <div className="metrics_container">
                    <p className="mvo-header">Translation Averaging Metrics</p>
                    <p className="mvo-text">Median Error: {props.json.translation_averaging_distance.median_error.toFixed(5)}</p>
                    <p className="mvo-text">Min Error: {props.json.translation_averaging_distance.min_error.toFixed(5)}</p>
                    <p className="mvo-text">Max Error: {props.json.translation_averaging_distance.max_error.toFixed(5)}</p>
                </div>

                <div className="metrics_container">
                    <p className="mvo-header">Translation to Direction Angle Metrics</p>
                    <p className="mvo-text">Median Error: {props.json.translation_to_direction_angle_deg.median_error.toFixed(5)}</p>
                    <p className="mvo-text">Min Error: {props.json.translation_to_direction_angle_deg.min_error.toFixed(5)}</p>
                    <p className="mvo-text">Max Error: {props.json.translation_to_direction_angle_deg.max_error.toFixed(5)}</p>
                </div>

                <button className="go_back_btn" onClick={() => props.toggleMVO(false)}>Go Back</button>
            </div>
        </div>
    )
}

export default MVOSummary;