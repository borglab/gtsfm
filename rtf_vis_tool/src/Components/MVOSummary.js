/* Component to render Multiview Optimizer Metrics. Specifically rotation averaging and
translation averaging metrics.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import '../stylesheets/MVOSummary.css'

function MVOSummary(props) {
    /*
    Args:
        props.json.rotation_averaging_angle_deg.median_error (float): Median error of rotation averaging.
        props.json.rotation_averaging_angle_deg.min_error (float): Min error of rotation averaging.
        props.json.rotation_averaging_angle_deg.max_error (float): Max error of rotation averaging.
        props.json.translation_averaging_distance.median_error (float): Median error of translation averaging.
        props.json.translation_averaging_distance.min_error (float): Min error of translation averaging.
        props.json.translation_averaging_distance.max_error (float): Max error of translation averaging.
        props.json.translation_to_direction_angle_deg.median_error (float): Median error of translation to direction.
        props.json.translation_to_direction_angle_deg.min_error (float): Min error of translation to direction.
        props.json.translation_to_direction_angle_deg.max_error (float): Max error of tranlation to direction.

    Returns:
        A component showing multiview optimizer metrics after clicking the 'Sparse Multiview Optimizer' Plate.
    */

    return (
        <div className="mvo_container">
            <h3>MultiView Optimizer Metrics</h3>
            <div className="mvo_sub_container">
                <div className="metrics_container">
                    <p className="mvo-header">Rotation Averaging Angle Metrics</p>

                    <p className="mvo-text">
                        Median Error: {props.json.rotation_averaging_angle_deg.median_error.toFixed(5)}
                    </p>

                    <p className="mvo-text">
                        Min Error: {props.json.rotation_averaging_angle_deg.min_error.toFixed(5)}
                    </p>

                    <p className="mvo-text">
                        Max Error: {props.json.rotation_averaging_angle_deg.max_error.toFixed(5)}
                    </p>
                </div>

                <div className="metrics_container">
                    <p className="mvo-header">Translation Averaging Metrics</p>

                    <p className="mvo-text">
                        Median Error: {props.json.translation_averaging_distance.median_error.toFixed(5)}
                    </p>

                    <p className="mvo-text">
                        Min Error: {props.json.translation_averaging_distance.min_error.toFixed(5)}
                    </p>

                    <p className="mvo-text">
                        Max Error: {props.json.translation_averaging_distance.max_error.toFixed(5)}
                    </p>
                </div>

                <div className="metrics_container">
                    <p className="mvo-header">Translation to Direction Angle Metrics</p>

                    <p className="mvo-text">
                        Median Error: {props.json.translation_to_direction_angle_deg.median_error.toFixed(5)}
                    </p>

                    <p className="mvo-text">
                        Min Error: {props.json.translation_to_direction_angle_deg.min_error.toFixed(5)}
                    </p>

                    <p className="mvo-text">
                        Max Error: {props.json.translation_to_direction_angle_deg.max_error.toFixed(5)}
                    </p>
                </div>

                <button className="go_back_btn" onClick={() => props.toggleMVO(false)}>Go Back</button>
            </div>
        </div>
    )
}

export default MVOSummary;