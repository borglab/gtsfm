/* Component to render Multiview Optimizer Metrics. Specifically rotation averaging and
translation averaging metrics. Only seen when "Sparse Multiview Optimizer" plate is clicked.

Authors: Adi Singh, Kevin Fu
*/
import React from "react";

// Local Imports.
import '../stylesheets/MVOSummary.css'

function MVOSummary(props) {
    /*
    Args:
        props.rotation_averaging_metrics.rotation_angle_error_deg.summary.median (float): Median error of rotation 
                                                        averaging.
        props.rotation_averaging_metrics.rotation_angle_error_deg.summary.min (float): Min angular error on global
                                                        rotation for any frame after rotation averaging (pre-BA).
        props.rotation_averaging_metrics.rotation_angle_error_deg.summary.max (float): Max angular error on global 
                                                        rotation for any frame after rotation averaging (pre-BA).
        props.translation_averaging_metrics.translation_error_distance.summary.median (float): Median Euclidean error of 
                                                        translation averaging.
        props.translation_averaging_metrics.translation_error_distance.summary.min (float): Min Euclidean error on global
                                                        translation for any frame after translation averaging (pre-BA).
        props.translation_averaging_metrics.translation_error_distance.summary.max (float): Max Euclidean error on global
                                                        translation for any frame after translation averaging (pre-BA).
        props.translation_averaging_metrics.translation_angle_error_deg.summary.median (float): Median error of 
                                                        translation to direction.
        props.translation_averaging_metrics.translation_angle_error_deg.summary.min (float): Min error of translation 
                                                        to direction.
        props.translation_averaging_metrics.translation_angle_error_deg.summary.max (float): Max error of tranlation to
                                                        direction.

    Returns:
        A component showing multiview optimizer metrics after clicking the 'Sparse Multiview Optimizer' Plate.
    */

    return (
        <div className="mvo_container">
            <h3>Sparse MultiView Optimizer Metrics</h3>
            <div className="mvo_sub_container">
                <div className="metrics_container">
                    <p className="mvo_header">Rotation Averaging Angle Metrics</p>

                    <p className="mvo_text">
                        Median Error: {props.rotation_averaging_metrics.rotation_angle_error_deg.summary.median.toFixed(5)}
                    </p>

                    <p className="mvo_text">
                        Min Error: {props.rotation_averaging_metrics.rotation_angle_error_deg.summary.min.toFixed(5)}
                    </p>

                    <p className="mvo_text">
                        Max Error: {props.rotation_averaging_metrics.rotation_angle_error_deg.summary.max.toFixed(5)}
                    </p>
                </div>

                <div className="metrics_container">
                    <p className="mvo_header">Translation Averaging Metrics</p>

                    <p className="mvo_text">
                        Median Error: {props.translation_averaging_metrics.translation_error_distance.summary.median.toFixed(5)}
                    </p>

                    <p className="mvo_text">
                        Min Error: {props.translation_averaging_metrics.translation_error_distance.summary.min.toFixed(5)}
                    </p>

                    <p className="mvo_text">
                        Max Error: {props.translation_averaging_metrics.translation_error_distance.summary.max.toFixed(5)}
                    </p>
                </div>

                <div className="metrics_container">
                    <p className="mvo_header">Translation to Direction Angle Metrics</p>

                    <p className="mvo_text">
                        Median Error: {props.translation_averaging_metrics.translation_angle_error_deg.summary.median.toFixed(5)}
                    </p>

                    <p className="mvo_text">
                        Min Error: {props.translation_averaging_metrics.translation_angle_error_deg.summary.min.toFixed(5)}
                    </p>

                    <p className="mvo_text">
                        Max Error: {props.translation_averaging_metrics.translation_angle_error_deg.summary.max.toFixed(5)}
                    </p>
                </div>

                <button className="go_back_btn" onClick={() => props.toggleMVO(false)}>Go Back</button>
            </div>
        </div>
    )
}

export default MVOSummary;
