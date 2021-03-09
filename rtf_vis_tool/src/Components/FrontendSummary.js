import React from "react";
import '../stylesheets/FrontendSummary.css'

const FrontendSummary = (props) => {
    return (
        <div className="fs_container">
            <div style={{position: "relative", padding: '0', width: '100%', height: '100%'}}>
                <h3>Frontend Summary Metrics</h3>
                <p style={{fontSize: '80%', marginBottom: '15%'}}>Format: (success count)/(valid entries)/(total entries)</p>

                <p className="fs-text">Rotation Success: {props.json.rotation.success_count}/{props.json.num_valid_entries}/{props.json.num_total_entries}</p>
                <p className="fs-text">Translation Success: {props.json.translation.success_count}/{props.json.num_valid_entries}/{props.json.num_total_entries}</p>
                <p className="fs-text">Pose Success: {props.json.pose.success_count}/{props.json.num_valid_entries}/{props.json.num_total_entries}</p>

                <p className="fs-text">Correspondence Inliners: {props.json.correspondences.all_inliers}</p>

                <button className="go_back_btn" onClick={() => props.toggleFS(false)}>Go Back</button>
            </div>
        </div>
    )
}

export default FrontendSummary;