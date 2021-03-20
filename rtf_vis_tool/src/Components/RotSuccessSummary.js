import React from "react";
import '../stylesheets/RotSuccessSummary.css'

const RotSuccessSummary = (props) => {
    return (
        <div className="rot_container">
            <div style={{position: "relative", padding: '0', width: '100%', height: '100%'}}>
                <p className="rot_fs-text">Rotation Success: {props.json.rotation.success_count}/{props.json.num_valid_entries}/{props.json.num_total_entries}</p>
                <button className="rot_go_back_btn" onClick={() => props.toggleRSS(false)}>Go Back</button>
            </div>
        </div>
    )
}

export default RotSuccessSummary;