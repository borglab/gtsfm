import React from "react";
import '../stylesheets/TranSuccessSummary.css'

const TranSuccessSummary = (props) => {
    console.log(props.json);
    return (
        <div className="tran_container">
            <div style={{position: "relative", padding: '0', width: '100%', height: '100%'}}>
                {/* <p className="tran_fs-text">Translation Success: 3/{props.json.num_valid_entries}/{props.json.num_total_entries}</p> */}
                <button className="tran_go_back_btn" onClick={() => props.toggleTSS(false)}>Go Back</button>
            </div>
        </div>
    )
}

export default TranSuccessSummary;