/* Component that renders a single Camera Frustum given 5 vertices. Used in AllFrustums.js.

Author: Adi Singh
*/
import React from "react";

// Third-Party Package Imports.
import {Line} from "@react-three/drei";

function Frustum (props) {
    /*
    Args:
        props.v0 (array: size 3 of floats): Camera's optical center.
        props.v1 (array: size 3 of floats): First vertex in camera's far plane.
        props.v2 (array: size 3 of floats): Second vertex in camera's far plane.
        props.v3 (array: size 3 of floats): Third vertex in camera's far plane.
        props.v4 (array: size 3 of floats): Fourth vertex in camera's far plane.
        props.width (float): Width of the line segments forming the frustum.

           (v1) o-------------------o (v2)
                |                   |
                |     (v0) o        |
                |                   |
           (v4) o-------------------o (v3)

        
    Returns:
        A component rendering a camera frustum consisting of 5 vertices.
    */
    const black = "#000000"

    return (
      <mesh>
        <Line points={[props.v0,props.v1]} color={black} lineWidth={props.width}/>
        <Line points={[props.v0,props.v2]} color={black} lineWidth={props.width}/>
        <Line points={[props.v0,props.v3]} color={black} lineWidth={props.width}/>
        <Line points={[props.v0,props.v4]} color={black} lineWidth={props.width}/>
        <Line points={[props.v1,props.v2]} color={black} lineWidth={props.width}/>
        <Line points={[props.v2,props.v3]} color={black} lineWidth={props.width}/>
        <Line points={[props.v3,props.v4]} color={black} lineWidth={props.width}/>
        <Line points={[props.v4,props.v1]} color={black} lineWidth={props.width}/>
      </mesh>
    )
}

export default Frustum;
