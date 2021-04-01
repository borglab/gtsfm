import React from "react";
import {extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import './App.css';

//allows the user to orbit their view around the react three fiber point cloud
extend({OrbitControls})

//Component which is rendered in index.js
const App = (props) => {
  return (
    <p>boilerplate App.js body.</p>
  )
}

export default App;