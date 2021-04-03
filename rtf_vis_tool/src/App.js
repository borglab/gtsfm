import React from "react";
import {extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import './App.css';

import DivGraph from './Components/DivGraph';
extend({OrbitControls}) //allows the user to orbit their view around the react three fiber point cloud

//Component which is rendered in index.js
const App = (props) => {
  return (
    <DivGraph/>
  )
}

export default App;