import React from "react";
import {extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import './App.css';

import LandingPageGraph from './Components/LandingPageGraph';
extend({OrbitControls}) //allows the user to orbit their view around the react three fiber point cloud

//Component which is rendered in index.js
const App = (props) => {
  return (
    <LandingPageGraph/>
  )
}

export default App;