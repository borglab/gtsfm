import React from "react";
import {Canvas, extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import './App.css';

extend({OrbitControls})

const App = (props) => {
  return (
    <p>boilerplate App.js body.</p>
  )
}

export default App;