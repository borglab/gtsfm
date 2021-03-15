import React from "react";
import {Canvas, extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import './App.css';

import DivGraph from './Components/DivGraph'
import PCViewer from "./Components/PCViewer";
extend({OrbitControls})

const App = (props) => {
  const showGraph = "divGraph"; //temporary display value

  if (showGraph == "divGraph") {
    return (
      <DivGraph/>
    )
  } else {
    return (
      <PCViewer/>
    )
  }
}

export default App;