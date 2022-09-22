/* Root Component which is rendered in index.html.

Author: Adi Singh
*/
/*
import React from 'react';

// Third-Party Package Imports.
import {extend} from "react-three-fiber";
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import ReactDOM from 'react-dom';

// Local Imports.
import LandingPageGraph from './Components/LandingPageGraph';
import './index.css';

// Allows the user to orbit their view around the react three fiber point cloud.
extend({OrbitControls}); 

ReactDOM.render(
  <React.StrictMode>
    <LandingPageGraph/>
  </React.StrictMode>,

  document.getElementById('root')
);
*/

import React from 'react';
import ReactDOM from 'react-dom';

import { ReactComponent as Graph } from './ui/dot_graph_output.svg';

ReactDOM.render(
  <React.StrictMode>
    <Graph/>
  </React.StrictMode>,

  document.getElementById('root')
);

