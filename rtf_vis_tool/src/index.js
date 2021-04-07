import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';

//Root component that is rendered in index.html
ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);