# GTSFM Computational Graph Visualizer

This React application was created for viewing metrics related to the various processes within the GTSFM pipeline. Also used to view resulting point clouds using React Three Fiber.

## Setup
1. Complete the instructions in the `README.md` file in the root `gtsfm` directory.

2. Inside the React directory, install all the Node dependencies specified within `package.json`:
```bash
cd rtf_vis_tool
npm install --legacy-peer-deps
```
3. Now, run the web application:
```bash
npm start
```
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.  
Note: running the application for the first time will auto generate a `/node_modules` folder and `.eslintcache` file which are included in `.gitignore`.


4. Run the React Unit Tests (within `gtsfm/rtf_vis_tool`):
```bash
npm test a
```
Currently, the unit tests are written for smaller, helper components that do not involve React Three Fiber. This is because the React unit testing framework, Enzyme, does not provide native support for third party packages like React Three Fiber. Thus, it's incompatible with components that render 3D-related components.

## Repository Structure
- `node_modules`: internal packages used throughout the application. Don't edit these.
- `public`: contains index.html file 
- `src`
    - `Components`: contains all React components used for rendering graph nodes, summaries, point cloud viewers, etc. Each file within this folder returns a format JSX, which is essentially a modified version of HTML that is meant to be readable by React code
    - `result_metrics`: folder generated from running `./run`. These metrics are displayed on the application
    - `stylesheets`: css files relating to different JS Components
    - `ViewFrustum_Ported`: SE3 and View_Frustum classes rewritten in JS. Used to render camera frustums in point cloud.
    - `tests`: contains unit tests for some helper React components.
    - `index.js`: the highest level React Component. Hierarchy of Rendering is `index.html` -> `index.js` -> `LandingPageGraph.js`
- `package-lock.json` & `package.json`: lists the dependencies for this react project. Don't edit these.

## Dependency Details  

package.json
- `@testing-library/jest-dom` : auto generated when initializing react app
- `@testing-library/react` : auto generated when initializing react app
- `@testing-library/user-event` : auto generated when initializing react app
- `drei` : used to render lines and text in the react three fiber point cloud (`Data_Association_PC.js`)
- `numjs` : used to render camera frustums (`AllFrustums.js`)
- `quaternion` : used to extract rotation matrix from Quaternion (`AllFrustums.js`)
- `react` : primary react import for all files in the `Components` folder
- `react-dom` : helper library of react to perform live webpage updates upon refresh
- `react-scripts` : used to run the application when user types `npm-start`
- `react-three-fiber`: allows for 3D rendering in the browser
- `react-xarrows`: renders arrows in the landing page graph (`DivGraph.js`)
- `three` : required by the `react-three-fiber` dependency to run properly 
- `enzyme` & `enzyme-adapter-react-16` : used to write unit tests for React.js components 

package-lock.json
- Essentially, the need for this is that on top of just a general listing of dependencies (defined in `package.json`), this file goes one step deeper and describes the dependency tree along with the ordering of installation of dependencies so that the every user can run the application in exactly the same manner. Read here for more details regarding why `package-lock.json` must be checked in:
    - [NPM Docs](https://docs.npmjs.com/cli/v7/configuring-npm/package-lock-json)
    - [Stack Overflow Post](https://stackoverflow.com/questions/44552348/should-i-commit-yarn-lock-and-package-lock-json-files)

## GTSFM Pipeline Graph Information

TwoViewEstimator Plate: displays frontend summary metrics  
MultiViewEstimor Plate: displays multiview optimizer metrics  
'SfMData' Node: displays the point cloud after data association

Note: information displayed is based on state of `result_metrics` as of
07/22/22. The relevant JSON files are imported at the top of the
`LandingPageGraph` component, and then passed to the `FrontendSummary` and
`MVOSummary` components. These components are all defined in their respective
`.js` files in `src/Components`.
