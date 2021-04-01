# GTSFM Pipeline Visualization Tool

This project was created with React for viewing metrics related to the various processes within the GTSFM pipeline. Also used to view resulting point clouds using React Three Fiber.

## Setup
1. After cloning the repository, [install Node.js](https://nodejs.org/en/download/). Node.js serves as JavaScript runtime environment which the React application will run on.

2. To verify Node's installation, in the terminal, run:
```bash
node -v
```

3. Inside the React directory, install all the Node dependencies specified within `package.json`:
```bash
cd rtf_vis_tool
npm install
```

4. Run the scene_optimizer to test GTSFM on the sample lund door dataset:
```bash
python gtsfm/runner/run_scene_optimizer.py
```
This will overwrite all the summary metric files within the `result_metrics` directory

5. Now, run the web application:
```bash
npm start
```
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.



## Repository Structure
- `node_modules`: internal packages used throughout the application. Don't edit these.
- `public`: contains index.html file 
- `src`
    - `Components`: contains all React components used for rendering graph nodes, summaries, point cloud viewers, etc...
    - `result_metrics`: folder generated from running `python gtsfm/runner/run_scene_optimizer.py`. These metrics are displayed on the application
    - `stylesheets`: css files relating to different JS Components
    - `ViewFrustum_Ported`: SE3 and View_Frustum classes rewritten in JS. Used to render camera frustums in point cloud.
- `package-lock.json` & `package.json`: lists the dependencies for this react project. Don't edit these.

## GTSFM Pipeline Graph Information

TwoViewEstimator Plate: displays frontned summary metrics  
MultiViewEstimor Plate: displays multiview optimizer metrics  
'SfMData' Node: displays the point cloud after data association
