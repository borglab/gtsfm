#!/bin/bash

##########################################################
# Run Evaluation pipeline
##########################################################

cd $GITHUB_WORKSPACE

# First run the scene optimizer, which writes metrics as JSON to the result_metrics directory. 
python gtsfm/runner/run_scene_optimizer.py

# Then read the JSON to generate plots HTML.
python gtsfm/runner/visualize_metrics.py --metrics_dir=result_metrics --output_dir=result_metrics