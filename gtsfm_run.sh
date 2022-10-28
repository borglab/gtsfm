#!/bin/bash
#!/bin/sh

function gtsfm() {
  echo 'Running GTSfM with data from' $1
  (trap 'kill 0' SIGINT; 
  npm start --prefix rtf_vis_tool & 
  python gtsfm/runner/run_scene_optimizer_olssonloader.py --dataset_root $1 --image_extension JPG --config_name sift_front_end.yaml --num_workers 4)
}
